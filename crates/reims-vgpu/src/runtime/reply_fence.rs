//! The reply commands a child FIFO holds when its doorbell rings.
//!
//! The guest sends `CmdHeapTextureSizeAndAlign` and reads the reply as soon as
//! its doorbell write returns: it waits for no stamp and no interrupt. Apple's
//! host answers inside that write. This device drains on a worker, so a
//! doorbell that returned at once left the guest reading the sixteen zero bytes
//! it had put there, and `heapTextureSizeAndAlignWithDescriptor:` returned
//! `{0, 0}` for most calls. On macOS 27.2 RenderBox sizes its texture heaps from
//! that answer, logged `unable to create heap: 0`, and iconservicesagent drew
//! no icon: the Dock showed placeholders.
//!
//! So the vCPU that rings a doorbell looks, with no device lock, for such
//! commands among the packets the ring holds. The query changes nothing in the
//! device, so the vCPU answers it there and then, ahead of whatever the ring
//! holds before it; a channel held behind a shader translation would otherwise
//! keep the guest waiting for as long as the translation takes. Everything
//! here is read from guest memory the way the drain reads it.

use crate::contract::endian::{ld16, ld32};
use crate::model::{
    child_reg_block_offset, pfn_to_gpa, published_byte_count, CHILD_OP_HEAP_TEXTURE_SIZE_AND_ALIGN,
    CHILD_REG_BASE_PFN, CHILD_REG_HEAD, CHILD_REG_TAIL, CHILD_RING_PFN_ENTRY_LEN,
    PACKET_HEADER_LEN, PACKET_OPCODE, PACKET_STAMP_COUNT, PACKET_STAMP_LEN, PACKET_TOTAL_SIZE,
};
use crate::runtime::host::{read_u32, HostMemory};

/// A reply command is a few dozen bytes. One that claims more is not read.
const MAX_QUERY_LEN: u32 = 1024;

/// A child ring as its page list describes it, kept by the caller between
/// doorbells so the list is read once per ring.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RingPages {
    pub root_page: u32,
    pub base_pfn: u32,
    pub page_gpas: Vec<u64>,
}

/// One queued reply command: where its packet starts, which names it to the
/// drain, and its payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    pub head: u32,
    pub payload: Vec<u8>,
}

/// The reply commands between a channel's head and tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pending {
    pub queries: Vec<Query>,
    head_gpa: u64,
    after: u32,
}

impl Pending {
    /// Whether the drain has run every command in [`Self::queries`]. An
    /// unreadable head counts as drained: there is nothing left to wait on.
    pub fn drained<M: HostMemory>(&self, mem: &M) -> bool {
        match read_u32(mem, self.head_gpa) {
            Ok(head) => head.wrapping_sub(self.after) as i32 >= 0,
            Err(_) => true,
        }
    }
}

/// Find the reply commands between `channel`'s head and tail.
///
/// `None` when the ring holds none, which is nearly every doorbell, or when the
/// ring cannot be read; the drain reports a ring it cannot read.
pub fn find<M: HostMemory>(
    mem: &M,
    root_page: u32,
    page_shift: u32,
    channel: u32,
    cache: &mut RingPages,
) -> Option<Pending> {
    let (queries, head_gpa, after) = scan(
        mem,
        root_page,
        page_shift,
        channel,
        cache,
        CHILD_OP_HEAP_TEXTURE_SIZE_AND_ALIGN,
    )?;
    (!queries.is_empty()).then_some(Pending {
        queries,
        head_gpa,
        after,
    })
}

/// Every `opcode` packet between `channel`'s head and tail that the drain has
/// not run yet, oldest first. Read from guest memory and nothing else, like
/// [`find`].
pub fn queued<M: HostMemory>(
    mem: &M,
    root_page: u32,
    page_shift: u32,
    channel: u32,
    opcode: u16,
) -> Vec<Query> {
    scan(
        mem,
        root_page,
        page_shift,
        channel,
        &mut RingPages::default(),
        opcode,
    )
    .map(|(queries, _, _)| queries)
    .unwrap_or_default()
}

fn scan<M: HostMemory>(
    mem: &M,
    root_page: u32,
    page_shift: u32,
    channel: u32,
    cache: &mut RingPages,
    opcode: u16,
) -> Option<(Vec<Query>, u64, u32)> {
    if root_page == 0 {
        return None;
    }
    let regs_gpa = pfn_to_gpa(root_page, page_shift) + child_reg_block_offset(channel)?;
    let head = read_u32(mem, regs_gpa + CHILD_REG_HEAD).ok()?;
    let tail = read_u32(mem, regs_gpa + CHILD_REG_TAIL).ok()?;
    if head == tail {
        return None;
    }
    let base_pfn = read_u32(mem, regs_gpa + CHILD_REG_BASE_PFN).ok()?;
    if base_pfn == 0 {
        return None;
    }
    let page_size = 1u64 << page_shift;
    if cache.root_page != root_page || cache.base_pfn != base_pfn || cache.page_gpas.is_empty() {
        let mut list = vec![0u8; page_size as usize];
        mem.read_gpa(pfn_to_gpa(base_pfn, page_shift), &mut list)
            .ok()?;
        *cache = RingPages {
            root_page,
            base_pfn,
            page_gpas: list
                .chunks_exact(CHILD_RING_PFN_ENTRY_LEN as usize)
                .map(ld32)
                .take_while(|&pfn| pfn != 0)
                .map(|pfn| pfn_to_gpa(pfn, page_shift))
                .collect(),
        };
    }
    let length = u32::try_from(cache.page_gpas.len() as u64 * page_size).ok()?;
    let mut remaining = published_byte_count(head, tail, length)?;
    let mut at = head;
    let mut queries = Vec::new();
    let mut after = head;
    while remaining >= PACKET_HEADER_LEN {
        let header = read_ring(
            mem,
            &cache.page_gpas,
            length,
            at,
            PACKET_HEADER_LEN,
            page_shift,
        )?;
        let total = ld32(&header[PACKET_TOTAL_SIZE..]);
        if total < PACKET_HEADER_LEN || total > remaining {
            break;
        }
        let payload_at =
            PACKET_HEADER_LEN + u32::from(ld16(&header[PACKET_STAMP_COUNT..])) * PACKET_STAMP_LEN;
        if ld16(&header[PACKET_OPCODE..]) == opcode
            && payload_at <= total
            && total <= MAX_QUERY_LEN
        {
            let packet = read_ring(mem, &cache.page_gpas, length, at, total, page_shift)?;
            queries.push(Query {
                head: at,
                payload: packet[payload_at as usize..].to_vec(),
            });
            after = at.wrapping_add(total);
        }
        at = at.wrapping_add(total);
        remaining -= total;
    }
    Some((queries, regs_gpa + CHILD_REG_HEAD, after))
}

fn read_ring<M: HostMemory>(
    mem: &M,
    page_gpas: &[u64],
    length: u32,
    at: u32,
    len: u32,
    page_shift: u32,
) -> Option<Vec<u8>> {
    let page_size = 1u64 << page_shift;
    let mut out = vec![0u8; len as usize];
    for (i, byte) in out.iter_mut().enumerate() {
        let off = u64::from(at.wrapping_add(i as u32) % length);
        let page = *page_gpas.get((off >> page_shift) as usize)?;
        mem.read_gpa(page + (off & (page_size - 1)), std::slice::from_mut(byte))
            .ok()?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{PACKET_COMPLETION_STAMP, PAGE_SHIFT_ARM64E};
    use crate::runtime::host::FakeHost;

    const ROOT_PFN: u32 = 0x10;
    const LIST_PFN: u32 = 0x11;
    const RING_PFN: u32 = 0x12;
    const CHANNEL: u32 = 2;
    const PAGE: u32 = 1 << PAGE_SHIFT_ARM64E;
    const DRAW: u16 = 0x22;

    fn gpa(pfn: u32) -> u64 {
        pfn_to_gpa(pfn, PAGE_SHIFT_ARM64E)
    }

    fn regs() -> u64 {
        gpa(ROOT_PFN) + child_reg_block_offset(CHANNEL).unwrap()
    }

    fn packet(opcode: u16, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; PACKET_HEADER_LEN as usize];
        out[PACKET_OPCODE..PACKET_OPCODE + 2].copy_from_slice(&opcode.to_le_bytes());
        let total = PACKET_HEADER_LEN + payload.len() as u32;
        out[PACKET_TOTAL_SIZE..PACKET_TOTAL_SIZE + 4].copy_from_slice(&total.to_le_bytes());
        out[PACKET_COMPLETION_STAMP..PACKET_COMPLETION_STAMP + 4]
            .copy_from_slice(&7u32.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    /// A one-page ring on `CHANNEL` holding `packets` from `head` on.
    fn ring(head: u32, packets: &[Vec<u8>]) -> FakeHost {
        let mut host = FakeHost::new();
        for pfn in [ROOT_PFN, LIST_PFN, RING_PFN] {
            host.map_range(gpa(pfn), PAGE as usize, 0);
        }
        host.put_u32(gpa(LIST_PFN), RING_PFN);
        let mut at = head;
        for packet in packets {
            for byte in packet {
                host.write_gpa(gpa(RING_PFN) + u64::from(at % PAGE), &[*byte])
                    .unwrap();
                at = at.wrapping_add(1);
            }
        }
        host.put_u32(regs() + CHILD_REG_HEAD, head);
        host.put_u32(regs() + CHILD_REG_TAIL, at);
        host.put_u32(regs() + CHILD_REG_BASE_PFN, LIST_PFN);
        host
    }

    fn find_in(host: &FakeHost) -> Option<Pending> {
        find(
            host,
            ROOT_PFN,
            PAGE_SHIFT_ARM64E,
            CHANNEL,
            &mut RingPages::default(),
        )
    }

    #[test]
    fn a_ring_without_a_reply_command_holds_nothing_to_answer() {
        assert_eq!(find_in(&ring(0x40, &[])), None);
        assert_eq!(find_in(&ring(0x40, &[packet(DRAW, &[1; 20])])), None);
    }

    #[test]
    fn a_reply_command_behind_another_packet_is_found_with_its_payload() {
        let query = [0xabu8; 64];
        let host = ring(
            0x40,
            &[
                packet(DRAW, &[1; 20]),
                packet(CHILD_OP_HEAP_TEXTURE_SIZE_AND_ALIGN, &query),
                packet(DRAW, &[2; 8]),
            ],
        );
        let pending = find_in(&host).expect("the query is in the ring");
        assert_eq!(
            pending.queries,
            vec![Query {
                head: 0x40 + 32,
                payload: query.to_vec(),
            }]
        );
    }

    #[test]
    fn the_ring_is_drained_once_the_head_passes_the_last_reply_command() {
        let query = packet(CHILD_OP_HEAP_TEXTURE_SIZE_AND_ALIGN, &[3; 64]);
        let after = 0x40 + query.len() as u32;
        let mut host = ring(0x40, &[query, packet(DRAW, &[2; 8])]);
        let pending = find_in(&host).expect("the query is in the ring");
        assert!(!pending.drained(&host));
        host.put_u32(regs() + CHILD_REG_HEAD, after - 1);
        assert!(!pending.drained(&host));
        host.put_u32(regs() + CHILD_REG_HEAD, after);
        assert!(pending.drained(&host));
    }

    #[test]
    fn a_reply_command_that_wraps_the_ring_is_read_whole() {
        let query: Vec<u8> = (0..64).collect();
        let head = PAGE - 20;
        let host = ring(
            head,
            &[packet(CHILD_OP_HEAP_TEXTURE_SIZE_AND_ALIGN, &query)],
        );
        let pending = find_in(&host).expect("the query is in the ring");
        assert_eq!(
            pending.queries,
            vec![Query {
                head,
                payload: query
            }]
        );
    }

    #[test]
    fn a_packet_longer_than_the_ring_holds_ends_the_walk() {
        let mut host = ring(0x40, &[packet(DRAW, &[1; 20])]);
        // A total size past the tail: the producer has not published it.
        host.put_u32(gpa(RING_PFN) + 0x40 + PACKET_TOTAL_SIZE as u64, 0x1000);
        assert_eq!(find_in(&host), None);
    }
}
