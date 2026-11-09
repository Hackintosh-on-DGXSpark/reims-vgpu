//! A fake 27.2 kernel and mapping in guest memory: the counts it reports, the
//! layouts it refuses, and the guard.
//!
//! `FakeHost` reads a kernel address from its sparse store at the address
//! itself when no CPU translation is scripted, which is what a vCPU read is
//! here. `kva_no_cpu` models the worker before the kernel walk validates.

use super::*;
use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
use crate::runtime::host::FakeHost;
use crate::runtime::kernel_walk::tests::{root, SCTLR_ON, TCR_16K_47};

const PAGE: u64 = 1 << 14;
/// The kernel's Mach-O header, and its segments after it.
const KERNEL: u64 = 0xffff_fe00_1000_0000;
const TEXT_CONST: u64 = KERNEL + 0x1000;
const DATA_CONST: u64 = KERNEL + 2 * PAGE;
const TEXT_EXEC: u64 = KERNEL + 3 * PAGE;
/// `VBAR_EL1`, at the start of `__TEXT_EXEC` as on 27.2, several pages up.
const VBAR: u64 = TEXT_EXEC;
/// The descriptors' class, inside `__DATA_CONST`, signed.
const VTABLE_SIGNED: u64 = 0x5b95_7e00_1000_8638;

const MID: u32 = 2;
const MI: u64 = 0xffff_fe1e_47e6_4c20;
const SURFACE: u64 = 0xffff_fe24_13d7_1398;
const TABLE: u64 = 0xffff_fe24_13d7_2048;
const MIXED: u64 = 0xffff_fe1e_47da_8100;
const PAGE_TABLE: u64 = 0xffff_fe9a_f64f_c000;
const KERNEL_TASK: u64 = 0xffff_fe1f_4a04_9c70;
/// The mapper device every MappingInternal points back to.
const MAPPER: u64 = 0xffff_fe2a_e214_1800;
/// Mapping 2's guest pages, apart so each is its own run.
const G: [u64; 3] = [0x1_0000_0000, 0x1_0010_0000, 0x1_0020_0000];

/// One type-11 mapping as 27.2 lays it out in guest memory.
struct Surface {
    mid: u32,
    mi: u64,
    surface: u64,
    table: u64,
    page_table: u64,
    gpas: Vec<u64>,
}

fn old_surface() -> Surface {
    Surface {
        mid: MID,
        mi: MI,
        surface: SURFACE,
        table: TABLE,
        page_table: PAGE_TABLE,
        gpas: G.to_vec(),
    }
}

fn entry(gpa: u64) -> u32 {
    ((gpa >> 14) as u32) << iosurface_pages::PAGE_ENTRY_PFN_SHIFT
        | iosurface_pages::PAGE_ENTRY_VALID
}

fn put64(host: &mut FakeHost, at: u64, v: u64) {
    host.write_gpa(at, &v.to_le_bytes()).unwrap();
}

fn put32(host: &mut FakeHost, at: u64, v: u32) {
    host.write_gpa(at, &v.to_le_bytes()).unwrap();
}

fn segment(name: &[u8], addr: u64, len: u64, sections: &[(&[u8], u64, u64)]) -> Vec<u8> {
    let mut b = Vec::new();
    let size = SEGMENT_COMMAND_64_LEN + sections.len() * SECTION_64_LEN;
    b.extend_from_slice(&LC_SEGMENT_64.to_le_bytes());
    b.extend_from_slice(&(size as u32).to_le_bytes());
    let mut n = [0u8; 16];
    n[..name.len()].copy_from_slice(name);
    b.extend_from_slice(&n);
    b.extend_from_slice(&addr.to_le_bytes());
    b.extend_from_slice(&len.to_le_bytes());
    b.extend_from_slice(&[0u8; 24]);
    b.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    for (sect, sa, sl) in sections {
        let mut s = [0u8; SECTION_64_LEN];
        s[..sect.len()].copy_from_slice(sect);
        s[16..16 + name.len()].copy_from_slice(name);
        s[32..40].copy_from_slice(&sa.to_le_bytes());
        s[40..48].copy_from_slice(&sl.to_le_bytes());
        b.extend_from_slice(&s);
    }
    b
}

/// A kernel image whose `__TEXT,__const` holds `version`.
fn put_kernel(host: &mut FakeHost, version: &str) {
    let mut cmds = segment(
        b"__TEXT",
        KERNEL,
        2 * PAGE,
        &[(b"__const", TEXT_CONST, 0x200)],
    );
    cmds.extend(segment(b"__DATA_CONST", DATA_CONST, PAGE, &[]));
    cmds.extend(segment(b"__TEXT_EXEC", TEXT_EXEC, PAGE, &[]));
    let mut head = Vec::new();
    for word in [
        MH_MAGIC_64,
        CPU_TYPE_ARM64,
        0xc000_0002,
        MH_EXECUTE,
        3,
        cmds.len() as u32,
        0,
        0,
    ] {
        head.extend_from_slice(&word.to_le_bytes());
    }
    host.write_gpa(KERNEL, &head).unwrap();
    host.write_gpa(KERNEL + MACH_HEADER_64_LEN as u64, &cmds)
        .unwrap();
    let mut text = b"unrelated\0".to_vec();
    text.extend_from_slice(version.as_bytes());
    text.push(0);
    host.write_gpa(TEXT_CONST + 0x100, &text).unwrap();
}

fn put_iomd(host: &mut FakeHost, at: u64, length: u64, wire: u32, pages: u32, buffer: u64) {
    put64(host, at, VTABLE_SIGNED);
    put64(host, at + IOMD_LENGTH, length);
    put64(host, at + IOMD_TASK, KERNEL_TASK);
    put32(host, at + IOMD_WIRE_COUNT, wire);
    host.write_gpa(at + IOMD_INITIALIZED, &[1]).unwrap();
    put64(host, at + IOMD_MEMORY_ENTRIES, MIXED);
    put32(host, at + IOMD_PAGES, pages);
    put64(host, at + IOBMD_BUFFER, buffer);
}

/// Mapping 2 as 27.2 lays it out: three pages, surface wired once, page table
/// twice (it is wired at creation as well).
/// A MappingInternal the mapper can resolve and the watch can check: its
/// surface wired `wire` times, its page table twice (it is wired at creation as
/// well), its page table in kernel memory and its pages in guest RAM.
fn put_surface(host: &mut FakeHost, s: &Surface, wire: u32) {
    let n = s.gpas.len() as u64;
    put32(host, s.mi + MI_STATE, 5);
    put64(
        host,
        s.mi + iosurface_pages::MAPPING_INTERNAL_BACKPTR,
        MAPPER,
    );
    put64(host, s.mi + MI_SURFACE_IOMD, s.surface);
    put32(host, s.mi + MI_ID, s.mid);
    put32(host, s.mi + MI_SIZE, MAPPING_INTERNAL_EXPECTED_SIZE);
    put64(host, s.mi + MI_TABLE_IOMD, s.table);
    put64(host, s.mi + MI_PAGE_COUNT, n);
    put_iomd(host, s.surface, n * PAGE, wire, n as u32, 0);
    put_iomd(host, s.table, PAGE, 2, 1, s.page_table);
    for (i, &gpa) in s.gpas.iter().enumerate() {
        put32(host, s.page_table + 4 * i as u64, entry(gpa));
        host.map_range(gpa, PAGE as usize, 0);
    }
    put64(host, MIXED + MIXED_DATA_LENGTH, 0x2000);
}

/// The device state a resolve of `s` leaves behind.
fn adopt(state: &mut DeviceState, s: &Surface) {
    state.map_surface(s.mid);
    let m = state.mappings.get_mut(&s.mid).unwrap();
    m.mapping_internal = s.mi;
    m.page_table_kva = s.page_table;
    m.page_entries = s.gpas.iter().map(|&g| entry(g)).collect();
}

fn set_surface_wire(host: &mut FakeHost, wire: i32) {
    put32(host, SURFACE + IOMD_WIRE_COUNT, wire as u32);
}

fn fixture(version: &str) -> (DeviceState, FakeHost) {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    put_kernel(&mut host, version);
    put_surface(&mut host, &old_surface(), 1);
    state.mapper_device_kva = MAPPER;
    state.kernel_walk.root = Some(KernelRoot {
        tcr: TCR_16K_47,
        sctlr: SCTLR_ON,
        vbar: Some(VBAR),
        ..root(0)
    });
    adopt(&mut state, &old_surface());
    (state, host)
}

fn lines_with(cap: &crate::observe::FailCapture, needle: &str) -> Vec<String> {
    cap.lines()
        .into_iter()
        .filter(|l| l.contains(needle))
        .collect()
}

/// The event lines of one kind, not the history lines that repeat them.
fn ev_lines(cap: &crate::observe::FailCapture, kind: &str) -> Vec<String> {
    let needle = format!(" ev={kind} ");
    cap.lines()
        .into_iter()
        .filter(|l| l.starts_with("OFF wire_watch_ev ") && l.contains(&needle))
        .collect()
}

/// [`before_write`] as a writer that starts a new frame: no read in the last
/// [`WRITE_RESAMPLE_US`] to lean on.
fn write_now(
    state: &mut DeviceState,
    host: &FakeHost,
    mapping_id: u32,
    site: &str,
    kind: WriteKind,
) -> bool {
    for w in state.wire_watch.watched.values_mut() {
        w.sampled_us = None;
    }
    before_write(state, host, mapping_id, site, kind)
}

/// The swap the display channel would carry, as the drain hands it over.
fn swap(state: &mut DeviceState, host: &FakeHost, stamp: u32) {
    let mut payload = vec![0u8; 40];
    payload[8..12].copy_from_slice(&MID.to_le_bytes());
    before_swap(state, host);
    on_present(state, "delivered");
    after_swap(
        state,
        host,
        SwapPacket {
            channel: 4,
            head: 0x40,
            slot: 4,
            opcode: crate::model::CHILD_OP_DISPLAY_SWAP,
            stamp,
            waits: &[],
            payload: &payload,
            mapping: Some(MID),
        },
        now_us(),
        SwapHow::Exec,
    );
}

#[test]
fn a_checked_mapping_reports_every_count_change_and_its_first_drop_once() {
    with_switches(true, false, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        let kernel = lines_with(&cap, "wire_watch_kernel verified");
        assert_eq!(kernel.len(), 1, "{:?}", cap.lines());
        assert!(
            kernel[0].contains(&format!("header={KERNEL:#x}"))
                && kernel[0].contains(&format!("version=\"{EXPECTED_KERNEL_VERSION}\"")),
            "{}",
            kernel[0]
        );
        let map = lines_with(&cap, "wire_watch map ");
        assert_eq!(map.len(), 1, "{:?}", cap.lines());
        assert!(
            map[0].contains("surface_wire=1 table_wire=2") && map[0].contains("at=map"),
            "{}",
            map[0]
        );

        // The hold that parks the display channel, then the first swap after it.
        on_hold(&mut state, 0x1c, 0x1c, 0x2);
        on_hold_release(&mut state, 0x1c);
        swap(&mut state, &host, 0x11);
        on_stamp(&mut state, &host, 4, 0x11, false);
        let stamp = ev_lines(&cap, "stamp");
        assert_eq!(stamp.len(), 1);
        assert!(
            stamp[0].contains("swaps=[") && stamp[0].contains(":2]"),
            "{}",
            stamp[0]
        );

        // The guest completes the surface once too often.
        set_surface_wire(&mut host, 0);
        assert!(write_now(
            &mut state,
            &host,
            MID,
            "render_store",
            WriteKind::Cpu
        ));
        let transition = lines_with(&cap, "wire_watch transition");
        assert_eq!(transition.len(), 1, "{:?}", cap.lines());
        assert!(
            transition[0].contains("field=surface old=1 new=0 map=1")
                && transition[0].contains("at=render_store")
                && transition[0].contains("last_swap={seq=")
                && transition[0].contains("last_stamp={seq="),
            "{}",
            transition[0]
        );
        let anomaly = lines_with(&cap, "wire_watch first_anomaly ");
        assert_eq!(anomaly.len(), 1);
        assert!(
            anomaly[0].contains("field=surface value=0 map=1")
                && anomaly[0].contains("boot_first=1")
                && anomaly[0].contains("iomd_fields=unchanged")
                && anomaly[0].contains("md_len=0x2000"),
            "{}",
            anomaly[0]
        );
        let ring = lines_with(&cap, "wire_watch first_anomaly_ring");
        let events: Vec<&str> = ring
            .iter()
            .map(|l| l.split(" ev=").nth(1).unwrap().split(' ').next().unwrap())
            .collect();
        assert_eq!(
            events,
            ["map", "hold", "hold_release", "present", "swap", "stamp"]
        );
        let after = lines_with(&cap, "wire_watch write_after_anomaly");
        assert_eq!(
            after.len(),
            1,
            "the write that found the drop is the first after it"
        );

        // Further moves are transitions; the anomaly is said once per field.
        set_surface_wire(&mut host, -1);
        write_now(&mut state, &host, MID, "render_store", WriteKind::Cpu);
        set_surface_wire(&mut host, 1);
        write_now(&mut state, &host, MID, "render_store", WriteKind::Cpu);
        assert_eq!(lines_with(&cap, "wire_watch transition").len(), 3);
        assert_eq!(lines_with(&cap, "wire_watch first_anomaly ").len(), 1);
        assert!(lines_with(&cap, "wire_watch transition")
            .iter()
            .any(|l| l.contains("old=0 new=-1")));

        on_unmap(&mut state, &host, MID);
        assert_eq!(lines_with(&cap, "wire_watch unmap").len(), 1);
        assert!(state.wire_watch.watched.is_empty());
    });
}

#[test]
fn a_table_count_below_its_map_value_is_an_anomaly_even_above_zero() {
    with_switches(true, false, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        put32(&mut host, TABLE + IOMD_WIRE_COUNT, 1);
        write_now(&mut state, &host, MID, "mapping_write", WriteKind::Cpu);
        let anomaly = lines_with(&cap, "wire_watch first_anomaly ");
        assert_eq!(anomaly.len(), 1, "{:?}", cap.lines());
        assert!(
            anomaly[0].contains("field=table value=1 map=2"),
            "{}",
            anomaly[0]
        );
    });
}

#[test]
fn another_kernel_turns_the_watch_off_and_nothing_else_is_said() {
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(
            "Darwin Kernel Version 26.6.0: Mon Jul 6 00:00:00 PDT 2026; \
             root:xnu-12377.1.1~1/RELEASE_ARM64_VMAPPLE",
        );
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        let off = cap.one("wire_watch_layout_unverified");
        assert!(
            off.contains("scope=kernel reason=kernel_version_mismatch") && off.contains("26.6.0"),
            "{off}"
        );
        set_surface_wire(&mut host, 0);
        assert!(write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
        swap(&mut state, &host, 1);
        tick(&mut state, &host);
        assert_eq!(cap.lines().len(), 1, "{:?}", cap.lines());
    });
}

#[test]
fn a_kernel_without_its_header_below_vbar_is_not_believed() {
    with_switches(true, false, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        // `__TEXT` that does not start at the header is a header for some other
        // image than the one running.
        let mut cmds = segment(b"__TEXT", KERNEL + PAGE, PAGE, &[]);
        cmds.extend(segment(b"__TEXT_EXEC", TEXT_EXEC, PAGE, &[]));
        host.write_gpa(KERNEL + MACH_HEADER_64_LEN as u64, &cmds)
            .unwrap();
        put32(&mut host, KERNEL + 16, 2);
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        assert!(cap
            .one("wire_watch_layout_unverified")
            .contains("reason=text_not_at_header"));
        assert!(state.wire_watch.off());
    });
}

/// The check a broken fixture should fail, and how to break it.
type Breakage = (&'static str, fn(&mut FakeHost));

#[test]
fn a_mapping_whose_layout_disagrees_is_not_watched() {
    with_switches(true, true, || {
        let cases: [Breakage; 4] = [
            ("surface_length", |h| {
                put64(h, SURFACE + IOMD_LENGTH, 2 * PAGE)
            }),
            ("mi_page_count", |h| put64(h, MI + MI_PAGE_COUNT, 4)),
            ("table_buffer", |h| {
                put64(h, TABLE + IOBMD_BUFFER, PAGE_TABLE + PAGE)
            }),
            ("iomd_class", |h| put64(h, TABLE, 0x5b95_7e00_2000_0638)),
        ];
        for (reason, break_it) in cases {
            let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
            break_it(&mut host);
            let cap = crate::observe::FailCapture::start();
            on_map(&mut state, &host, MID);
            let refused = cap.one("wire_watch_layout_unverified");
            assert!(
                refused.contains("scope=mapping mapping=2")
                    && refused.contains(&format!("reason={reason} ")),
                "{reason}: {refused}"
            );
            assert!(lines_with(&cap, "wire_watch map ").is_empty());
            set_surface_wire(&mut host, 0);
            assert!(
                write_now(&mut state, &host, MID, "mapping_write", WriteKind::Cpu),
                "{reason}: an unchecked count must never refuse a write"
            );
            on_map(&mut state, &host, MID);
            assert!(lines_with(&cap, "wire_watch transition").is_empty());
            assert_eq!(
                lines_with(&cap, "wire_watch_layout_unverified").len(),
                1,
                "said once"
            );
        }
    });
}

#[test]
fn a_count_that_cannot_be_read_is_neither_a_transition_nor_a_refusal() {
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        // The worker, with no validated walk to read through.
        host.kva_no_cpu = true;
        set_surface_wire(&mut host, 0);
        assert!(write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
        assert!(lines_with(&cap, "wire_watch transition").is_empty());
        assert!(state.wire_watch.cost.unreadable > 0);
        host.kva_no_cpu = false;
        assert!(!write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
        assert_eq!(lines_with(&cap, "wire_watch transition").len(), 1);
    });
}

#[test]
fn the_guard_condemns_at_the_first_drop_and_never_lets_go() {
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        assert!(write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
        assert!(write_now(
            &mut state,
            &host,
            MID,
            "gpu_writeback",
            WriteKind::Gpu
        ));
        set_surface_wire(&mut host, 0);
        assert!(!write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
        let condemned = cap.one("wire_guard_condemned");
        assert!(
            condemned.contains("mapping=2 field=surface") && condemned.contains("debt=none"),
            "{condemned}"
        );
        assert!(lines_with(&cap, "wire_watch first_anomaly ")[0].contains("guard=condemned"));
        let refused = cap.one("wire_guard_refused");
        assert!(
            refused.contains("mapping=2 site=mapping_write kind=Cpu surface_wire=0"),
            "{refused}"
        );
        assert!(
            !write_now(&mut state, &host, MID, "gpu_writeback", WriteKind::Gpu),
            "a GPU copy into a condemned incarnation is refused too"
        );
        // The guest prepares it again; the incarnation stays condemned.
        set_surface_wire(&mut host, 1);
        assert!(!write_now(
            &mut state,
            &host,
            MID,
            "wbdebt_pay",
            WriteKind::Cpu
        ));
        assert!(cap
            .one("wire_guard_still_condemned")
            .contains("mapping=2 field=surface value=1"));
        assert_eq!(state.wire_watch.cost.guard_refused, 3);
        assert_eq!(state.wire_watch.cost.condemned, 1);
        // Another mapping is none of the guard's business.
        assert!(write_now(
            &mut state,
            &host,
            7,
            "mapping_write",
            WriteKind::Cpu
        ));
        // A new incarnation under the same id starts clean.
        on_unmap(&mut state, &host, MID);
        state.mappings.get_mut(&MID).unwrap().mapping_internal = MI;
        on_map(&mut state, &host, MID);
        assert!(write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
    });
    with_switches(true, false, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        set_surface_wire(&mut host, 0);
        assert!(
            write_now(&mut state, &host, MID, "mapping_write", WriteKind::Cpu),
            "the watch alone never refuses"
        );
        assert!(lines_with(&cap, "wire_guard").is_empty());
        assert!(lines_with(&cap, "wire_watch first_anomaly ")[0].contains("guard=off"));
    });
}

#[test]
fn a_mapping_checked_after_map_reports_a_count_already_at_zero() {
    with_switches(true, false, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        // `MAP` arrived before the page list resolved.
        let entries = std::mem::take(&mut state.mappings.get_mut(&MID).unwrap().page_entries);
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        assert!(lines_with(&cap, "wire_watch map ").is_empty());
        set_surface_wire(&mut host, 0);
        state.mappings.get_mut(&MID).unwrap().page_entries = entries;
        tick(&mut state, &host);
        let map = lines_with(&cap, "wire_watch map ");
        assert_eq!(map.len(), 1, "{:?}", cap.lines());
        assert!(map[0].contains("at=late"), "{}", map[0]);
        let anomaly = lines_with(&cap, "wire_watch first_anomaly ");
        assert_eq!(anomaly.len(), 1, "{:?}", cap.lines());
        assert!(anomaly[0].contains("value=0 map=?") && anomaly[0].contains("at=late_check"));
    });
}

#[test]
fn a_reattached_internal_retires_the_watch() {
    with_switches(true, false, || {
        let (mut state, host) = fixture(EXPECTED_KERNEL_VERSION);
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        state.mappings.get_mut(&MID).unwrap().mapping_internal = MI + 0x200;
        write_now(&mut state, &host, MID, "mapping_write", WriteKind::Cpu);
        assert_eq!(
            lines_with(
                &cap,
                "wire_watch retired mapping=2 reason=internal_reattached"
            )
            .len(),
            1
        );
        assert!(state.wire_watch.watched.is_empty());
    });
}

#[test]
fn off_by_default_it_reads_nothing_and_says_nothing() {
    with_switches(false, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        set_surface_wire(&mut host, 0);
        swap(&mut state, &host, 1);
        on_stamp(&mut state, &host, 4, 1, false);
        tick(&mut state, &host);
        assert!(write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
        assert!(cap.lines().is_empty(), "{:?}", cap.lines());
        assert!(state.wire_watch.watched.is_empty());
    });
}

/// A stamp that has not reached a swap's value does not complete it, and
/// one past it completes every earlier swap on that slot.
#[test]
fn a_stamp_completes_only_the_swaps_it_reaches() {
    with_switches(true, false, || {
        let (mut state, host) = fixture(EXPECTED_KERNEL_VERSION);
        let cap = crate::observe::FailCapture::start();
        on_map(&mut state, &host, MID);
        swap(&mut state, &host, 0x20);
        swap(&mut state, &host, 0x21);
        on_stamp(&mut state, &host, 4, 0x1f, false);
        on_stamp(&mut state, &host, 3, 0x30, false);
        assert!(ev_lines(&cap, "stamp").is_empty());
        on_stamp(&mut state, &host, 4, 0x21, true);
        let stamp = ev_lines(&cap, "stamp").remove(0);
        assert!(
            stamp.contains("route=queued") && stamp.matches(":2").count() == 2,
            "{stamp}"
        );
        assert!(state.wire_watch.unstamped.is_empty());
    });
}

// MARK: - Through the FIFO

const CHANNEL: u32 = 4;
const SWAP_STAMP: u32 = 0x55;
/// The display FIFO's page list and the ring page it names.
const LIST_PFN: u32 = 0x20;
const RING_PFN: u32 = 0x30;

/// The display channel's FIFO holding one `CmdDisplaySwapMapping` for mapping
/// 2, as the guest publishes it.
fn put_display_fifo(state: &mut DeviceState, host: &mut FakeHost) -> (u64, Vec<u8>) {
    use crate::model::{
        child_reg_block_offset, CHILD_OP_DISPLAY_SWAP, CHILD_REG_BASE_PFN, CHILD_REG_HEAD,
        CHILD_REG_STAMP_INDEX, CHILD_REG_TAIL,
    };
    let page = state.page_size() as usize;
    let (root_pfn, list_pfn, ring_pfn, stamp_pfn) = (0x10u32, LIST_PFN, RING_PFN, 0x40u32);
    for pfn in [root_pfn, list_pfn, ring_pfn, stamp_pfn] {
        host.map_range(state.pfn_gpa(pfn), page, 0);
    }
    let mut payload = vec![0u8; 40];
    payload[8..12].copy_from_slice(&MID.to_le_bytes());
    payload[16..20].copy_from_slice(&0x0010_4000u32.to_le_bytes());
    let mut packet = Vec::new();
    packet.extend_from_slice(&CHILD_OP_DISPLAY_SWAP.to_le_bytes());
    packet.extend_from_slice(&0u16.to_le_bytes());
    packet.extend_from_slice(&(12 + payload.len() as u32).to_le_bytes());
    packet.extend_from_slice(&SWAP_STAMP.to_le_bytes());
    packet.extend_from_slice(&payload);
    let ring_gpa = state.pfn_gpa(ring_pfn);
    host.write_gpa(ring_gpa, &packet).unwrap();
    host.put_u32(state.pfn_gpa(list_pfn), ring_pfn);
    let regs = state.pfn_gpa(root_pfn) + child_reg_block_offset(CHANNEL).unwrap();
    host.put_u32(regs + CHILD_REG_TAIL, packet.len() as u32);
    host.put_u32(regs + CHILD_REG_HEAD, 0);
    host.put_u32(regs + CHILD_REG_STAMP_INDEX, CHANNEL);
    host.put_u32(regs + CHILD_REG_BASE_PFN, list_pfn);
    state.gfx.root_page = root_pfn;
    state.gfx.fifo_base_page = stamp_pfn;
    state.child_rings[CHANNEL as usize] = crate::model::ChannelRing {
        valid: true,
        base_pfn: list_pfn,
        length: page as u32,
        page_gpas: vec![ring_gpa],
    };
    (regs, packet)
}

/// The drop found right after the swap that caused it, with that swap first
/// in the history, and the stamp that completed it named after.
#[test]
fn a_drop_after_a_display_swap_is_aligned_with_that_swap_and_its_stamp() {
    with_switches(true, false, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        let (regs, packet) = put_display_fifo(&mut state, &mut host);
        on_map(&mut state, &host, MID);
        // Past the kernel check, the root is not needed; without it the
        // present path's resolve does not try to validate a fixture walk.
        state.kernel_walk.root = None;
        // Keep the worker's own cadence out of the way, so the drop is found by
        // the swap and not by a tick that happens to come first.
        state.wire_watch.all_sampled_us = Some(u64::MAX);
        state.wire_watch.display_sampled_us = Some(u64::MAX);
        state.active_child_mask = 1 << CHANNEL;
        state.pending.child_mask = 1 << CHANNEL;
        set_surface_wire(&mut host, 0);
        let cap = crate::observe::FailCapture::start();
        crate::runtime::drain::drain_pending(&mut state, &mut host);
        assert_eq!(
            host.get_u32(regs + crate::model::CHILD_REG_HEAD),
            packet.len() as u32,
            "{:?}",
            cap.lines()
        );
        let swap = ev_lines(&cap, "swap");
        assert_eq!(swap.len(), 1, "{:?}", cap.lines());
        assert!(
            swap[0].contains(&format!(
                "ch={CHANNEL} head=0x0 op=0x8 stamp={SWAP_STAMP:#x} slot={CHANNEL} mapping=2 how=exec"
            )) && swap[0].contains("words=[0,0,2,0,104000,0,0,0,0,0]"),
            "{}",
            swap[0]
        );
        let anomaly = lines_with(&cap, "wire_watch first_anomaly ");
        assert_eq!(anomaly.len(), 1, "{:?}", cap.lines());
        assert!(
            anomaly[0].contains("at=swap_end") && anomaly[0].contains("ev=swap ch=4"),
            "{}",
            anomaly[0]
        );
        let stamp = ev_lines(&cap, "stamp");
        assert_eq!(stamp.len(), 1, "{:?}", cap.lines());
        assert!(
            stamp[0].contains(&format!(
                "slot={CHANNEL} value={SWAP_STAMP:#x} route=direct swaps=["
            )),
            "{}",
            stamp[0]
        );
    });
}

/// A swap parked behind a translation hold is seen at its FIFO's head when
/// the worker looks, once, with the mapping and stamp it will carry.
#[test]
fn a_display_swap_held_behind_a_translation_is_seen_before_it_runs() {
    with_switches(true, false, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        put_display_fifo(&mut state, &mut host);
        on_map(&mut state, &host, MID);
        let peeked = crate::runtime::drain::peek_child_head(&state, &host, CHANNEL).unwrap();
        assert_eq!(
            (peeked.head, peeked.completion_stamp, peeked.mapping),
            (0, SWAP_STAMP, Some(MID))
        );
        let cap = crate::observe::FailCapture::start();
        let held = 1 << CHANNEL | 1 << 2;
        state.translation_order_hold_mask = held;
        on_hold(&mut state, held, held, 1 << 3);
        tick(&mut state, &host);
        state.wire_watch.peeked_us = None;
        tick(&mut state, &host);
        let held = ev_lines(&cap, "swap_held");
        assert_eq!(held.len(), 1, "{:?}", cap.lines());
        assert!(
            held[0].contains(&format!(
                "cause=translation_hold ch={CHANNEL} head=0x0 op=0x8 stamp={SWAP_STAMP:#x} mapping=2"
            )),
            "{}",
            held[0]
        );
        assert_eq!(ev_lines(&cap, "hold").len(), 1);
        assert!(
            state.wire_watch.watched[&MID].display,
            "a buffer seen waiting to be shown is sampled at the display cadence"
        );
    });
}

/// The frame owed to a condemned incarnation is dropped, not landed later.
#[test]
fn a_condemned_mapping_drops_its_owed_frame() {
    use crate::runtime::writeback_debt::test_resident_identity;
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        on_map(&mut state, &host, MID);
        let _ = state
            .pending_writebacks
            .arm(MID, test_resident_identity(MID, 4, 4, 1), 4, 4, 1);
        let cap = crate::observe::FailCapture::start();
        set_surface_wire(&mut host, 0);
        tick(&mut state, &host);
        assert!(cap.one("wire_guard_condemned").contains("debt=dropped"));
        assert!(state.pending_writebacks.get(MID).is_none());
        assert_eq!(state.wire_watch.cost.debts_dropped, 1);
    });
}

/// A display present that names a condemned incarnation and was already
/// published when the drop was seen is refused: consumed and stamped, never
/// shown, no present event. One published after it runs. Both stamps reach the
/// guest.
#[test]
fn a_swap_published_before_the_drop_is_refused_and_still_stamped() {
    use crate::model::{CHILD_REG_HEAD, CHILD_REG_TAIL};
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        let (regs, first) = put_display_fifo(&mut state, &mut host);
        on_map(&mut state, &host, MID);
        state.kernel_walk.root = None;
        state.active_child_mask = 1 << CHANNEL;
        // The drop is seen while the swap is still queued.
        set_surface_wire(&mut host, 0);
        let cap = crate::observe::FailCapture::start();
        tick(&mut state, &host);
        let condemned = cap.one("wire_guard_condemned");
        assert!(
            condemned.contains(&format!(
                "horizon=[{CHANNEL}:gen0:pfn{LIST_PFN:#x}:0x0..{:#x}]",
                first.len()
            )),
            "{condemned}"
        );
        // A later swap of the same surface, published after the drop.
        let mut second = first.clone();
        second[8..12].copy_from_slice(&(SWAP_STAMP + 0x100).to_le_bytes());
        let ring = state.child_rings[CHANNEL as usize].page_gpas[0];
        host.write_gpa(ring + first.len() as u64, &second).unwrap();
        host.put_u32(regs + CHILD_REG_TAIL, (first.len() + second.len()) as u32);
        state.pending.child_mask = 1 << CHANNEL;
        state.wire_watch.all_sampled_us = Some(u64::MAX);
        state.wire_watch.display_sampled_us = Some(u64::MAX);
        crate::runtime::drain::drain_pending(&mut state, &mut host);
        assert_eq!(
            host.get_u32(regs + CHILD_REG_HEAD),
            (first.len() + second.len()) as u32,
            "{:?}",
            cap.lines()
        );
        let refused = cap.one("wire_guard_swap_refused");
        assert!(
            refused.contains(&format!(
                "mapping=2 ch={CHANNEL} head=0x0 horizon=0x0..{:#x} gen=0",
                first.len()
            )),
            "{refused}"
        );
        let swaps = ev_lines(&cap, "swap");
        assert_eq!(swaps.len(), 2, "{:?}", cap.lines());
        assert!(
            swaps[0].contains("head=0x0")
                && swaps[0].contains("how=refused")
                && swaps[0].contains("present=-"),
            "a refused present raises no present event: {}",
            swaps[0]
        );
        assert!(swaps[1].contains("how=exec") && swaps[1].contains("present=no_gpa"));
        assert!(
            ev_lines(&cap, "stamp")
                .last()
                .unwrap()
                .contains(&format!("value={:#x}", SWAP_STAMP + 0x100)),
            "{:?}",
            cap.lines()
        );
        assert_eq!(state.wire_watch.cost.swaps_refused, 1);
    });
}

/// A FIFO that was freed and defined again starts its positions over. A
/// horizon taken on the old one (tail 0x100) must not refuse the new
/// incarnation's swap at head 0.
#[test]
fn a_rebuilt_fifo_is_not_judged_by_the_old_horizon() {
    use crate::model::{
        CHILD_REG_HEAD, CHILD_REG_TAIL, PACKET_HEADER_LEN, ROOT_OP_DEFINE_FIFO, ROOT_OP_FREE_FIFO,
    };
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        let (regs, packet) = put_display_fifo(&mut state, &mut host);
        on_map(&mut state, &host, MID);
        state.kernel_walk.root = None;
        state.active_child_mask = 1 << CHANNEL;
        host.put_u32(regs + CHILD_REG_TAIL, 0x100);
        set_surface_wire(&mut host, 0);
        let cap = crate::observe::FailCapture::start();
        tick(&mut state, &host);
        assert!(cap
            .one("wire_guard_condemned")
            .contains(&format!("{CHANNEL}:gen0:pfn{LIST_PFN:#x}:0x0..0x100")));
        // The guest restarts: the channel is freed and defined again, and the
        // new incarnation publishes a swap of the same surface from offset 0.
        for opcode in [ROOT_OP_FREE_FIFO, ROOT_OP_DEFINE_FIFO] {
            crate::runtime::drain::process_root_packet(
                &mut state,
                &mut host,
                &crate::runtime::drain::Packet {
                    opcode,
                    stamp_waits: Vec::new(),
                    total_size: PACKET_HEADER_LEN + 4,
                    completion_stamp: 0,
                    payload: CHANNEL.to_le_bytes().to_vec(),
                    next_head: 0,
                },
            );
        }
        assert_eq!(
            lines_with(
                &cap,
                "wire_guard_horizon_retired mapping=2 ch=4 reason=free_fifo"
            )
            .len(),
            1,
            "{:?}",
            cap.lines()
        );
        host.put_u32(regs + CHILD_REG_HEAD, 0);
        host.put_u32(regs + CHILD_REG_TAIL, packet.len() as u32);
        state.pending.child_mask = 1 << CHANNEL;
        state.wire_watch.all_sampled_us = Some(u64::MAX);
        state.wire_watch.display_sampled_us = Some(u64::MAX);
        crate::runtime::drain::drain_pending(&mut state, &mut host);
        assert_eq!(host.get_u32(regs + CHILD_REG_HEAD), packet.len() as u32);
        assert!(
            lines_with(&cap, "wire_guard_swap_refused").is_empty(),
            "{:?}",
            cap.lines()
        );
        let swaps = ev_lines(&cap, "swap");
        assert_eq!(swaps.len(), 1, "{:?}", cap.lines());
        assert!(
            swaps[0].contains("head=0x0") && swaps[0].contains("how=exec"),
            "{}",
            swaps[0]
        );
    });
}

/// The same, when the guest points the FIFO at another ring without a
/// define: the ring is a different identity and the old horizon is retired.
#[test]
fn a_fifo_moved_to_another_ring_is_not_judged_by_the_old_horizon() {
    use crate::model::{CHILD_REG_BASE_PFN, CHILD_REG_HEAD, CHILD_REG_TAIL};
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        let (regs, packet) = put_display_fifo(&mut state, &mut host);
        on_map(&mut state, &host, MID);
        state.kernel_walk.root = None;
        state.active_child_mask = 1 << CHANNEL;
        host.put_u32(regs + CHILD_REG_TAIL, 0x100);
        set_surface_wire(&mut host, 0);
        let cap = crate::observe::FailCapture::start();
        tick(&mut state, &host);
        // A second page list naming the same ring page.
        let list2 = state.pfn_gpa(LIST_PFN + 1);
        host.map_range(list2, state.page_size() as usize, 0);
        host.put_u32(list2, RING_PFN);
        host.put_u32(regs + CHILD_REG_BASE_PFN, LIST_PFN + 1);
        host.put_u32(regs + CHILD_REG_HEAD, 0);
        host.put_u32(regs + CHILD_REG_TAIL, packet.len() as u32);
        state.pending.child_mask = 1 << CHANNEL;
        state.wire_watch.all_sampled_us = Some(u64::MAX);
        state.wire_watch.display_sampled_us = Some(u64::MAX);
        crate::runtime::drain::drain_pending(&mut state, &mut host);
        assert_eq!(host.get_u32(regs + CHILD_REG_HEAD), packet.len() as u32);
        assert_eq!(
            lines_with(
                &cap,
                "wire_guard_horizon_retired mapping=2 ch=4 reason=ring_changed"
            )
            .len(),
            1,
            "{:?}",
            cap.lines()
        );
        assert!(lines_with(&cap, "wire_guard_swap_refused").is_empty());
        assert!(ev_lines(&cap, "swap")[0].contains("how=exec"));
    });
}

/// A frame landed a row at a time reads the counts once for the burst, and
/// the guard decides on that read.
#[test]
fn a_burst_of_row_writes_reads_the_counts_once() {
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        on_map(&mut state, &host, MID);
        set_surface_wire(&mut host, 0);
        assert!(!write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
        let reads = state.wire_watch.cost.reads;
        let rows = (0..64)
            .filter(|_| !before_write(&mut state, &host, MID, "mapping_write", WriteKind::Cpu))
            .count();
        assert_eq!(rows, 64, "every row of the burst is refused");
        assert!(
            state.wire_watch.cost.reads - reads < 64 * 3,
            "a row reuses a read taken within the window: {} reads",
            state.wire_watch.cost.reads - reads
        );
        assert_eq!(state.wire_watch.watched[&MID].refused, 65);
    });
}

// MARK: - Withdrawn claims and the surface quarantine

use crate::runtime::mapper::{
    first_surface_page_collision, surface_pages_are_exclusively_owned, vouch_mapping_pages_verdict,
    write_mapping_bytes,
};
use crate::runtime::surface_quarantine::{is_quarantined, is_withdrawn};

/// The surface the guest gave one of mapping 2's pages to, as mid=9 on the
/// regression boot.
const NEW: u32 = 9;
/// A page no other surface in these tests holds.
const SPARE: u64 = 0x1_0030_0000;

fn new_surface(gpas: &[u64]) -> Surface {
    Surface {
        mid: NEW,
        mi: 0xffff_fe1e_47e7_0000,
        surface: 0xffff_fe24_13d8_0000,
        table: 0xffff_fe24_13d8_1000,
        page_table: 0xffff_fe9a_f650_0000,
        gpas: gpas.to_vec(),
    }
}

fn route(name: &str) -> u64 {
    crate::runtime::drain::census::store_route_count(name)
}

/// Regression 1: mapping 2 condemned and withdrawn, then a new mapping over
/// one of its pages: the new one passes its collision check and is written;
/// mapping 2 takes nothing.
#[test]
fn a_new_mapping_over_a_withdrawn_incarnations_pages_is_written() {
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        on_map(&mut state, &host, MID);
        let new = new_surface(&[G[1]]);
        put_surface(&mut host, &new, 1);
        let cap = crate::observe::FailCapture::start();
        set_surface_wire(&mut host, 0);
        tick(&mut state, &host);
        let withdrawn = cap.one("surface_quarantine_withdrawn");
        assert!(
            withdrawn.contains("mid=2 ") && withdrawn.contains("evidence=[wire field=surface"),
            "{withdrawn}"
        );
        assert!(is_withdrawn(&state, MID) && is_quarantined(&state, MID));

        adopt(&mut state, &new);
        on_map(&mut state, &host, NEW);
        assert!(surface_pages_are_exclusively_owned(
            &mut state, NEW, "rewire"
        ));
        assert!(!is_quarantined(&state, NEW));
        assert!(cap.one("surface_quarantine_resolved").contains(
            "reason=owner_condemned old=mid=2 new=mid=9 order=condemned_first site=rewire"
        ));
        let token = vouch_mapping_pages_verdict(&mut state, &host, NEW)
            .1
            .expect("the new mapping is vouched");
        assert!(write_mapping_bytes(
            &mut state, &mut host, NEW, 0, b"NEW", &token
        ));
        let mut b = [0u8; 3];
        host.read_gpa(G[1], &mut b).unwrap();
        assert_eq!(&b, b"NEW");
        assert!(vouch_mapping_pages_verdict(&mut state, &host, MID)
            .1
            .is_none());
    });
}

/// Regression 2: a withdrawn owner is skipped, a valid one is not.
#[test]
fn a_third_valid_owner_still_blocks_the_new_mapping() {
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        on_map(&mut state, &host, MID);
        set_surface_wire(&mut host, 0);
        tick(&mut state, &host);
        assert!(is_withdrawn(&state, MID));
        // Mapping 7 holds SPARE, and nothing says it gave it up.
        host.map_range(SPARE, PAGE as usize, 0);
        state.map_surface(7);
        let m = state.mappings.get_mut(&7).unwrap();
        m.mapping_internal = 0xffff_fe1e_47e8_0000;
        m.page_entries = vec![entry(SPARE)];
        let new = new_surface(&[G[1], SPARE]);
        put_surface(&mut host, &new, 1);
        adopt(&mut state, &new);
        let cap = crate::observe::FailCapture::start();
        assert!(!surface_pages_are_exclusively_owned(
            &mut state, NEW, "rewire"
        ));
        assert!(is_quarantined(&state, NEW) && is_quarantined(&state, 7));
        let line = cap
            .lines()
            .into_iter()
            .find(|l| l.starts_with("surface_quarantine site=rewire"))
            .expect("a quarantine line");
        assert!(line.contains("owner=[mid=7 "), "{line}");
        assert!(state.mappings[&NEW].quarantine.as_ref().unwrap().partners == vec![7]);
    });
}

/// Regression 3: the guest preparing the old incarnation again gives it
/// nothing back: writes and its owed frame are refused for good.
#[test]
fn a_withdrawn_incarnation_stays_refused_when_its_count_recovers() {
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        on_map(&mut state, &host, MID);
        set_surface_wire(&mut host, 0);
        tick(&mut state, &host);
        assert!(is_withdrawn(&state, MID));
        let generation = state.mappings[&MID].map_generation;
        let _ = state.pending_writebacks.arm(
            MID,
            crate::runtime::writeback_debt::test_resident_identity(MID, 4, 4, 1),
            4,
            4,
            generation,
        );
        set_surface_wire(&mut host, 1);
        let cap = crate::observe::FailCapture::start();
        assert!(!write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
        assert!(cap.one("wire_guard_still_condemned").contains("value=1"));
        assert!(vouch_mapping_pages_verdict(&mut state, &host, MID)
            .1
            .is_none());
        let dropped = route("wbdebt_dropped_withdrawn");
        crate::runtime::writeback_debt::pay_for_mapping(&mut state, &mut host, MID);
        assert!(
            state.pending_writebacks.get(MID).is_none(),
            "never lands, never kept"
        );
        assert_eq!(route("wbdebt_dropped_withdrawn"), dropped + 1);
        assert!(is_withdrawn(&state, MID));
    });
}

/// Regression 4: a new MappingInternal for the old id is another incarnation.
/// It is a valid owner again and the guard does not carry over.
#[test]
fn a_new_incarnation_of_the_old_id_inherits_nothing() {
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        on_map(&mut state, &host, MID);
        set_surface_wire(&mut host, 0);
        tick(&mut state, &host);
        assert!(is_withdrawn(&state, MID));
        let again = Surface {
            mid: MID,
            mi: MI + 0x2000,
            surface: SURFACE + 0x2000,
            table: TABLE + 0x2000,
            page_table: PAGE_TABLE + 0x4000,
            gpas: vec![SPARE],
        };
        put_surface(&mut host, &again, 1);
        assert!(state.attach_mapping_internal(MID, again.mi));
        adopt(&mut state, &again);
        crate::runtime::surface_quarantine::confirm_new_backing(&mut state, MID);
        assert!(!is_quarantined(&state, MID) && !is_withdrawn(&state, MID));
        assert!(write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
        // Another mapping over its new page collides with it as with any owner.
        state.map_surface(11);
        let m = state.mappings.get_mut(&11).unwrap();
        m.mapping_internal = 0xffff_fe1e_47e9_0000;
        m.page_entries = vec![entry(SPARE)];
        assert_eq!(first_surface_page_collision(&state, 11), Some((SPARE, MID)));
    });
}

/// Regression 5 and the reverse order: the collision comes first and
/// quarantines both. The condemnation alone excludes nothing; the worker's
/// next tick waits for writes still in flight to mapping 2's pages, and only
/// then withdraws it and frees mapping 9, after re-resolving it and reading
/// its own counts.
#[test]
fn in_flight_writes_settle_before_a_condemned_owner_is_excluded() {
    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        on_map(&mut state, &host, MID);
        let new = new_surface(&[G[1]]);
        put_surface(&mut host, &new, 1);
        // Condemned by a write's own sample, not yet withdrawn.
        set_surface_wire(&mut host, 0);
        assert!(!write_now(
            &mut state,
            &host,
            MID,
            "mapping_write",
            WriteKind::Cpu
        ));
        assert!(!is_withdrawn(&state, MID));
        adopt(&mut state, &new);
        assert!(
            !surface_pages_are_exclusively_owned(&mut state, NEW, "rewire"),
            "until it is withdrawn, mapping 2 still owns its pages"
        );
        assert!(is_quarantined(&state, NEW) && is_quarantined(&state, MID));
        assert!(state.mappings[&NEW].page_entries.is_empty());

        crate::backend::vulkan::engine::test_arm_guest_write(&[G[0]]);
        assert!(crate::backend::vulkan::engine::guest_writes_outstanding());
        let settles = route("settle_quarantine_overlap");
        let cap = crate::observe::FailCapture::start();
        tick(&mut state, &host);
        assert!(!crate::backend::vulkan::engine::guest_writes_outstanding());
        assert_eq!(route("settle_quarantine_overlap"), settles + 1);
        assert!(cap
            .one("surface_quarantine_withdrawn")
            .contains("in_flight_before=overlap"));
        assert!(is_withdrawn(&state, MID) && is_quarantined(&state, MID));
        let resolved = cap.one("surface_quarantine_resolved");
        assert!(
            resolved.contains("reason=owner_condemned old=mid=2 new=mid=9 order=collision_first")
                && resolved.contains("evidence=[wire internal=")
                && resolved.contains("surface_wire=1 table_wire=2"),
            "{resolved}"
        );
        assert!(!is_quarantined(&state, NEW));
        assert_eq!(state.mappings[&NEW].page_entries, vec![entry(G[1])]);
    });
}

/// Without wire evidence for either side, a collision quarantines both and
/// the worker frees neither: the guard off, or no anomaly.
#[test]
fn without_wire_evidence_both_sides_stay_quarantined() {
    for (guard, wire) in [(false, 0), (true, 1)] {
        with_switches(true, guard, || {
            let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
            on_map(&mut state, &host, MID);
            let new = new_surface(&[G[1]]);
            put_surface(&mut host, &new, 1);
            set_surface_wire(&mut host, wire);
            tick(&mut state, &host);
            adopt(&mut state, &new);
            let cap = crate::observe::FailCapture::start();
            assert!(!surface_pages_are_exclusively_owned(
                &mut state, NEW, "rewire"
            ));
            state.wire_watch.all_sampled_us = None;
            tick(&mut state, &host);
            assert!(is_quarantined(&state, NEW) && is_quarantined(&state, MID));
            assert!(!is_withdrawn(&state, MID));
            assert!(lines_with(&cap, "surface_quarantine_withdrawn").is_empty());
            assert!(lines_with(&cap, "surface_quarantine_resolved").is_empty());
        });
    }
}

// MARK: - Reads of a condemned incarnation

/// After the drop, the bytes another owner left in mapping 2's pages reach no
/// reader: each read path refuses by site and leaves its buffer as it was.
/// Before the drop the same read reaches the pages, which is what makes the
/// refusals mean something, and a new incarnation under the id is read again.
#[test]
fn a_condemned_incarnations_pages_are_never_read() {
    use crate::runtime::mapper::{ensure_contig_view, mapping_page_gpas, read_mapping_bytes};
    use crate::runtime::mapping_write::{read_raw_rows, read_rect_raw_at, Rect, SurfaceWindow};
    use crate::runtime::scanout::read_mapping_bgra8;

    with_switches(true, true, || {
        let (mut state, mut host) = fixture(EXPECTED_KERNEL_VERSION);
        on_map(&mut state, &host, MID);
        // What the next owner of the pages wrote into them.
        for &gpa in &G {
            host.write_gpa(gpa, &[0xEE; 64]).unwrap();
        }
        let mut before = [0x11u8; 16];
        assert!(read_mapping_bytes(
            &mut state,
            &mut host,
            MID,
            0,
            &mut before
        ));
        assert_eq!(
            before, [0xEE; 16],
            "before the drop the read reaches the pages"
        );
        assert!(!guest_pages_condemned(&state, MID));

        let cap = crate::observe::FailCapture::start();
        set_surface_wire(&mut host, 0);
        tick(&mut state, &host);
        assert!(guest_pages_condemned(&state, MID));
        assert!(is_withdrawn(&state, MID));

        let mut buf = [0x11u8; 16];
        assert!(!read_mapping_bytes(&mut state, &mut host, MID, 0, &mut buf));
        assert!(mapping_page_gpas(&mut state, &mut host, MID).is_none());
        assert!(ensure_contig_view(&mut state, &mut host, MID).is_none());
        let window = SurfaceWindow {
            base_off: 0,
            bpr: 8,
            span_end: 16,
            bpp: 4,
        };
        let rect = Rect {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 2,
        };
        assert!(!read_rect_raw_at(
            &mut state, &mut host, MID, window, rect, &mut buf, 8
        ));
        assert!(!read_raw_rows(
            &mut state, &mut host, MID, &mut buf, 8, 8, 2, 2
        ));
        assert!(!read_mapping_bgra8(
            &mut state, &mut host, MID, &mut buf, 8, 2, 2
        ));
        assert_eq!(buf, [0x11; 16], "not one byte of the other owner's");
        for site in [
            "mapping_read",
            "page_list",
            "contig_view",
            "read_rect",
            "read_raw_rows",
            "paint_mapping",
        ] {
            let lines = lines_with(
                &cap,
                &format!("wire_guard_read_refused mapping=2 site={site} "),
            );
            assert_eq!(lines.len(), 1, "{site}: {:?}", cap.lines());
            assert!(lines[0].contains(" n=1 "), "{}", lines[0]);
        }
        assert_eq!(state.wire_watch.cost.reads_refused, 6);
        // A refusal repeats silently until its site's next power of two.
        assert!(!read_mapping_bytes(&mut state, &mut host, MID, 0, &mut buf));
        assert!(!read_mapping_bytes(&mut state, &mut host, MID, 0, &mut buf));
        assert!(!read_mapping_bytes(&mut state, &mut host, MID, 0, &mut buf));
        assert_eq!(
            lines_with(&cap, "wire_guard_read_refused mapping=2 site=mapping_read ").len(),
            3,
            "n=1, 2 and 4"
        );

        // UNMAP ends the incarnation; a new MappingInternal is read as any other.
        on_unmap(&mut state, &host, MID);
        assert!(state.unmap_surface(MID));
        let again = Surface {
            mi: MI + 0x1000,
            ..old_surface()
        };
        put_surface(&mut host, &again, 1);
        for &gpa in &G {
            host.write_gpa(gpa, &[0x5A; 64]).unwrap();
        }
        adopt(&mut state, &again);
        on_map(&mut state, &host, MID);
        assert!(!guest_pages_condemned(&state, MID));
        assert!(read_mapping_bytes(&mut state, &mut host, MID, 0, &mut buf));
        assert_eq!(buf, [0x5A; 16]);
    });
}
