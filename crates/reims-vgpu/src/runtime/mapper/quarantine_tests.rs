//! Tests for the quarantine of two type-11 mappings that claim one guest page.
//! See [`crate::runtime::surface_quarantine`].

use super::*;
use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
use crate::model::{DeviceId, PAGE_SHIFT_X86};
use crate::runtime::host::FakeHost;
use crate::runtime::host_writes::HostWriteVerdict;
use crate::runtime::surface_quarantine::is_quarantined;

const PAGE: u64 = 1 << PAGE_SHIFT_X86;
/// The old surface's two pages, apart so writes take the fragmented path.
const P0: u64 = 0x1000_0000;
const P1: u64 = 0x2000_0000;
/// The display surface the guest stopped owning without an UNMAP, and the new
/// surface handed one of its pages, as on the 27 boot (mid=2, mid=8).
const OLD: u32 = 2;
const NEW: u32 = 8;

fn entry(gpa: u64) -> u32 {
    ((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT | PAGE_ENTRY_VALID
}

fn surface(state: &mut DeviceState, mid: u32, pages: &[u64]) {
    state.map_surface(mid);
    let m = state.mappings.get_mut(&mid).unwrap();
    m.page_entries = pages.iter().map(|&p| entry(p)).collect();
    m.mapping_internal = 1;
    m.has_geom = true;
    m.width = 4;
    m.height = 4;
    m.format = crate::contract::pixel_format::MTL_FORMAT_RGBA16_FLOAT;
}

fn fixture() -> (FakeHost, DeviceState) {
    let mut host = FakeHost::new();
    host.strict_linux_map = true;
    host.map_range(P0, PAGE as usize, 0);
    host.map_range(P1, PAGE as usize, 0);
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    surface(&mut state, OLD, &[P0, P1]);
    (host, state)
}

fn owe_a_frame(state: &mut DeviceState, mid: u32) {
    let generation = state.mappings[&mid].map_generation;
    let _ = state.pending_writebacks.arm(
        mid,
        crate::runtime::writeback_debt::test_resident_identity(mid, 4, 4, 1),
        4,
        4,
        generation,
    );
}

fn untouched(host: &FakeHost) -> bool {
    [P0, P1].iter().all(|&gpa| {
        let mut b = [0u8; 16];
        host.read_gpa(gpa, &mut b).unwrap();
        b == [0u8; 16]
    })
}

/// The guest hands a page of a live surface to a new one: both are
/// quarantined, the evidence is logged with both sides, and the old surface
/// takes nothing through any writer — a flush, a proof minted before the
/// collision, its owed frame — nor after its own table is re-read unchanged.
#[test]
fn a_collision_quarantines_both_and_the_old_surface_takes_no_write() {
    let (mut host, mut state) = fixture();
    owe_a_frame(&mut state, OLD);
    let early = vouch_mapping_pages_verdict(&mut state, &host, OLD)
        .1
        .expect("vouched before the collision");
    surface(&mut state, NEW, &[P1]);
    let epoch = state.host_writes.epoch();

    let cap = crate::observe::sink::FailCapture::start();
    let (gpa, owner) = first_surface_page_collision(&state, NEW).expect("the new surface collides");
    assert_eq!((gpa, owner), (P1, OLD));
    fail_closed_surface_page_collision(&mut state, NEW, gpa, owner, 1, "rewire");

    assert!(is_quarantined(&state, OLD) && is_quarantined(&state, NEW));
    let line = cap
        .lines()
        .into_iter()
        .find(|l| l.contains("surface_quarantine site=rewire"))
        .expect("a quarantine line");
    assert!(line.contains(&format!("gpa={P1:#x}")), "{line}");
    assert!(line.contains("new=[mid=8 gen="), "{line}");
    assert!(line.contains("owner=[mid=2 gen="), "{line}");
    assert!(
        line.contains("owes=true pages=2"),
        "the old side's debt: {line}"
    );
    assert!(line.contains("in_flight="), "{line}");

    let frame: Vec<u8> = [89u8, 101, 112, 255].repeat(16);
    assert!(!crate::runtime::mapping_write::write_bgra8_uncached(
        &mut state, &mut host, OLD, &frame, 16, 4, 4
    ));
    assert!(!write_mapping_bytes(
        &mut state, &mut host, OLD, 0, b"EARLY", &early
    ));
    crate::runtime::writeback_debt::pay_for_mapping(&mut state, &mut host, OLD);
    assert!(
        state.pending_writebacks.get(OLD).is_some(),
        "the frame is held, neither written nor dropped"
    );

    // A resolve that reads the same old table back changes nothing.
    {
        let m = state.mappings.get_mut(&OLD).unwrap();
        m.page_entries = vec![entry(P0), entry(P1)];
        m.validated_ms = crate::observe::elapsed_ms() as u64;
    }
    assert!(!crate::runtime::mapping_write::write_bgra8_uncached(
        &mut state, &mut host, OLD, &frame, 16, 4, 4
    ));

    assert!(untouched(&host), "no byte reached either page");
    assert_eq!(
        state.host_writes.wrote_any_since(epoch, &[P0, P1]),
        HostWriteVerdict::Quiet
    );
}

/// The reverse of the 27 boot's sequence: the new surface is unmapped after
/// the collision. That proves the new surface gave its pages up, not that the
/// old one owns them again, so the old surface stays quarantined: its flush and
/// its owed frame write nothing.
#[test]
fn the_new_surfaces_unmap_does_not_restore_the_old_one() {
    let (mut host, mut state) = fixture();
    owe_a_frame(&mut state, OLD);
    // As on the 27 boot: two new surfaces, each handed one of the old one's
    // pages, then both unmapped in turn.
    surface(&mut state, NEW, &[P1]);
    fail_closed_surface_page_collision(&mut state, NEW, P1, OLD, 1, "rewire");
    surface(&mut state, 9, &[P0]);
    fail_closed_surface_page_collision(&mut state, 9, P0, OLD, 1, "rewire");
    let epoch = state.host_writes.epoch();

    assert!(state.unmap_surface(NEW));
    assert!(!is_quarantined(&state, NEW));
    assert!(state.unmap_surface(9));
    assert!(!is_quarantined(&state, 9));
    assert!(
        is_quarantined(&state, OLD),
        "a partner's release is not the old surface's evidence"
    );

    let frame: Vec<u8> = [89u8, 101, 112, 255].repeat(16);
    assert!(!crate::runtime::mapping_write::write_bgra8_uncached(
        &mut state, &mut host, OLD, &frame, 16, 4, 4
    ));
    crate::runtime::writeback_debt::pay_for_mapping(&mut state, &mut host, OLD);
    assert!(state.pending_writebacks.get(OLD).is_some(), "still held");
    assert!(untouched(&host), "no byte reached either page");
    assert_eq!(
        state.host_writes.wrote_any_since(epoch, &[P0, P1]),
        HostWriteVerdict::Quiet
    );
}

/// The guest's UNMAP of the old surface is its own evidence: its quarantine
/// lifts and its frame is dropped unwritten. The new surface stays quarantined:
/// its own MAP only arms the lift, and the resolve that adopts a table other
/// than the quarantined one completes it.
#[test]
fn each_side_is_lifted_only_by_its_own_lifecycle() {
    let (mut host, mut state) = fixture();
    owe_a_frame(&mut state, OLD);
    surface(&mut state, NEW, &[P1]);
    fail_closed_surface_page_collision(&mut state, NEW, P1, OLD, 1, "rewire");
    let cap = crate::observe::sink::FailCapture::start();

    assert!(state.unmap_surface(OLD));
    assert!(!is_quarantined(&state, OLD));
    assert!(
        state.pending_writebacks.get(OLD).is_none(),
        "dropped, not paid"
    );
    assert!(cap.lines().iter().any(|l| l
        .contains("surface_quarantine_lifted mid=2 evidence=unmap")
        && l.contains("debt_dropped=true")));
    assert!(
        is_quarantined(&state, NEW),
        "the old surface's release is not the new one's evidence"
    );

    assert!(state.map_surface(NEW));
    assert!(is_quarantined(&state, NEW), "a MAP alone lifts nothing");
    // What a resolve does after adopting a table: the same one lifts nothing,
    // another one does.
    state.mappings.get_mut(&NEW).unwrap().page_entries = vec![entry(P1)];
    crate::runtime::surface_quarantine::confirm_new_backing(&mut state, NEW);
    assert!(is_quarantined(&state, NEW), "the quarantined table again");
    state.mappings.get_mut(&NEW).unwrap().page_entries = vec![entry(P0)];
    crate::runtime::surface_quarantine::confirm_new_backing(&mut state, NEW);
    assert!(!is_quarantined(&state, NEW));

    let token = vouch_mapping_pages_verdict(&mut state, &host, NEW)
        .1
        .expect("the new backing can be vouched");
    assert!(write_mapping_bytes(
        &mut state, &mut host, NEW, 0, b"MINE", &token
    ));
    let mut got = [0u8; 4];
    host.read_gpa(P0, &mut got).unwrap();
    assert_eq!(&got, b"MINE");
}

/// A proof minted before a quarantine never covers again, even if the
/// quarantine were cleared at the same generation: the revocation moved the
/// mapping's write epoch.
#[test]
fn a_proof_from_before_the_quarantine_never_covers_again() {
    let (host, mut state) = fixture();
    let early = vouch_mapping_pages_verdict(&mut state, &host, OLD)
        .1
        .expect("vouched before");
    assert!(early.covers(&state, OLD));
    surface(&mut state, NEW, &[P1]);
    fail_closed_surface_page_collision(&mut state, NEW, P1, OLD, 1, "rewire");
    let generation = state.mappings[&OLD].map_generation;
    state.mappings.get_mut(&OLD).unwrap().quarantine = None;
    assert_eq!(state.mappings[&OLD].map_generation, generation);
    assert!(!early.covers(&state, OLD));
}

/// A store that finds the collision itself — the resolve inside its write is
/// what quarantines the mapping — defers the frame: back in the ledger at its
/// generation, not lost. Without a quarantine the refusal is a loss as before.
#[test]
fn a_store_that_finds_the_quarantine_defers_the_frame() {
    use crate::runtime::surface_quarantine::{after_refused_store, RefusedStore};
    let (_host, mut state) = fixture();
    owe_a_frame(&mut state, OLD);
    let debt = state.pending_writebacks.take(OLD).unwrap();
    let armed_gen = debt.map_generation;
    assert!(matches!(
        after_refused_store(&mut state, OLD, debt.clone()),
        RefusedStore::Lost(_)
    ));

    surface(&mut state, NEW, &[P1]);
    fail_closed_surface_page_collision(&mut state, NEW, P1, OLD, 1, "rewire");
    assert_eq!(
        after_refused_store(&mut state, OLD, debt),
        RefusedStore::Deferred
    );
    assert_eq!(
        state.pending_writebacks.get(OLD).map(|d| d.map_generation),
        Some(armed_gen),
        "held at the generation it was armed at"
    );
}

// MARK: - Through the resolve

mod through_resolve {
    use super::*;
    use crate::contract::iosurface_pages::{
        MAPPING_INTERNAL_BACKPTR, MAPPING_INTERNAL_EXPECTED_SIZE, MAPPING_INTERNAL_ID,
        MAPPING_INTERNAL_PAGE_COUNT, MAPPING_INTERNAL_PAGE_FIELD_48, MAPPING_INTERNAL_SIZE,
        MAPPING_PAGE_TABLE_FROM_F48,
    };
    use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SIZE_ARM64E};

    const KVA: u64 = 0xfffffe00_2000_0000;
    const TABLE: u64 = KVA + 0x3000;
    const A: u32 = 0x3_1000;
    const B: u32 = 0x3_2000;

    fn put(host: &mut FakeHost, at: u64, bytes: &[u8]) {
        host.map_range(at, bytes.len(), 0);
        host.write_gpa(at, bytes).unwrap();
    }

    fn entry16(pfn: u32) -> u32 {
        (pfn << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID
    }

    /// Mapping `OLD` with a MappingInternal whose one-page table names `A`.
    fn resolvable() -> (FakeHost, DeviceState) {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let (internal, mapper, page_obj) = (KVA, KVA + 0x1000, KVA + 0x2000);
        put(
            &mut host,
            internal + MAPPING_INTERNAL_BACKPTR,
            &mapper.to_le_bytes(),
        );
        put(
            &mut host,
            internal + MAPPING_INTERNAL_ID,
            &OLD.to_le_bytes(),
        );
        put(
            &mut host,
            internal + MAPPING_INTERNAL_SIZE,
            &MAPPING_INTERNAL_EXPECTED_SIZE.to_le_bytes(),
        );
        put(
            &mut host,
            internal + MAPPING_INTERNAL_PAGE_FIELD_48,
            &page_obj.to_le_bytes(),
        );
        put(
            &mut host,
            internal + MAPPING_INTERNAL_PAGE_COUNT,
            &1u64.to_le_bytes(),
        );
        put(
            &mut host,
            page_obj + MAPPING_PAGE_TABLE_FROM_F48,
            &TABLE.to_le_bytes(),
        );
        put(&mut host, TABLE, &entry16(A).to_le_bytes());
        for pfn in [A, B] {
            host.map_range(
                u64::from(pfn) << PAGE_SHIFT_ARM64E,
                PAGE_SIZE_ARM64E as usize,
                0,
            );
        }
        state.mapper_device_kva = mapper;
        assert!(state.attach_mapping_internal(OLD, internal));
        assert!(resolve_mapping_backing(&mut state, &host, OLD));
        (host, state)
    }

    /// The 27 sequence with a MAP for the old id: the MAP arms nothing but a
    /// re-check, the resolve that re-reads the same stale table keeps the
    /// quarantine, and only a resolve adopting a different table lifts it.
    #[test]
    fn a_map_with_the_same_stale_table_lifts_nothing_and_a_new_table_does() {
        let (mut host, mut state) = resolvable();
        let gpa = u64::from(A) << PAGE_SHIFT_ARM64E;
        state.map_surface(NEW);
        state.mappings.get_mut(&NEW).unwrap().page_entries = vec![entry16(A)];
        fail_closed_surface_page_collision(&mut state, NEW, gpa, OLD, 1, "rewire");
        assert!(is_quarantined(&state, OLD));

        assert!(state.map_surface(OLD));
        let _ = resolve_mapping_backing(&mut state, &host, OLD);
        assert_eq!(state.mappings[&OLD].page_entries, vec![entry16(A)]);
        assert!(
            is_quarantined(&state, OLD),
            "the same stale table re-announced"
        );

        put(&mut host, TABLE, &entry16(B).to_le_bytes());
        assert!(resolve_mapping_backing(&mut state, &host, OLD));
        assert_eq!(state.mappings[&OLD].page_entries, vec![entry16(B)]);
        assert!(!is_quarantined(&state, OLD), "a new backing");
    }
}
