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

/// The guest's UNMAP of the old surface is the evidence: its quarantine lifts
/// and its frame is dropped unwritten, and the new surface, left without a
/// counterpart, is lifted with it and writes again.
#[test]
fn the_old_surfaces_unmap_lifts_both_and_drops_only_its_frame() {
    let (mut host, mut state) = fixture();
    owe_a_frame(&mut state, OLD);
    surface(&mut state, NEW, &[P1]);
    fail_closed_surface_page_collision(&mut state, NEW, P1, OLD, 1, "rewire");
    let cap = crate::observe::sink::FailCapture::start();

    assert!(state.unmap_surface(OLD));
    assert!(!is_quarantined(&state, OLD) && !is_quarantined(&state, NEW));
    assert!(
        state.pending_writebacks.get(OLD).is_none(),
        "dropped, not paid"
    );
    let lines = cap.lines();
    assert!(lines.iter().any(
        |l| l.contains("surface_quarantine_lifted mid=2 evidence=unmap")
            && l.contains("debt_dropped=true")
    ));
    assert!(lines
        .iter()
        .any(|l| l.contains("surface_quarantine_lifted mid=8 evidence=partner_unmap")));

    // The new surface re-resolves its page and owns it now.
    state.mappings.get_mut(&NEW).unwrap().page_entries = vec![entry(P1)];
    let token = vouch_mapping_pages_verdict(&mut state, &host, NEW)
        .1
        .expect("the survivor can be vouched");
    assert!(write_mapping_bytes(
        &mut state, &mut host, NEW, 0, b"MINE", &token
    ));
    let mut got = [0u8; 4];
    host.read_gpa(P1, &mut got).unwrap();
    assert_eq!(&got, b"MINE");
}

/// A new MAP of the old id is a rebind: its quarantine lifts and its old
/// frame is dropped. A MappingInternal re-stated unchanged is not evidence.
#[test]
fn a_rebind_lifts_the_quarantine_and_a_restatement_does_not() {
    let (_host, mut state) = fixture();
    owe_a_frame(&mut state, OLD);
    surface(&mut state, NEW, &[P1]);
    fail_closed_surface_page_collision(&mut state, NEW, P1, OLD, 1, "rewire");

    assert!(
        state.attach_mapping_internal(OLD, 1),
        "the same MappingInternal"
    );
    assert!(
        is_quarantined(&state, OLD),
        "a re-statement is not a rebind"
    );

    assert!(state.map_surface(OLD));
    assert!(!is_quarantined(&state, OLD));
    assert!(state.pending_writebacks.get(OLD).is_none());
    assert!(
        !is_quarantined(&state, NEW),
        "its only counterpart was rebound"
    );
}
