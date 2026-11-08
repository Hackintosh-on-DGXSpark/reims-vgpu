//! Type-11 mappings that claim the same guest page, held out of writing until
//! the guest says which one it released.
//!
//! # Why
//!
//! Two live IOSurface mappings over one physical page means one of them names
//! memory it no longer owns. On macOS 27 the guest handed a display surface's
//! pages to the kernel and to new surfaces without any UNMAP for the old one;
//! the device saw the new surface collide, closed the *new* mapping, and went on
//! writing frames through the old one — into the kernel's memory. Nothing on the
//! device side says which of the two is stale: the old mapping's table re-reads
//! identically, a fresh walk only proves the address translation, and `mapped`
//! only says no UNMAP arrived.
//!
//! # What this does
//!
//! On a collision both mappings' current incarnations are quarantined. Their
//! write authority is revoked at once — a page proof (`PagesVouched`) covers
//! neither, so every CPU write-back, import write and direct write refuses for
//! the whole mapping, not only the colliding page — and GPU writes already
//! submitted to either page list are waited for. Their owed frames stay in the
//! ledger and are not written: the collision alone does not say whose they are.
//!
//! A quarantine is lifted only by protocol evidence about that mapping: an
//! UNMAP, a new MAP or MappingInternal for its id, or its backing's delete; and
//! a partner whose every counterpart was released that way is lifted with it.
//! Re-reading the same page table lifts nothing. There is no page blacklist:
//! the pages are free to belong to whoever the next protocol event names.

use crate::model::DeviceState;

/// Why one mapping is not written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Quarantine {
    /// The mappings it collided with, in order.
    pub partners: Vec<u32>,
    /// The first page they were found to share.
    pub gpa: u64,
    /// The mapping's own `map_generation` when it was quarantined.
    pub generation: u32,
    pub since_ms: u64,
}

pub fn is_quarantined(state: &DeviceState, mapping_id: u32) -> bool {
    state
        .mappings
        .get(&mapping_id)
        .is_some_and(|m| m.quarantine.is_some())
}

/// What one side of a collision held at the moment it was found.
struct Side {
    mapping_id: u32,
    generation: u32,
    validated_ago_ms: Option<u64>,
    owes: bool,
    pages: Vec<u64>,
}

impl Side {
    fn of(state: &DeviceState, mapping_id: u32, now_ms: u64) -> Option<Self> {
        let m = state.mappings.get(&mapping_id)?;
        let shift = state.page_shift;
        Some(Self {
            mapping_id,
            generation: m.map_generation,
            validated_ago_ms: (m.validated_ms != 0).then(|| now_ms.saturating_sub(m.validated_ms)),
            owes: state.pending_writebacks.get(mapping_id).is_some(),
            pages: m
                .page_entries
                .iter()
                .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, shift))
                .collect(),
        })
    }

    fn render(&self) -> String {
        format!(
            "mid={} gen={} validated_ago_ms={} owes={} pages={}",
            self.mapping_id,
            self.generation,
            self.validated_ago_ms
                .map_or_else(|| "never".to_string(), |ms| ms.to_string()),
            self.owes,
            self.pages.len()
        )
    }
}

/// Whether GPU writes already submitted can reach these pages.
fn in_flight(pages: &[u64]) -> &'static str {
    #[cfg(feature = "backend-vulkan")]
    {
        use crate::backend::vulkan::engine::{self, GuestWriteReach};
        if !engine::guest_writes_outstanding() {
            return "none";
        }
        match engine::guest_writes_reaching(pages) {
            GuestWriteReach::Disjoint => "none",
            GuestWriteReach::Overlap => "overlap",
            GuestWriteReach::Unnamed => "unnamed",
        }
    }
    #[cfg(not(feature = "backend-vulkan"))]
    {
        let _ = pages;
        "none"
    }
}

/// Quarantine both mappings of a collision `mapping_id` found with `owner` at
/// `gpa`. Called before the caller fails the new mapping's resolve, while both
/// page lists are still as they were.
pub fn quarantine_collision(
    state: &mut DeviceState,
    mapping_id: u32,
    owner: u32,
    gpa: u64,
    site: &str,
) {
    let now_ms = crate::observe::elapsed_ms() as u64;
    let (Some(mine), Some(theirs)) = (
        Side::of(state, mapping_id, now_ms),
        Side::of(state, owner, now_ms),
    ) else {
        return;
    };
    let mut both: Vec<u64> = mine.pages.iter().chain(&theirs.pages).copied().collect();
    both.sort_unstable();
    both.dedup();
    let in_flight = in_flight(&both);
    let mut fresh = false;
    for (side, partner) in [(&mine, owner), (&theirs, mapping_id)] {
        let Some(m) = state.mappings.get_mut(&side.mapping_id) else {
            continue;
        };
        match m.quarantine.as_mut() {
            Some(q) => {
                if !q.partners.contains(&partner) {
                    q.partners.push(partner);
                    fresh = true;
                }
            }
            None => {
                m.quarantine = Some(Quarantine {
                    partners: vec![partner],
                    gpa,
                    generation: side.generation,
                    since_ms: now_ms,
                });
                fresh = true;
            }
        }
    }
    crate::runtime::drain::note_store_route("surface_quarantine");
    // Writes already submitted to either page list land before anything else
    // is decided about them.
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(
        crate::runtime::render_writeback::SettleSite::Quarantine,
        || Some(both),
    );
    if fresh {
        crate::observe::fail(format!(
            "surface_quarantine site={site} gpa={gpa:#x} new=[{}] owner=[{}] in_flight={in_flight} \
             (two live mappings claim one guest page; neither is written until a release, \
             rebind or delete of one of them says which)",
            mine.render(),
            theirs.render()
        ));
    }
}

/// A protocol event about `mapping_id` that ends its claim: lift its
/// quarantine, drop its owed frame unwritten, and lift every partner left
/// without a counterpart, keeping that partner's frame.
pub fn lift(state: &mut DeviceState, mapping_id: u32, evidence: &'static str) {
    let now_ms = crate::observe::elapsed_ms() as u64;
    let Some(q) = state
        .mappings
        .get_mut(&mapping_id)
        .and_then(|m| m.quarantine.take())
    else {
        return;
    };
    let dropped = drop_debt(state, mapping_id);
    crate::runtime::drain::note_store_route("surface_quarantine_lifted");
    crate::observe::fail(format!(
        "surface_quarantine_lifted mid={mapping_id} evidence={evidence} held_ms={} gen_then={} \
         partners={:?} debt_dropped={dropped}",
        now_ms.saturating_sub(q.since_ms),
        q.generation,
        q.partners
    ));
    for partner in q.partners {
        let orphaned = state
            .mappings
            .get_mut(&partner)
            .and_then(|m| m.quarantine.as_mut())
            .map(|pq| {
                pq.partners.retain(|&p| p != mapping_id);
                pq.partners.is_empty()
            })
            .unwrap_or(false);
        if !orphaned {
            continue;
        }
        let Some(pq) = state
            .mappings
            .get_mut(&partner)
            .and_then(|m| m.quarantine.take())
        else {
            continue;
        };
        crate::runtime::drain::note_store_route("surface_quarantine_lifted");
        crate::observe::fail(format!(
            "surface_quarantine_lifted mid={partner} evidence=partner_{evidence} partner={mapping_id} \
             held_ms={} gen_then={} debt_dropped=false (its counterpart released its claim; \
             this mapping's frame is kept)",
            now_ms.saturating_sub(pq.since_ms),
            pq.generation
        ));
    }
}

/// The released side's owed frame: dropped, never written.
fn drop_debt(state: &mut DeviceState, mapping_id: u32) -> bool {
    let Some(debt) = state.pending_writebacks.take(mapping_id) else {
        return false;
    };
    #[cfg(feature = "backend-vulkan")]
    crate::backend::vulkan::engine::note_resident_content_copied_out(&debt.identity);
    #[cfg(not(feature = "backend-vulkan"))]
    let _ = debt;
    true
}

/// A write a quarantine refused. Counted every time, logged once a mapping.
pub fn note_refused(mapping_id: u32, writer: &'static str) {
    crate::runtime::drain::note_store_route("surface_quarantine_refused");
    if crate::observe::first_sight("surface_quarantine_refused", u64::from(mapping_id)) {
        crate::observe::fail(format!(
            "surface_quarantine_refused mid={mapping_id} writer={writer} (a write to a mapping \
             whose pages another live mapping also claims; not written)"
        ));
    }
}
