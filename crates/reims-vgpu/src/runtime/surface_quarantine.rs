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
//! A quarantine is lifted only by protocol evidence about that mapping itself:
//! its UNMAP, or a new MAP or MappingInternal for its id *followed by* a resolve
//! that adopts a page table different from the one it was quarantined with. A
//! MAP notification alone is not that: it can trail the resolve that already
//! adopted the stale table, and a delete that only condemns the table can be
//! reprieved by re-reading it.
//! A partner's release is not such evidence: it says the partner no longer owns
//! the pages, not that this mapping owns them again — the stale side of a
//! collision whose new surfaces are later unmapped stays quarantined. Re-reading
//! the same page table lifts nothing. There is no page blacklist: the pages are
//! free to belong to whoever the next protocol event names.

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
    /// The page entries it held when quarantined: a new backing is a table
    /// other than this one.
    pub entries: Vec<u32>,
    /// A MAP or new MappingInternal arrived for this id since; the next resolve
    /// adopting a different table lifts the quarantine.
    pub rebind_seen: Option<&'static str>,
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
        // Every page proof minted before this is void, whatever happens later.
        m.write_epoch = m.write_epoch.wrapping_add(1);
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
                    entries: m.page_entries.clone(),
                    rebind_seen: None,
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

/// A protocol event about `mapping_id` itself that ends its claim: lift its
/// quarantine and drop its owed frame unwritten. Its partners stay
/// quarantined, even with no partner left: see the module doc.
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
    // The partners keep their quarantine; only the record of who they collided
    // with is updated, for their own lift line.
    for partner in q.partners {
        if let Some(pq) = state
            .mappings
            .get_mut(&partner)
            .and_then(|m| m.quarantine.as_mut())
        {
            pq.partners.retain(|&p| p != mapping_id);
        }
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

/// A MAP, or a new MappingInternal, arrived for a quarantined id. It does not
/// lift anything by itself; it arms the lift that [`confirm_new_backing`]
/// performs when a resolve adopts a different table.
pub fn note_rebind(state: &mut DeviceState, mapping_id: u32, what: &'static str) {
    let Some(q) = state
        .mappings
        .get_mut(&mapping_id)
        .and_then(|m| m.quarantine.as_mut())
    else {
        return;
    };
    if q.rebind_seen.is_none() {
        q.rebind_seen = Some(what);
        crate::observe::fail(format!(
            "surface_quarantine_rebind_seen mid={mapping_id} evidence={what} (held until a \
             resolve adopts a page table other than the quarantined one)"
        ));
    }
}

/// A resolve has just adopted `mapping_id`'s page table. If a rebind was seen
/// and the table is not the one quarantined, the guest has given the id a new
/// backing: lift. The same table re-announced lifts nothing.
pub fn confirm_new_backing(state: &mut DeviceState, mapping_id: u32) {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return;
    };
    let Some(q) = m.quarantine.as_ref() else {
        return;
    };
    let Some(what) = q.rebind_seen else {
        return;
    };
    if m.page_entries.is_empty() || m.page_entries == q.entries {
        return;
    }
    lift(
        state,
        mapping_id,
        match what {
            "new_internal" => "new_internal_new_backing",
            _ => "map_new_backing",
        },
    );
}

/// What a payment whose store wrote nothing owes afterwards.
#[derive(Debug, PartialEq, Eq)]
pub enum RefusedStore {
    /// The store found the mapping quarantined, possibly by discovering the
    /// collision itself: the frame is back in the ledger, not lost, and not
    /// copied out.
    Deferred,
    /// Any other refusal: the frame is lost, as before.
    Lost(crate::runtime::writeback_debt::WritebackDebt),
}

/// Decide a payment whose store returned without writing.
pub fn after_refused_store(
    state: &mut DeviceState,
    mapping_id: u32,
    debt: crate::runtime::writeback_debt::WritebackDebt,
) -> RefusedStore {
    if !is_quarantined(state, mapping_id) {
        return RefusedStore::Lost(debt);
    }
    state.pending_writebacks.restore(mapping_id, debt);
    note_refused(mapping_id, "debt_store");
    RefusedStore::Deferred
}
