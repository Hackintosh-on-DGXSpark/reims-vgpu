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
//! its UNMAP, or a new MappingInternal incarnation for its id whose table, once
//! adopted and past every check, shares no valid page with the quarantined one.
//! A MAP notification is not that: it can trail the resolve that already adopted
//! the stale table. Neither is any change to the table, nor a delete that only
//! condemns it, which a reprieve re-reads.
//! A partner's release is not such evidence: it says the partner no longer owns
//! the pages, not that this mapping owns them again — the stale side of a
//! collision whose new surfaces are later unmapped stays quarantined. Re-reading
//! the same page table lifts nothing. There is no page blacklist: the pages are
//! free to belong to whoever the next protocol event names.
//!
//! # The one exception: an incarnation that withdrew its claims
//!
//! With `REIMS_VGPU_WIRE_GUARD`, [`crate::runtime::mapper::wire_watch`] can prove
//! the guest unwired a surface it still had mapped: its own wire counts, read
//! through a layout checked against the kernel, dropped below what `MAP` saw.
//! That incarnation then withdraws its page claims ([`withdraw`]): it is
//! quarantined for good, in-flight GPU writes to it are waited for, its owed
//! frame is dropped, and only then do collision checks stop counting it as an
//! owner. A new mapping over its pages is checked against every other owner as
//! usual, and one already quarantined because of it loses only that partner,
//! and is freed only after a fresh validation of its own. Without that proof
//! both sides of a collision stay quarantined, as above.

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
    /// The MappingInternal of an incarnation that withdrew its page claims: see
    /// [`withdraw`]. Only that incarnation; a new one for the id is not.
    pub withdrawn: Option<u64>,
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
                    withdrawn: None,
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
    // A new MappingInternal outranks a MAP: it is the one that can confirm.
    if q.rebind_seen.is_none() || what == "new_internal" {
        q.rebind_seen = Some(what);
        crate::observe::fail(format!(
            "surface_quarantine_rebind_seen mid={mapping_id} evidence={what} (held until a \
             resolve adopts a new MappingInternal's table that shares no valid page with the \
             quarantined one)"
        ));
    }
}

/// A resolve has just adopted `mapping_id`'s page table and passed every check
/// on it. The quarantine lifts only for a new backing incarnation: a new
/// MappingInternal for the id since the quarantine, whose table has valid
/// pages and shares none with the quarantined table's.
///
/// A table merely different is not that. A late MAP re-announces the stale
/// table; filling a hole, shortening the list or changing an invalid entry
/// keeps the old pages; and one changed page says nothing about the rest of a
/// window whose pages the guest may already have given away. Without the
/// stronger signal the mapping stays quarantined: refusing a write costs a
/// frame, and a wrong lift costs guest memory.
pub fn confirm_new_backing(state: &mut DeviceState, mapping_id: u32) {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return;
    };
    let Some(q) = m.quarantine.as_ref() else {
        return;
    };
    if q.rebind_seen != Some("new_internal") {
        return;
    }
    let shift = state.page_shift;
    let valid = |entries: &[u32]| -> std::collections::BTreeSet<u64> {
        entries
            .iter()
            .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, shift))
            .collect()
    };
    let (old, new) = (valid(&q.entries), valid(&m.page_entries));
    if new.is_empty() || !old.is_disjoint(&new) {
        crate::runtime::drain::note_store_route("surface_quarantine_rebind_rejected");
        return;
    }
    lift(state, mapping_id, "new_internal_disjoint_backing");
}

// MARK: - Withdrawn claims

/// Whether this mapping's current incarnation has withdrawn its page claims.
/// See [`withdraw`].
pub fn entry_withdrawn(m: &crate::model::MappingEntry) -> bool {
    m.mapping_internal != 0
        && m.quarantine.as_ref().and_then(|q| q.withdrawn) == Some(m.mapping_internal)
}

/// [`entry_withdrawn`] by id.
pub fn is_withdrawn(state: &DeviceState, mapping_id: u32) -> bool {
    state.mappings.get(&mapping_id).is_some_and(entry_withdrawn)
}

/// The wire guard proved the guest unwired `mapping_id`'s incarnation
/// `internal` while it was still mapped: that incarnation withdraws its page
/// claims. `evidence` is the guard's, for the log.
///
/// The steps run in this order because each licenses the next:
///
/// 1. its write authority is revoked: it is quarantined, which voids every page
///    proof minted for it and refuses every CPU, import and direct write to
///    the whole mapping;
/// 2. GPU writes already submitted to its pages are waited for;
/// 3. its owed frame is dropped, never landed (the resident is kept: a later
///    present of the surface shows it without touching guest memory);
/// 4. only then is it marked withdrawn, and collision checks stop counting it
///    as an owner of any page.
///
/// Nothing restores it. Its UNMAP ends it, and a new MappingInternal for the id
/// is another incarnation, which [`entry_withdrawn`] does not match.
///
/// Every mapping quarantined because of this one loses exactly that partner and
/// keeps any others. Returned, with the partners each still has, for the caller
/// to validate afresh and free with [`release_withdrawn_partner`]; nothing here
/// frees them.
pub fn withdraw(
    state: &mut DeviceState,
    mapping_id: u32,
    internal: u64,
    evidence: &str,
) -> Option<Vec<(u32, Vec<u32>)>> {
    let now_ms = crate::observe::elapsed_ms() as u64;
    let shift = state.page_shift;
    let m = state.mappings.get_mut(&mapping_id)?;
    if internal == 0 || m.mapping_internal != internal || entry_withdrawn(m) {
        return None;
    }
    m.write_epoch = m.write_epoch.wrapping_add(1);
    if m.quarantine.is_none() {
        m.quarantine = Some(Quarantine {
            partners: Vec::new(),
            gpa: 0,
            generation: m.map_generation,
            since_ms: now_ms,
            entries: m.page_entries.clone(),
            rebind_seen: None,
            withdrawn: None,
        });
    }
    let quarantined = m
        .quarantine
        .as_ref()
        .map(|q| q.entries.as_slice())
        .unwrap_or(&[]);
    let mut pages: Vec<u64> = m
        .page_entries
        .iter()
        .chain(quarantined)
        .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, shift))
        .collect();
    pages.sort_unstable();
    pages.dedup();
    let in_flight = in_flight(&pages);
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(
        crate::runtime::render_writeback::SettleSite::Quarantine,
        || Some(pages.clone()),
    );
    let debt = state.pending_writebacks.take(mapping_id).is_some();
    if let Some(q) = state
        .mappings
        .get_mut(&mapping_id)
        .and_then(|m| m.quarantine.as_mut())
    {
        q.withdrawn = Some(internal);
    }
    let mut freed = Vec::new();
    for (&id, other) in state.mappings.iter_mut() {
        if id == mapping_id {
            continue;
        }
        if let Some(q) = other.quarantine.as_mut() {
            if q.partners.contains(&mapping_id) {
                q.partners.retain(|&p| p != mapping_id);
                freed.push((id, q.partners.clone()));
            }
        }
    }
    crate::runtime::drain::note_store_route("surface_quarantine_withdrawn");
    crate::observe::fail(format!(
        "surface_quarantine_withdrawn mid={mapping_id} internal={internal:#x} pages={} \
         in_flight_before={in_flight} debt={} freed={:?} evidence=[{evidence}] (the guest unwired \
         this incarnation while it was mapped: its writes stay refused, in-flight writes to it \
         have settled, and it no longer owns any page in a collision check)",
        pages.len(),
        if debt { "dropped" } else { "none" },
        freed.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    ));
    Some(freed)
}

/// Free `mapping_id`, whose only remaining partner `old` has withdrawn its
/// claims, after the caller validated it afresh (`evidence`). Its owed frame
/// stays owed: it is the mapping's own and lands as any other.
pub fn release_withdrawn_partner(
    state: &mut DeviceState,
    mapping_id: u32,
    old: u32,
    evidence: &str,
) {
    let now_ms = crate::observe::elapsed_ms() as u64;
    let Some(m) = state.mappings.get_mut(&mapping_id) else {
        return;
    };
    if !m.quarantine.as_ref().is_some_and(|q| q.partners.is_empty()) {
        return;
    }
    let Some(q) = m.quarantine.take() else {
        return;
    };
    crate::runtime::drain::note_store_route("surface_quarantine_resolved");
    crate::observe::fail(format!(
        "surface_quarantine_resolved reason=owner_condemned old=mid={old} new=mid={mapping_id} \
         order=collision_first held_ms={} owes={} evidence=[{evidence}] (the old incarnation \
         withdrew its claims; this mapping was validated afresh and is written again)",
        now_ms.saturating_sub(q.since_ms),
        state.pending_writebacks.get(mapping_id).is_some()
    ));
}

/// A collision check found `mapping_id`'s pages overlapping `old`'s withdrawn
/// incarnation and did not count it: said once a pair. Not said while
/// `mapping_id` is itself held, where [`release_withdrawn_partner`] speaks.
pub fn note_withdrawn_overlap(mapping_id: u32, old: u32, gpa: u64, site: &str, held: bool) {
    crate::runtime::drain::note_store_route("surface_quarantine_withdrawn_skipped");
    if held {
        return;
    }
    let key = (u64::from(mapping_id) << 32) | u64::from(old);
    if crate::observe::first_sight("surface_quarantine_resolved", key) {
        crate::observe::fail(format!(
            "surface_quarantine_resolved reason=owner_condemned old=mid={old} new=mid={mapping_id} \
             order=condemned_first site={site} gpa={gpa:#x} (the old incarnation withdrew its \
             claims; the new mapping is checked against every other owner)"
        ));
    }
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
