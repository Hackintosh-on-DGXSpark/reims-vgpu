//! The guest's wire count on every type-11 surface this device maps, sampled
//! where this device acts, with the first time it goes wrong said out loud.
//!
//! Off unless `REIMS_VGPU_WIRE_WATCH` is on. Nothing here changes what the guest
//! sees, except the optional guard below.
//!
//! # Why
//!
//! Two macOS 27.2 panics were read back out of guest RAM. In both, a mapped
//! display swap buffer (1920×1080 RGBA16F, `MAP`ped once, never `UNMAP`ped,
//! still in the mapper's wired set) had its surface descriptor at a wire count
//! of zero while every other mapping stood at one. Its pages were then
//! compressed and reused by the kernel, and this device's next full-frame write
//! into the mapping's page list landed on kernel memory.
//!
//! `IOSurfaceParavirtMappingInternal::wire_mapping` prepares the surface
//! descriptor and the page-table descriptor before it sends `MAP`, and
//! `unwire_mapping` completes both only after `UNMAP` succeeds. So between those
//! two requests the guest's own driver keeps both counts at or above their
//! `MAP`-time values. Something else completed the surface once too often; the
//! static reading points at the display swap path. This module records the
//! first moment the count moves, beside the swap, present, stamp, hold and
//! write events this device produced around it.
//!
//! # What it reads, and why it believes it
//!
//! The offsets below are 27.2's (26B123), inferred from the driver's code and
//! checked against both dumps. Nothing here is trusted before it is checked:
//!
//! 1. The kernel. The Mach-O header of the kernel is found by stepping down
//!    page by page from `VBAR_EL1` (captured with the kernel root, see
//!    [`crate::runtime::kernel_walk`]); its `__TEXT` must start at the header
//!    itself, `VBAR_EL1` must lie in one of its segments, and `__TEXT,__const`
//!    must hold exactly [`EXPECTED_KERNEL_VERSION`]. Any other kernel turns the
//!    watch off for the boot.
//! 2. Each mapping. The `MappingInternal` must name the mapping's id, the
//!    `0x200` descriptor size and the page count this device resolved; the
//!    page-table descriptor's buffer must be the page table this device reads;
//!    the surface descriptor's length must be the mapping's size; both
//!    descriptors must be initialised and of one kernel class. A mapping that
//!    fails any of these is not watched.
//!
//! A failed check is `wire_watch_layout_unverified`. A read that could not be
//! made — off a vCPU before the kernel walk validated — is not a failure and is
//! tried again; it is counted as `unreadable`.
//!
//! # When it samples
//!
//! On `MAP` (on the publishing vCPU, after the resolve), before `UNMAP`, before
//! every host write into a watched mapping (a write within
//! [`WRITE_RESAMPLE_US`] of the last read uses it, so a frame landed row by row
//! is not read a thousand times), around every display swap packet, after every
//! stamp that completes one, and from the drain worker: the swap buffers every
//! [`DISPLAY_SAMPLE_US`] and every mapping every [`ALL_SAMPLE_US`]. A queued
//! stamp reaches the guest later than it is sampled here; the worker's next
//! pass is what sees the guest's answer to it.
//!
//! While a FIFO is held, its head packet is read without consuming it every
//! [`PEEK_US`], so a display swap waiting behind the hold is seen when it
//! arrives and not only when it runs.
//!
//! # The guard, and what it is not
//!
//! With `REIMS_VGPU_WIRE_GUARD` on as well, the first anomaly condemns the
//! mapping's incarnation (`wire_guard_condemned`): the guest completed a
//! descriptor it still had mapped, which on 27.2 is IOMFB cancelling a swap that
//! waited on this device past its five-second limit. From then on, for as long
//! as the `MappingInternal` is the same, every host write into it, CPU or GPU,
//! is refused (`wire_guard_refused`), its owed frame is dropped rather than
//! landed, and a display present naming it that was already published in its
//! FIFO when the drop was seen is refused (`wire_guard_swap_refused`): its stamp
//! is written so the FIFO moves, but it is not shown and raises no present
//! event. That refusal does not claim the guest cancelled that request; see
//! [`refuse_swap`]. None of that is undone if a count turns positive again
//! (`wire_guard_still_condemned`). The resident stays, so a later present of the
//! surface shows it without writing guest memory.
//!
//! A positive count is not a permission: it is read, then the write happens,
//! and the guest can complete the descriptor in between, so the guard only
//! stops writes after a drop it has seen. A render target the GPU writes in
//! guest memory (`REIMS_VGPU_SHARED_TARGET`) is not seen at all. A reader of the
//! mapping reads whatever the guest pages hold.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::MapperMem;
use crate::contract::iosurface_pages::{
    self, guest_kernel_va, PagesMemory, MAPPING_INTERNAL_EXPECTED_SIZE,
};
use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::kernel_walk::{self, KernelRoot};

// MARK: - Switches

/// Whether the watch is on for this process. `REIMS_VGPU_WIRE_WATCH`, read once.
pub fn enabled() -> bool {
    #[cfg(test)]
    if let Some((watch, _)) = TEST_SWITCHES.with(std::cell::Cell::get) {
        return watch;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let (state, value) = crate::env::read(crate::env::WIRE_WATCH);
        let on = state == crate::env::Switch::On;
        if value.is_some() {
            crate::observe::off(format!(
                "wire_watch_mode on={on} guard={} value={}",
                guard_switch(),
                value.unwrap_or_default()
            ));
        }
        on
    })
}

fn guard_switch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::env::switch(crate::env::WIRE_GUARD) == crate::env::Switch::On)
}

/// Whether a zero count refuses a host CPU write. Needs the watch as well.
fn guard_enabled() -> bool {
    #[cfg(test)]
    if let Some((watch, guard)) = TEST_SWITCHES.with(std::cell::Cell::get) {
        return watch && guard;
    }
    enabled() && guard_switch()
}

#[cfg(test)]
thread_local! {
    static TEST_SWITCHES: std::cell::Cell<Option<(bool, bool)>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with the watch and the guard switched as given on this thread.
#[cfg(test)]
pub(crate) fn with_switches<R>(watch: bool, guard: bool, f: impl FnOnce() -> R) -> R {
    TEST_SWITCHES.with(|s| s.set(Some((watch, guard))));
    let out = f();
    TEST_SWITCHES.with(|s| s.set(None));
    out
}

// MARK: - Layout

/// `version[]` of the kernel the offsets below were read from.
pub const EXPECTED_KERNEL_VERSION: &str = "Darwin Kernel Version 27.2.0: Wed Oct 14 23:44:09 PDT \
     2026; root:xnu-13432.41.4~25/RELEASE_ARM64_VMAPPLE";
const VERSION_PREFIX: &[u8] = b"Darwin Kernel Version ";
/// Longest version string read back for the log.
const VERSION_MAX: usize = 256;

const MH_MAGIC_64: u32 = 0xfeed_facf;
const CPU_TYPE_ARM64: u32 = 0x0100_000c;
const MH_EXECUTE: u32 = 2;
const LC_SEGMENT_64: u32 = 0x19;
const MACH_HEADER_64_LEN: usize = 32;
const SEGMENT_COMMAND_64_LEN: usize = 72;
const SECTION_64_LEN: usize = 80;
/// Load commands larger than this are not a kernel header.
const LOAD_COMMANDS_MAX: usize = 0x10000;
/// `__TEXT,__const` larger than this is not read.
const TEXT_CONST_MAX: u64 = 8 << 20;
/// How far below `VBAR_EL1` the kernel header is looked for. On 27.2 it is
/// 0x2ec0000 below; this is a bound on the search, not a guess at the answer.
const KERNEL_SCAN_BYTES: u64 = 256 << 20;

/// `IOSurfaceParavirtMappingInternal`, 27.2. `+0x10` is the driver's own
/// `refs * 4 | wired` word, `+0x20` the surface descriptor, `+0x58` the
/// page-table descriptor (macOS 14 and later moved the page fields by
/// [`iosurface_pages::MAPPING_INTERNAL_SONOMA_SHIFT`]).
const MI_STATE: u64 = 0x10;
const MI_SURFACE_IOMD: u64 = 0x20;
const MI_ID: u64 = iosurface_pages::MAPPING_INTERNAL_ID;
const MI_SIZE: u64 = iosurface_pages::MAPPING_INTERNAL_SIZE;
const MI_FIELD_48: u64 = iosurface_pages::MAPPING_INTERNAL_PAGE_FIELD_48;
const MI_FIELD_50: u64 = iosurface_pages::MAPPING_INTERNAL_PAGE_FIELD_50;
const MI_TABLE_IOMD: u64 = iosurface_pages::MAPPING_INTERNAL_PAGE_FIELD_48
    + iosurface_pages::MAPPING_INTERNAL_SONOMA_SHIFT;
const MI_PAGE_COUNT: u64 =
    iosurface_pages::MAPPING_INTERNAL_PAGE_COUNT + iosurface_pages::MAPPING_INTERNAL_SONOMA_SHIFT;
/// One read covers every field above.
const MI_WINDOW: std::ops::Range<u64> = MI_STATE..MI_PAGE_COUNT + 8;

/// `IOBufferMemoryDescriptor` on 27.2's xnu: `IOMemoryDescriptor::_length`,
/// `IOGeneralMemoryDescriptor::_task`, `_wireCount`, `_initialized`,
/// `_memoryEntries`, `_pages`, and `IOBufferMemoryDescriptor::_buffer`.
const IOMD_LENGTH: u64 = 0x50;
const IOMD_TASK: u64 = 0x70;
const IOMD_WIRE_COUNT: u64 = 0x88;
const IOMD_INITIALIZED: u64 = 0x8c;
const IOMD_MEMORY_ENTRIES: u64 = 0x90;
const IOMD_PAGES: u64 = 0x98;
const IOBMD_BUFFER: u64 = iosurface_pages::MAPPING_PAGE_TABLE_FROM_F48;
const IOMD_WINDOW: u64 = IOBMD_BUFFER + 8;
/// The length word of `_memoryEntries`. `0x30` once `complete()` has dropped
/// the page list, larger while wired.
const MIXED_DATA_LENGTH: u64 = 0x18;
/// Bytes per page-table entry, as [`iosurface_pages`] reads them.
const TABLE_ENTRY_LEN: u64 = iosurface_pages::U32_SIZE as u64;

// MARK: - Cadence

/// Swap buffers are sampled from the worker at most this often.
pub const DISPLAY_SAMPLE_US: u64 = 5_000;
/// Every watched mapping is sampled from the worker at most this often.
pub const ALL_SAMPLE_US: u64 = 100_000;
/// A held FIFO's head is read at most this often.
pub const PEEK_US: u64 = 20_000;
/// The cost line.
const COST_US: u64 = 10_000_000;
/// The history a first anomaly prints. A window over the most recent events
/// for the log, not a store anything is looked up in.
const RING: usize = 16;
/// Lines one mapping may log of each repeating kind, at its 1st, 2nd, 4th, 8th...
/// occurrence; the count carries on.
const LINES_PER_MAPPING: u32 = 24;
/// A write this soon after the counts were read uses that read. A writer that
/// lands a frame a row at a time calls once per row.
pub const WRITE_RESAMPLE_US: u64 = 250;
/// Completed swaps waiting for their stamp, oldest dropped past this (counted).
const UNSTAMPED_MAX: usize = 64;

fn now_us() -> u64 {
    crate::observe::elapsed_us()
}

// MARK: - Events

#[derive(Clone, Debug, PartialEq, Eq)]
enum Ev {
    Swap {
        channel: u32,
        head: u32,
        opcode: u16,
        stamp: u32,
        slot: u32,
        mapping: u32,
        how: SwapHow,
        present: &'static str,
    },
    SwapHeld {
        cause: &'static str,
        channel: u32,
        head: u32,
        opcode: u16,
        stamp: u32,
        mapping: u32,
    },
    Stamp {
        slot: u32,
        value: u32,
        queued: bool,
        swaps: Vec<(u64, u32)>,
    },
    Present {
        kind: &'static str,
    },
    Hold {
        held: u32,
        new: u32,
        producer: u32,
    },
    HoldRelease {
        held: u32,
        held_us: u64,
    },
    Map {
        mapping: u32,
        surface: u32,
        table: u32,
    },
    Unmap {
        mapping: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Event {
    seq: u64,
    t_us: u64,
    /// Display refresh-class present signals raised so far; see
    /// [`note_refresh_signal`].
    refresh: u64,
    ev: Ev,
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "seq={} t_us={} refresh={} ",
            self.seq, self.t_us, self.refresh
        )?;
        match &self.ev {
            Ev::Swap {
                channel,
                head,
                opcode,
                stamp,
                slot,
                mapping,
                how,
                present,
            } => write!(
                f,
                "ev=swap ch={channel} head={head:#x} op={opcode:#x} stamp={stamp:#x} slot={slot} \
                 mapping={mapping} how={} present={present}",
                how.name()
            ),
            Ev::SwapHeld {
                cause,
                channel,
                head,
                opcode,
                stamp,
                mapping,
            } => write!(
                f,
                "ev=swap_held cause={cause} ch={channel} head={head:#x} op={opcode:#x} \
                 stamp={stamp:#x} mapping={mapping}"
            ),
            Ev::Stamp {
                slot,
                value,
                queued,
                swaps,
            } => {
                write!(
                    f,
                    "ev=stamp slot={slot} value={value:#x} route={} swaps=[",
                    if *queued { "queued" } else { "direct" }
                )?;
                for (i, (seq, mapping)) in swaps.iter().enumerate() {
                    let sep = if i == 0 { "" } else { "," };
                    write!(f, "{sep}{seq}:{mapping}")?;
                }
                write!(f, "]")
            }
            Ev::Present { kind } => write!(f, "ev=present kind={kind}"),
            Ev::Hold {
                held,
                new,
                producer,
            } => write!(
                f,
                "ev=hold held_mask={held:#x} new_mask={new:#x} producer_mask={producer:#x}"
            ),
            Ev::HoldRelease { held, held_us } => {
                write!(f, "ev=hold_release held_mask={held:#x} held_us={held_us}")
            }
            Ev::Map {
                mapping,
                surface,
                table,
            } => write!(
                f,
                "ev=map mapping={mapping} surface={surface} table={table}"
            ),
            Ev::Unmap { mapping } => write!(f, "ev=unmap mapping={mapping}"),
        }
    }
}

/// One display swap the device completed, waiting for the stamp that tells the
/// guest so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Unstamped {
    seq: u64,
    slot: u32,
    stamp: u32,
    mapping: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WriteNote {
    mapping: u32,
    site: String,
    t_us: u64,
}

impl std::fmt::Display for WriteNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{mapping={},site={},t_us={}}}",
            self.mapping, self.site, self.t_us
        )
    }
}

// MARK: - State

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct KernelImage {
    header: u64,
    data_const: (u64, u64),
    va_bits: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum KernelCheck {
    #[default]
    Unchecked,
    Verified(KernelImage),
    /// Read, and not the kernel these offsets belong to. Off for the boot.
    Unverified,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    surface: u32,
    table: u32,
    /// `MappingInternal +0x10`.
    mi_state: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    Surface,
    Table,
    MiState,
}

impl Field {
    fn name(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::Table => "table",
            Self::MiState => "mi_state",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// The `MappingInternal` is known and its layout has not been checked yet.
    Pending,
    Active,
}

/// What the descriptor held when it was checked, for a later comparison that
/// says whether anything besides the count moved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct IomdFields {
    vtable: u64,
    length: u64,
    task: u64,
    wire: u32,
    initialized: u8,
    memory_entries: u64,
    pages: u32,
    buffer: u64,
}

impl IomdFields {
    fn decode(b: &[u8]) -> Self {
        let u64_at = |o: u64| u64::from_le_bytes(b[o as usize..o as usize + 8].try_into().unwrap());
        let u32_at = |o: u64| u32::from_le_bytes(b[o as usize..o as usize + 4].try_into().unwrap());
        Self {
            vtable: u64_at(0),
            length: u64_at(IOMD_LENGTH),
            task: u64_at(IOMD_TASK),
            wire: u32_at(IOMD_WIRE_COUNT),
            initialized: b[IOMD_INITIALIZED as usize],
            memory_entries: u64_at(IOMD_MEMORY_ENTRIES),
            pages: u32_at(IOMD_PAGES),
            buffer: u64_at(IOBMD_BUFFER),
        }
    }

    /// Every field but the count and the mixed-data object, which `prepare()`
    /// and `complete()` are allowed to move.
    fn differs_besides_wiring(&self, other: &Self) -> Vec<&'static str> {
        let mut out = Vec::new();
        for (name, a, b) in [
            ("vtable", self.vtable, other.vtable),
            ("length", self.length, other.length),
            ("task", self.task, other.task),
            ("pages", u64::from(self.pages), u64::from(other.pages)),
            ("buffer", self.buffer, other.buffer),
            ("memory_entries", self.memory_entries, other.memory_entries),
            (
                "initialized",
                u64::from(self.initialized),
                u64::from(other.initialized),
            ),
        ] {
            if a != b {
                out.push(name);
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
struct Watched {
    internal: u64,
    phase: Phase,
    surface_iomd: u64,
    table_iomd: u64,
    pages: u64,
    surface_fields: IomdFields,
    table_fields: IomdFields,
    /// `None` when the layout was checked after `MAP`, so the `MAP`-time value
    /// was never seen.
    at_map: Option<Counts>,
    last: Counts,
    /// Named by a display swap.
    display: bool,
    anomalous: BTreeSet<&'static str>,
    writes: u64,
    writes_after_anomaly: u64,
    lines_after_anomaly: u32,
    guard_lines: u32,
    refused: u64,
    last_write: Option<WriteNote>,
    /// When the counts were last read.
    sampled_us: Option<u64>,
    /// Set by the guard at the first anomaly: this incarnation is the guest's no
    /// longer, whatever its counts read afterwards.
    condemned: Option<Condemned>,
    /// A count went positive again after the condemnation; said once.
    recovered_said: bool,
}

/// A mapping incarnation the guest unwired while it was still mapped.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Condemned {
    seq: u64,
    t_us: u64,
    field: &'static str,
    /// Where each child FIFO stood when the unwire was seen.
    horizon: Vec<Horizon>,
}

/// One child FIFO's consumer and producer positions when an unwire was seen,
/// and which incarnation of that FIFO they belong to.
///
/// Heads and tails are free-running offsets of one ring. A guest that frees
/// and redefines the FIFO, or points it at another ring, starts a new sequence
/// from wherever it likes, usually zero, so a position is compared only with
/// positions of the same incarnation: the same define generation (bumped by
/// `DefineFifo` and `FreeFifo`, see [`on_fifo_lifecycle`]) and the same ring
/// base. Anything else and the horizon says nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Horizon {
    channel: u32,
    generation: u64,
    base_pfn: u32,
    head: u32,
    tail: u32,
}

impl Horizon {
    /// Whether the packet at `head` of this FIFO incarnation was published
    /// when the horizon was taken and not yet consumed.
    fn holds(&self, head: u32) -> bool {
        head.wrapping_sub(self.head) as i32 >= 0 && self.tail.wrapping_sub(head) as i32 > 0
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Cost {
    samples: u64,
    reads: u64,
    unreadable: u64,
    us: u64,
    transitions: u64,
    anomalies: u64,
    guard_refused: u64,
    unstamped_dropped: u64,
    peeks: u64,
    condemned: u64,
    debts_dropped: u64,
    swaps_refused: u64,
}

/// The watch's state for one boot. Lives in [`DeviceState`] and is replaced
/// with it at a reset.
#[derive(Clone, Debug, Default)]
pub struct WireWatch {
    kernel: KernelCheck,
    watched: BTreeMap<u32, Watched>,
    /// Mappings whose layout failed, by `MappingInternal`: said once.
    unverified: BTreeSet<(u32, u64)>,
    ring: VecDeque<Event>,
    seq: u64,
    last_swap: Option<Event>,
    last_stamp: Option<Event>,
    last_write: Option<WriteNote>,
    unstamped: VecDeque<Unstamped>,
    /// The present signal raised while the swap being executed ran.
    present_in_swap: Option<&'static str>,
    hold_since_us: Option<u64>,
    /// Held swaps already reported, by `(channel, head, stamp)`.
    held_seen: BTreeSet<(u32, u32, u32)>,
    first_anomaly_seen: bool,
    /// When each cadence last ran; `None` runs it on the next tick.
    display_sampled_us: Option<u64>,
    all_sampled_us: Option<u64>,
    peeked_us: Option<u64>,
    cost_logged_us: Option<u64>,
    /// Mappings the guard condemned whose debts and horizon are still to be
    /// taken, which needs the whole device state.
    condemn_pending: Vec<u32>,
    /// How often each child FIFO was defined or freed; see [`Horizon`].
    fifo_generations: BTreeMap<u32, u64>,
    cost: Cost,
}

impl WireWatch {
    fn push(&mut self, ev: Ev) -> Event {
        self.seq += 1;
        let event = Event {
            seq: self.seq,
            t_us: now_us(),
            refresh: REFRESH_SIGNALS.load(std::sync::atomic::Ordering::Relaxed),
            ev,
        };
        if self.ring.len() == RING {
            self.ring.pop_front();
        }
        self.ring.push_back(event.clone());
        event
    }

    fn context(&self) -> String {
        let brace = |e: &Option<Event>| match e {
            Some(e) => format!("{{{e}}}"),
            None => "-".to_string(),
        };
        format!(
            "last_swap={} last_stamp={} last_write={}",
            brace(&self.last_swap),
            brace(&self.last_stamp),
            self.last_write
                .as_ref()
                .map_or_else(|| "-".to_string(), WriteNote::to_string)
        )
    }

    fn fifo_generation(&self, channel: u32) -> u64 {
        self.fifo_generations.get(&channel).copied().unwrap_or(0)
    }

    fn off(&self) -> bool {
        self.kernel == KernelCheck::Unverified
    }

    fn cost_line(&self, boot: u64) -> String {
        let c = self.cost;
        format!(
            "wire_watch_cost boot={boot} watched={} samples={} reads={} unreadable={} us={} \
             transitions={} anomalies={} guard_refused={} condemned={} debts_dropped={} \
             swaps_refused={} peeks={} unstamped_dropped={} kernel={}",
            self.watched.len(),
            c.samples,
            c.reads,
            c.unreadable,
            c.us,
            c.transitions,
            c.anomalies,
            c.guard_refused,
            c.condemned,
            c.debts_dropped,
            c.swaps_refused,
            c.peeks,
            c.unstamped_dropped,
            match self.kernel {
                KernelCheck::Unchecked => "unchecked",
                KernelCheck::Verified(_) => "verified",
                KernelCheck::Unverified => "unverified",
            }
        )
    }
}

// MARK: - Reading

/// Kernel reads for the watch: through the CPU on a vCPU, through the kernel
/// walk elsewhere once it has validated, and nothing otherwise.
struct Reader<'a, H: HostMemory + HostOps> {
    mem: MapperMem<'a, H>,
    reads: std::cell::Cell<u64>,
    failed: std::cell::Cell<u64>,
}

impl<'a, H: HostMemory + HostOps> Reader<'a, H> {
    fn new(state: &DeviceState, host: &'a H) -> Self {
        // The strict policy whatever the mode: compat only says what a write
        // falls back to, and the watch never writes.
        let walk = state.kernel_walk.policy(kernel_walk::Mode::Strict);
        Self {
            mem: MapperMem::with_walk(host, walk, None),
            reads: std::cell::Cell::new(0),
            failed: std::cell::Cell::new(0),
        }
    }

    fn bytes(&self, kva: u64, buf: &mut [u8]) -> bool {
        self.reads.set(self.reads.get() + 1);
        let ok = guest_kernel_va(kva) && self.mem.read(kva, buf);
        if !ok {
            self.failed.set(self.failed.get() + 1);
        }
        ok
    }

    fn u32(&self, kva: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.bytes(kva, &mut b).then(|| u32::from_le_bytes(b))
    }

    fn u64(&self, kva: u64) -> Option<u64> {
        let mut b = [0u8; 8];
        self.bytes(kva, &mut b).then(|| u64::from_le_bytes(b))
    }

    fn fold(&self, cost: &mut Cost, started: std::time::Instant) {
        cost.reads += self.reads.get();
        cost.unreadable += self.failed.get();
        cost.us += started.elapsed().as_micros() as u64;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MiFields {
    state: u32,
    surface: u64,
    id: u32,
    size: u32,
    field_48: u64,
    field_50: u64,
    table: u64,
    pages: u64,
}

fn read_mi<H: HostMemory + HostOps>(rd: &Reader<'_, H>, internal: u64) -> Option<MiFields> {
    let mut b = [0u8; (MI_WINDOW.end - MI_WINDOW.start) as usize];
    if !rd.bytes(internal.checked_add(MI_WINDOW.start)?, &mut b) {
        return None;
    }
    let at = |o: u64| (o - MI_WINDOW.start) as usize;
    let u64_at = |o: u64| u64::from_le_bytes(b[at(o)..at(o) + 8].try_into().unwrap());
    let u32_at = |o: u64| u32::from_le_bytes(b[at(o)..at(o) + 4].try_into().unwrap());
    Some(MiFields {
        state: u32_at(MI_STATE),
        surface: u64_at(MI_SURFACE_IOMD),
        id: u32_at(MI_ID),
        size: u32_at(MI_SIZE),
        field_48: u64_at(MI_FIELD_48),
        field_50: u64_at(MI_FIELD_50),
        table: u64_at(MI_TABLE_IOMD),
        pages: u64_at(MI_PAGE_COUNT),
    })
}

fn read_iomd<H: HostMemory + HostOps>(rd: &Reader<'_, H>, iomd: u64) -> Option<IomdFields> {
    let mut b = [0u8; IOMD_WINDOW as usize];
    rd.bytes(iomd, &mut b).then(|| IomdFields::decode(&b))
}

/// The length word of a descriptor's mixed-data object, for the log.
fn mixed_data_len<H: HostMemory + HostOps>(rd: &Reader<'_, H>, iomd: u64) -> String {
    rd.u64(iomd + IOMD_MEMORY_ENTRIES)
        .filter(|&p| guest_kernel_va(p))
        .and_then(|p| rd.u64(p + MIXED_DATA_LENGTH))
        .map_or_else(|| "?".to_string(), |v| format!("{v:#x}"))
}

/// A kernel pointer with its pointer authentication code taken off.
fn strip(pointer: u64, va_bits: u32) -> u64 {
    pointer | 1u64.checked_shl(va_bits).map_or(0, |bit| !(bit - 1))
}

// MARK: - Kernel

enum KernelVerdict {
    Verified(KernelImage, String, u64),
    /// Not this kernel, or not a kernel at all, with what was seen.
    Wrong(&'static str, String),
    /// Reads failed; try again later.
    Unreadable,
}

fn find_section(
    cmds: &[u8],
    ncmds: u32,
    want_segment: &[u8],
    want_section: &[u8],
) -> Option<(u64, u64)> {
    let mut off = 0usize;
    for _ in 0..ncmds {
        let cmd = u32::from_le_bytes(cmds.get(off..off + 4)?.try_into().ok()?);
        let size = u32::from_le_bytes(cmds.get(off + 4..off + 8)?.try_into().ok()?) as usize;
        if size < 8 || off + size > cmds.len() {
            return None;
        }
        if cmd == LC_SEGMENT_64 && size >= SEGMENT_COMMAND_64_LEN {
            let name = &cmds[off + 8..off + 24];
            let nsects = u32::from_le_bytes(cmds[off + 64..off + 68].try_into().ok()?) as usize;
            if name_is(name, want_segment) {
                for s in 0..nsects {
                    let at = off + SEGMENT_COMMAND_64_LEN + s * SECTION_64_LEN;
                    let sect = cmds.get(at..at + SECTION_64_LEN)?;
                    if name_is(&sect[0..16], want_section) {
                        let addr = u64::from_le_bytes(sect[32..40].try_into().ok()?);
                        let len = u64::from_le_bytes(sect[40..48].try_into().ok()?);
                        return Some((addr, len));
                    }
                }
            }
        }
        off += size;
    }
    None
}

/// Every `LC_SEGMENT_64` as `(name, vmaddr, vmsize)`.
fn segments(cmds: &[u8], ncmds: u32) -> Vec<(Vec<u8>, u64, u64)> {
    let mut out = Vec::new();
    let mut off = 0usize;
    for _ in 0..ncmds {
        let (Some(cmd), Some(size)) = (cmds.get(off..off + 4), cmds.get(off + 4..off + 8)) else {
            break;
        };
        let cmd = u32::from_le_bytes(cmd.try_into().unwrap());
        let size = u32::from_le_bytes(size.try_into().unwrap()) as usize;
        if size < 8 || off + size > cmds.len() {
            break;
        }
        if cmd == LC_SEGMENT_64 && size >= SEGMENT_COMMAND_64_LEN {
            let name = cmds[off + 8..off + 24]
                .iter()
                .copied()
                .take_while(|&c| c != 0)
                .collect();
            let addr = u64::from_le_bytes(cmds[off + 24..off + 32].try_into().unwrap());
            let len = u64::from_le_bytes(cmds[off + 32..off + 40].try_into().unwrap());
            out.push((name, addr, len));
        }
        off += size;
    }
    out
}

fn name_is(field: &[u8], want: &[u8]) -> bool {
    field.len() >= want.len()
        && &field[..want.len()] == want
        && field[want.len()..].iter().all(|&c| c == 0)
}

fn check_kernel<H: HostMemory + HostOps>(
    rd: &Reader<'_, H>,
    root: &KernelRoot,
    page_size: u64,
) -> KernelVerdict {
    let Some(vbar) = root.vbar else {
        return KernelVerdict::Wrong("no_vbar", String::new());
    };
    let va_bits = 64 - ((root.tcr >> 16) & 0x3f) as u32;
    let top = vbar & !(page_size - 1);
    if rd.u32(top).is_none() {
        return KernelVerdict::Unreadable;
    }
    let mut header = None;
    let mut scanned = 0u64;
    let mut at = top;
    while top - at < KERNEL_SCAN_BYTES {
        scanned += 1;
        let mut b = [0u8; 16];
        if rd.bytes(at, &mut b) {
            let word = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
            if word(0) == MH_MAGIC_64 && word(1) == CPU_TYPE_ARM64 && word(3) == MH_EXECUTE {
                header = Some(at);
                break;
            }
        }
        let Some(next) = at.checked_sub(page_size).filter(|&a| guest_kernel_va(a)) else {
            break;
        };
        at = next;
    }
    let vbar_text = format!("vbar={vbar:#x} scanned_pages={scanned}");
    let Some(header) = header else {
        return KernelVerdict::Wrong("kernel_header_not_found", vbar_text);
    };
    let mut head = [0u8; MACH_HEADER_64_LEN];
    if !rd.bytes(header, &mut head) {
        return KernelVerdict::Unreadable;
    }
    let ncmds = u32::from_le_bytes(head[16..20].try_into().unwrap());
    let sizeofcmds = u32::from_le_bytes(head[20..24].try_into().unwrap()) as usize;
    let at_header = format!("header={header:#x} {vbar_text}");
    if sizeofcmds > LOAD_COMMANDS_MAX {
        return KernelVerdict::Wrong("load_commands_oversized", at_header);
    }
    let mut cmds = vec![0u8; sizeofcmds];
    if !rd.bytes(header + MACH_HEADER_64_LEN as u64, &mut cmds) {
        return KernelVerdict::Unreadable;
    }
    let segs = segments(&cmds, ncmds);
    let seg = |want: &[u8]| {
        segs.iter()
            .find(|(n, _, _)| n == want)
            .map(|&(_, a, l)| (a, l))
    };
    // The booter rebases the load commands in place, so `__TEXT` names the
    // header's own address. A header that does not is not the one running.
    if seg(b"__TEXT").map(|(a, _)| a) != Some(header) {
        return KernelVerdict::Wrong("text_not_at_header", at_header);
    }
    if !segs.iter().any(|&(_, a, l)| vbar >= a && vbar - a < l) {
        return KernelVerdict::Wrong("vbar_outside_kernel", at_header);
    }
    let Some((dc, dc_len)) = seg(b"__DATA_CONST") else {
        return KernelVerdict::Wrong("no_data_const", at_header);
    };
    let Some((sect, sect_len)) = find_section(&cmds, ncmds, b"__TEXT", b"__const") else {
        return KernelVerdict::Wrong("no_text_const", at_header);
    };
    if sect_len > TEXT_CONST_MAX {
        return KernelVerdict::Wrong("text_const_oversized", at_header);
    }
    let mut text = vec![0u8; sect_len as usize];
    if !rd.bytes(sect, &mut text) {
        return KernelVerdict::Unreadable;
    }
    let Some(start) = text
        .windows(VERSION_PREFIX.len())
        .position(|w| w == VERSION_PREFIX)
    else {
        return KernelVerdict::Wrong("no_version_string", at_header);
    };
    let version: String = text[start..]
        .iter()
        .take(VERSION_MAX)
        .take_while(|&&c| c != 0)
        .map(|&c| {
            if c.is_ascii_graphic() || c == b' ' {
                c as char
            } else {
                '?'
            }
        })
        .collect();
    if version != EXPECTED_KERNEL_VERSION {
        return KernelVerdict::Wrong(
            "kernel_version_mismatch",
            format!("version=\"{version}\" {at_header}"),
        );
    }
    let image = KernelImage {
        header,
        data_const: (dc, dc.saturating_add(dc_len)),
        va_bits,
    };
    KernelVerdict::Verified(image, version, scanned)
}

/// The kernel check, once a boot. `None` while it cannot be made yet.
fn kernel_image<H: HostMemory + HostOps>(
    ww: &mut WireWatch,
    state: &DeviceState,
    rd: &Reader<'_, H>,
) -> Option<KernelImage> {
    match ww.kernel {
        KernelCheck::Verified(image) => return Some(image),
        KernelCheck::Unverified => return None,
        KernelCheck::Unchecked => {}
    }
    let root = state.kernel_walk.root?;
    let started = std::time::Instant::now();
    match check_kernel(rd, &root, state.page_size()) {
        KernelVerdict::Verified(image, version, scanned) => {
            crate::observe::off(format!(
                "wire_watch_kernel verified version=\"{version}\" header={:#x} vbar={:#x} \
                 data_const={:#x}..{:#x} va_bits={} scanned_pages={scanned} us={}",
                image.header,
                root.vbar.unwrap_or(0),
                image.data_const.0,
                image.data_const.1,
                image.va_bits,
                started.elapsed().as_micros()
            ));
            ww.kernel = KernelCheck::Verified(image);
            Some(image)
        }
        KernelVerdict::Wrong(reason, seen) => {
            crate::observe::fail(format!(
                "wire_watch_layout_unverified scope=kernel reason={reason} {seen} expected=\"{}\" \
                 (the watch is off for this boot)",
                EXPECTED_KERNEL_VERSION
            ));
            ww.kernel = KernelCheck::Unverified;
            None
        }
        KernelVerdict::Unreadable => None,
    }
}

// MARK: - Mapping layout

/// A `MappingInternal` and its surface and page-table descriptors, checked.
type Checked = (MiFields, IomdFields, IomdFields);

/// The checks of the module doc, on one mapping. `Err` names the first that
/// failed; `Ok(None)` is a read that could not be made yet.
fn check_mapping<H: HostMemory + HostOps>(
    rd: &Reader<'_, H>,
    state: &DeviceState,
    image: &KernelImage,
    mapping_id: u32,
    internal: u64,
    at_map: bool,
) -> Result<Option<Checked>, (&'static str, String)> {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Ok(None);
    };
    if m.page_entries.is_empty() || m.page_table_kva == 0 {
        return Ok(None);
    }
    let Some(mi) = read_mi(rd, internal) else {
        return Ok(None);
    };
    let pages = m.page_entries.len() as u64;
    let seen = |what: String| format!("internal={internal:#x} {what}");
    if mi.id != mapping_id {
        return Err(("mi_id", seen(format!("id={}", mi.id))));
    }
    if mi.size != MAPPING_INTERNAL_EXPECTED_SIZE {
        return Err(("mi_size", seen(format!("size={:#x}", mi.size))));
    }
    if mi.field_48 != 0 || mi.field_50 != 0 {
        return Err((
            "mi_unshifted_layout",
            seen(format!("f48={:#x} f50={:#x}", mi.field_48, mi.field_50)),
        ));
    }
    if mi.pages != pages {
        return Err((
            "mi_page_count",
            seen(format!("mi_pages={} device_pages={pages}", mi.pages)),
        ));
    }
    if !guest_kernel_va(mi.surface) || !guest_kernel_va(mi.table) || mi.surface == mi.table {
        return Err((
            "iomd_pointer",
            seen(format!("surface={:#x} table={:#x}", mi.surface, mi.table)),
        ));
    }
    let (Some(surface), Some(table)) = (read_iomd(rd, mi.surface), read_iomd(rd, mi.table)) else {
        return Ok(None);
    };
    let iomds = |what: String| {
        seen(format!(
            "surface_iomd={:#x} table_iomd={:#x} {what}",
            mi.surface, mi.table
        ))
    };
    if table.buffer != m.page_table_kva {
        return Err((
            "table_buffer",
            iomds(format!(
                "buffer={:#x} page_table={:#x}",
                table.buffer, m.page_table_kva
            )),
        ));
    }
    let page_size = state.page_size();
    if surface.length != pages.saturating_mul(page_size) || u64::from(surface.pages) != pages {
        return Err((
            "surface_length",
            iomds(format!(
                "length={:#x} iomd_pages={} mapping_size={:#x}",
                surface.length,
                surface.pages,
                pages.saturating_mul(page_size)
            )),
        ));
    }
    let table_len = (pages * TABLE_ENTRY_LEN).div_ceil(page_size) * page_size;
    if table.length != table_len || u64::from(table.pages) != table_len / page_size {
        return Err((
            "table_length",
            iomds(format!(
                "length={:#x} iomd_pages={} expected={table_len:#x}",
                table.length, table.pages
            )),
        ));
    }
    if surface.initialized != 1 || table.initialized != 1 {
        return Err(("iomd_initialized", iomds(String::new())));
    }
    let (sv, tv) = (
        strip(surface.vtable, image.va_bits),
        strip(table.vtable, image.va_bits),
    );
    if sv != tv || sv < image.data_const.0 || sv >= image.data_const.1 {
        return Err((
            "iomd_class",
            iomds(format!(
                "surface_vtable={sv:#x} table_vtable={tv:#x} data_const={:#x}..{:#x}",
                image.data_const.0, image.data_const.1
            )),
        ));
    }
    // At `MAP` both were prepared by `wire_mapping` before the request went out.
    if at_map && ((surface.wire as i32) <= 0 || (table.wire as i32) <= 0) {
        return Err((
            "map_wire_not_positive",
            iomds(format!(
                "surface_wire={} table_wire={}",
                surface.wire as i32, table.wire as i32
            )),
        ));
    }
    Ok(Some((mi, surface, table)))
}

/// Check a pending mapping's layout, and start watching it when it passes.
fn activate<H: HostMemory + HostOps>(
    ww: &mut WireWatch,
    rd: &Reader<'_, H>,
    state: &DeviceState,
    mapping_id: u32,
    at_map: bool,
) {
    let Some(image) = kernel_image(ww, state, rd) else {
        if ww.off() {
            ww.watched.clear();
        }
        return;
    };
    let Some(w) = ww.watched.get(&mapping_id) else {
        return;
    };
    let internal = w.internal;
    match check_mapping(rd, state, &image, mapping_id, internal, at_map) {
        Ok(None) => {}
        Err((reason, seen)) => {
            ww.watched.remove(&mapping_id);
            if ww.unverified.insert((mapping_id, internal)) {
                crate::observe::fail(format!(
                    "wire_watch_layout_unverified scope=mapping mapping={mapping_id} \
                     reason={reason} {seen} (this mapping is not watched)"
                ));
            }
        }
        Ok(Some((mi, surface, table))) => {
            let counts = Counts {
                surface: surface.wire,
                table: table.wire,
                mi_state: mi.state,
            };
            let gen = state
                .mappings
                .get(&mapping_id)
                .map_or(0, |m| m.map_generation);
            let event = ww.push(Ev::Map {
                mapping: mapping_id,
                surface: surface.wire,
                table: table.wire,
            });
            crate::observe::off(format!(
                "wire_watch map mapping={mapping_id} internal={internal:#x} surface_iomd={:#x} \
                 table_iomd={:#x} surface_wire={} table_wire={} mi_state={:#x} pages={} \
                 length={:#x} md_len={} gen={gen} at={} seq={} t_us={}",
                mi.surface,
                mi.table,
                surface.wire as i32,
                table.wire as i32,
                mi.state,
                mi.pages,
                surface.length,
                mixed_data_len(rd, mi.surface),
                if at_map { "map" } else { "late" },
                event.seq,
                event.t_us
            ));
            let w = ww.watched.get_mut(&mapping_id).expect("looked up above");
            w.phase = Phase::Active;
            w.surface_iomd = mi.surface;
            w.table_iomd = mi.table;
            w.pages = mi.pages;
            w.surface_fields = surface;
            w.table_fields = table;
            w.at_map = at_map.then_some(counts);
            w.last = counts;
            w.sampled_us = Some(now_us());
            // Checked after `MAP`, a count can already be wrong.
            if !at_map {
                for field in [Field::Surface, Field::Table] {
                    let value = match field {
                        Field::Surface => counts.surface,
                        _ => counts.table,
                    };
                    if (value as i32) <= 0 {
                        first_anomaly(ww, rd, mapping_id, field, value, "late_check");
                    }
                }
            }
        }
    }
}

// MARK: - Sampling

/// Read one watched mapping's counts and report what moved. `false` when the
/// counts could not be read now.
fn sample_one<H: HostMemory + HostOps>(
    ww: &mut WireWatch,
    rd: &Reader<'_, H>,
    state: &DeviceState,
    mapping_id: u32,
    at: &str,
) -> bool {
    let Some(w) = ww.watched.get(&mapping_id) else {
        return false;
    };
    let retire = |ww: &mut WireWatch, reason: &str| {
        ww.watched.remove(&mapping_id);
        crate::observe::off(format!(
            "wire_watch retired mapping={mapping_id} reason={reason} at={at}"
        ));
    };
    match state.mappings.get(&mapping_id) {
        None => {
            retire(ww, "mapping_gone");
            return false;
        }
        Some(m) if m.mapping_internal != w.internal => {
            retire(ww, "internal_reattached");
            return false;
        }
        Some(_) => {}
    }
    if w.phase == Phase::Pending {
        activate(ww, rd, state, mapping_id, false);
        return false;
    }
    let (internal, surface_iomd, table_iomd) = (w.internal, w.surface_iomd, w.table_iomd);
    ww.cost.samples += 1;
    let Some(mi) = read_mi(rd, internal) else {
        return false;
    };
    if mi.id != mapping_id || mi.surface != surface_iomd || mi.table != table_iomd {
        retire(
            ww,
            &format!(
                "internal_changed id={} surface={:#x} table={:#x}",
                mi.id, mi.surface, mi.table
            ),
        );
        return false;
    }
    let (Some(surface), Some(table)) = (
        rd.u32(surface_iomd + IOMD_WIRE_COUNT),
        rd.u32(table_iomd + IOMD_WIRE_COUNT),
    ) else {
        return false;
    };
    let now = Counts {
        surface,
        table,
        mi_state: mi.state,
    };
    let w = ww.watched.get_mut(&mapping_id).expect("checked above");
    w.sampled_us = Some(now_us());
    let (last, at_map) = (w.last, w.at_map);
    if now == last {
        return true;
    }
    for (field, old, new, iomd) in [
        (Field::Surface, last.surface, now.surface, surface_iomd),
        (Field::Table, last.table, now.table, table_iomd),
        (Field::MiState, last.mi_state, now.mi_state, internal),
    ] {
        if old == new {
            continue;
        }
        ww.cost.transitions += 1;
        let map = at_map.map_or_else(
            || "?".to_string(),
            |c| match field {
                Field::Surface => (c.surface as i32).to_string(),
                Field::Table => (c.table as i32).to_string(),
                Field::MiState => format!("{:#x}", c.mi_state),
            },
        );
        let (old_s, new_s) = match field {
            Field::MiState => (format!("{old:#x}"), format!("{new:#x}")),
            _ => ((old as i32).to_string(), (new as i32).to_string()),
        };
        crate::observe::off(format!(
            "wire_watch transition mapping={mapping_id} iomd={iomd:#x} field={} old={old_s} \
             new={new_s} map={map} t_us={} at={at} {}",
            field.name(),
            now_us(),
            ww.context()
        ));
        let below_map = match (field, at_map) {
            (Field::Surface, Some(c)) => (new as i32) < (c.surface as i32),
            (Field::Table, Some(c)) => (new as i32) < (c.table as i32),
            _ => false,
        };
        if field != Field::MiState && ((new as i32) <= 0 || below_map) {
            first_anomaly(ww, rd, mapping_id, field, new, at);
        }
        // A condemned incarnation stays condemned: the guest completed a
        // descriptor it still had mapped, and a later prepare does not say the
        // frames this device holds for it are the guest's again.
        let w = ww.watched.get_mut(&mapping_id).expect("checked above");
        if field != Field::MiState
            && (new as i32) > 0
            && w.condemned.is_some()
            && !std::mem::replace(&mut w.recovered_said, true)
        {
            crate::observe::fail(format!(
                "wire_guard_still_condemned mapping={mapping_id} field={} value={} t_us={} \
                 (the count is positive again; the incarnation stays condemned)",
                field.name(),
                new as i32,
                now_us()
            ));
        }
    }
    if let Some(w) = ww.watched.get_mut(&mapping_id) {
        w.last = now;
    }
    true
}

/// The first time one count of one mapping reads zero, below zero, or below
/// its `MAP`-time value: what else is true of the descriptor, and what this
/// device did just before.
fn first_anomaly<H: HostMemory + HostOps>(
    ww: &mut WireWatch,
    rd: &Reader<'_, H>,
    mapping_id: u32,
    field: Field,
    value: u32,
    at: &str,
) {
    let Some(w) = ww.watched.get_mut(&mapping_id) else {
        return;
    };
    if !w.anomalous.insert(field.name()) {
        return;
    }
    ww.cost.anomalies += 1;
    let (iomd, recorded) = match field {
        Field::Surface => (w.surface_iomd, w.surface_fields),
        _ => (w.table_iomd, w.table_fields),
    };
    let (internal, writes, at_map) = (w.internal, w.writes, w.at_map);
    let own_write = w
        .last_write
        .as_ref()
        .map_or_else(|| "-".to_string(), WriteNote::to_string);
    let boot_first = !std::mem::replace(&mut ww.first_anomaly_seen, true);
    // Whether anything but the wiring moved: a corrupted descriptor would show
    // it, a `complete()` would not.
    let fields = match read_iomd(rd, iomd) {
        Some(now) => {
            let moved = recorded.differs_besides_wiring(&now);
            if moved.is_empty() {
                "unchanged".to_string()
            } else {
                format!("changed:{}", moved.join(","))
            }
        }
        None => "unreadable".to_string(),
    };
    let md_len = mixed_data_len(rd, iomd);
    let condemned = guard_enabled() && {
        let w = ww.watched.get_mut(&mapping_id).expect("looked up above");
        let fresh = w.condemned.is_none();
        if fresh {
            w.condemned = Some(Condemned {
                seq: ww.seq,
                t_us: now_us(),
                field: field.name(),
                horizon: Vec::new(),
            });
            ww.condemn_pending.push(mapping_id);
        }
        fresh
    };
    let map = at_map.map_or_else(
        || "?".to_string(),
        |c| match field {
            Field::Surface => (c.surface as i32).to_string(),
            _ => (c.table as i32).to_string(),
        },
    );
    crate::observe::fail(format!(
        "wire_watch first_anomaly mapping={mapping_id} field={} value={} map={map} iomd={iomd:#x} \
         internal={internal:#x} boot_first={} iomd_fields={fields} md_len={md_len} \
         device_wrote_iomd_frame={} device_writes_to_mapping={writes} own_last_write={own_write} \
         t_us={} at={at} guard={} {} ring={}",
        field.name(),
        value as i32,
        boot_first as u8,
        rd.device_wrote(iomd),
        now_us(),
        if condemned { "condemned" } else { "off" },
        ww.context(),
        ww.ring.len()
    ));
    for (i, event) in ww.ring.iter().enumerate() {
        crate::observe::fail(format!(
            "wire_watch first_anomaly_ring mapping={mapping_id} field={} i={i} {event}",
            field.name()
        ));
    }
}

impl<H: HostMemory + HostOps> Reader<'_, H> {
    /// Whether this device ever wrote the frame a kernel object lives in:
    /// `0` exonerates its write rails, `1` does not convict them.
    fn device_wrote(&self, kva: u64) -> &'static str {
        let kernel_walk::Serve::Walk(root) = self.mem.walk.serve else {
            return "unknown";
        };
        match kernel_walk::translate(self.mem.host, &root, kva) {
            Ok(gpa) => match crate::observe::footprint::wrote_gpa(gpa) {
                true => "1",
                false => "0",
            },
            Err(_) => "unknown",
        }
    }
}

fn sample_where<H: HostMemory + HostOps>(
    ww: &mut WireWatch,
    rd: &Reader<'_, H>,
    state: &DeviceState,
    at: &str,
    which: impl Fn(&Watched) -> bool,
) {
    let ids: Vec<u32> = ww
        .watched
        .iter()
        .filter(|(_, w)| which(w))
        .map(|(&id, _)| id)
        .collect();
    for id in ids {
        sample_one(ww, rd, state, id, at);
    }
}

/// Run `f` over the watch with a reader, putting the watch back afterwards.
fn with_watch<H: HostMemory + HostOps, R>(
    state: &mut DeviceState,
    host: &H,
    f: impl FnOnce(&mut WireWatch, &Reader<'_, H>, &DeviceState) -> R,
) -> R {
    let started = std::time::Instant::now();
    let mut ww = std::mem::take(&mut state.wire_watch);
    let rd = Reader::new(state, host);
    let out = f(&mut ww, &rd, state);
    rd.fold(&mut ww.cost, started);
    state.wire_watch = ww;
    if !state.wire_watch.condemn_pending.is_empty() {
        apply_condemnations(state, host);
    }
    out
}

/// What a condemnation does that needs the whole device: where every child
/// FIFO stands, so the display presents already published can be told from
/// later ones, and the frame owed to the mapping, which is dropped rather than
/// landed in pages the guest has let go of. The resident stays: a later present
/// of the surface shows it without writing guest memory.
fn apply_condemnations<H: HostMemory + HostOps>(state: &mut DeviceState, host: &H) {
    for mapping_id in std::mem::take(&mut state.wire_watch.condemn_pending) {
        let positions = crate::runtime::drain::child_positions(
            state,
            host,
            state.active_child_mask | state.pending.child_mask,
        );
        let debt = state.pending_writebacks.take(mapping_id).is_some();
        let ww = &mut state.wire_watch;
        let horizon: Vec<Horizon> = positions
            .iter()
            .map(|p| Horizon {
                channel: p.channel,
                generation: ww.fifo_generation(p.channel),
                base_pfn: p.base_pfn,
                head: p.head,
                tail: p.tail,
            })
            .collect();
        ww.cost.condemned += 1;
        if debt {
            ww.cost.debts_dropped += 1;
            crate::runtime::drain::note_store_route("wbdebt_dropped_condemned");
        }
        let Some(c) = ww
            .watched
            .get_mut(&mapping_id)
            .and_then(|w| w.condemned.as_mut())
        else {
            continue;
        };
        c.horizon = horizon;
        let horizon: Vec<String> = c
            .horizon
            .iter()
            .map(|h| {
                format!(
                    "{}:gen{}:pfn{:#x}:{:#x}..{:#x}",
                    h.channel, h.generation, h.base_pfn, h.head, h.tail
                )
            })
            .collect();
        crate::observe::fail(format!(
            "wire_guard_condemned mapping={mapping_id} field={} seq={} t_us={} horizon=[{}] \
             debt={} (the guest unwired a surface it still has mapped: writes into this \
             incarnation are refused, its owed frame is dropped, and display presents naming \
             it that were already published are refused rather than shown)",
            c.field,
            c.seq,
            c.t_us,
            horizon.join(","),
            if debt { "dropped" } else { "none" }
        ));
    }
}

/// A child FIFO was defined or freed: its positions start over, so no horizon
/// taken on its previous incarnation may judge a packet on it again.
pub fn on_fifo_lifecycle(state: &mut DeviceState, channel: u32, what: &'static str) {
    if !enabled() {
        return;
    }
    let ww = &mut state.wire_watch;
    *ww.fifo_generations.entry(channel).or_default() += 1;
    for (mapping_id, w) in ww.watched.iter_mut() {
        let Some(c) = w.condemned.as_mut() else {
            continue;
        };
        let before = c.horizon.len();
        c.horizon.retain(|h| h.channel != channel);
        if c.horizon.len() != before {
            crate::observe::off(format!(
                "wire_guard_horizon_retired mapping={mapping_id} ch={channel} reason={what}"
            ));
        }
    }
}

// MARK: - Hooks: mapper ring

/// After a `MAP` was applied and resolved, on the publishing vCPU.
pub fn on_map<H: HostMemory + HostOps>(state: &mut DeviceState, host: &H, mapping_id: u32) {
    if !enabled() || state.wire_watch.off() {
        return;
    }
    let Some(internal) = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.mapping_internal)
        .filter(|&i| i != 0)
    else {
        return;
    };
    with_watch(state, host, |ww, rd, state| {
        if let Some(old) = ww.watched.get(&mapping_id) {
            if old.internal != internal {
                crate::observe::off(format!(
                    "wire_watch retired mapping={mapping_id} reason=remapped internal={:#x}",
                    old.internal
                ));
            }
        }
        ww.watched.insert(
            mapping_id,
            Watched {
                internal,
                phase: Phase::Pending,
                surface_iomd: 0,
                table_iomd: 0,
                pages: 0,
                surface_fields: IomdFields::default(),
                table_fields: IomdFields::default(),
                at_map: None,
                last: Counts::default(),
                display: false,
                anomalous: BTreeSet::new(),
                writes: 0,
                writes_after_anomaly: 0,
                lines_after_anomaly: 0,
                guard_lines: 0,
                refused: 0,
                last_write: None,
                sampled_us: None,
                condemned: None,
                recovered_said: false,
            },
        );
        activate(ww, rd, state, mapping_id, true);
    });
}

/// Before an `UNMAP` is applied: the last sample, then the mapping leaves the
/// watch. The guest completes both descriptors only after this request.
pub fn on_unmap<H: HostMemory + HostOps>(state: &mut DeviceState, host: &H, mapping_id: u32) {
    if !enabled() || !state.wire_watch.watched.contains_key(&mapping_id) {
        return;
    }
    with_watch(state, host, |ww, rd, state| {
        sample_one(ww, rd, state, mapping_id, "unmap");
        let event = ww.push(Ev::Unmap {
            mapping: mapping_id,
        });
        if let Some(w) = ww.watched.remove(&mapping_id) {
            crate::observe::off(format!(
                "wire_watch unmap mapping={mapping_id} surface_wire={} table_wire={} pages={} \
                 writes={} writes_after_anomaly={} anomalous={:?} seq={}",
                w.last.surface as i32,
                w.last.table as i32,
                w.pages,
                w.writes,
                w.writes_after_anomaly,
                w.anomalous,
                event.seq
            ));
        }
    });
}

// MARK: - Hooks: writes

/// Which rail is about to put bytes into a mapping's guest pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteKind {
    /// The host CPU copies into the pages.
    Cpu,
    /// The GPU copies into the pages. Refused only where the guard has
    /// condemned the incarnation, like a CPU write.
    Gpu,
}

/// Right before a write into `mapping_id`'s guest pages. `false` only when the
/// guard is on and has condemned this incarnation, having said so. This read
/// can itself be what condemns it.
pub fn before_write<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
    site: &str,
    kind: WriteKind,
) -> bool {
    if !enabled() {
        return true;
    }
    let t_us = now_us();
    let note = WriteNote {
        mapping: mapping_id,
        site: site.to_string(),
        t_us,
    };
    state.wire_watch.last_write = Some(note.clone());
    if !state.wire_watch.watched.contains_key(&mapping_id) {
        return true;
    }
    with_watch(state, host, |ww, rd, state| {
        // A row-at-a-time writer calls this per row; one read serves a burst.
        let recent = ww
            .watched
            .get(&mapping_id)
            .and_then(|w| w.sampled_us)
            .is_some_and(|s| t_us.saturating_sub(s) < WRITE_RESAMPLE_US);
        let fresh = recent || sample_one(ww, rd, state, mapping_id, site);
        let ctx = ww.context();
        let Some(w) = ww.watched.get_mut(&mapping_id) else {
            return true;
        };
        w.writes += 1;
        w.last_write = Some(note);
        let surface = w.last.surface as i32;
        if !w.anomalous.is_empty() {
            w.writes_after_anomaly += 1;
            if w.writes_after_anomaly.is_power_of_two() && w.lines_after_anomaly < LINES_PER_MAPPING
            {
                w.lines_after_anomaly += 1;
                crate::observe::fail(format!(
                    "wire_watch write_after_anomaly mapping={mapping_id} site={site} kind={kind:?} \
                     n={} surface_wire={surface} table_wire={} fresh={} t_us={t_us} {ctx}",
                    w.writes_after_anomaly,
                    w.last.table as i32,
                    fresh as u8
                ));
            }
        }
        // Latched, not read off the count: a positive count after the guest
        // unwired a mapped surface does not hand its frames back.
        let refuse = guard_enabled() && w.condemned.is_some();
        if refuse {
            ww.cost.guard_refused += 1;
            let w = ww.watched.get_mut(&mapping_id).expect("checked above");
            w.refused += 1;
            if w.refused.is_power_of_two() && w.guard_lines < LINES_PER_MAPPING {
                w.guard_lines += 1;
                crate::observe::fail(format!(
                    "wire_guard_refused mapping={mapping_id} site={site} kind={kind:?} \
                     surface_wire={surface} n={} t_us={t_us} (the guest unwired this incarnation \
                     while it was mapped; the write is refused)",
                    w.refused
                ));
            }
            crate::runtime::drain::note_store_route("wire_guard_refused");
        }
        !refuse
    })
}

// MARK: - Hooks: display

/// What became of a display present packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapHow {
    /// It ran.
    Exec,
    /// It stays at its head and runs again later.
    Deferred,
    /// The guard refused to run it; see [`refuse_swap`].
    Refused,
}

impl SwapHow {
    fn name(self) -> &'static str {
        match self {
            Self::Exec => "exec",
            Self::Deferred => "deferred",
            Self::Refused => "refused",
        }
    }
}

/// Whether the guard refuses to run this display present.
///
/// It does for a present that names a condemned incarnation and was already
/// published, in this incarnation of its FIFO, when the unwire was seen. That
/// is a refusal, not a claim about the guest: the horizon is taken when the
/// watch sees the drop, up to [`DISPLAY_SAMPLE_US`] after the guest made it, so
/// it proves only "published before we looked", not that the guest cancelled
/// this particular request. What it does establish is that the present names
/// a backing the guest has let go of, which is reason enough not to show it.
///
/// What refusing does and does not tell the guest. The drain still consumes
/// the packet and writes its completion stamp: the stamp is the FIFO's command
/// fence, and other packets wait on its slot, so withholding it would wedge the
/// FIFO. What it skips is everything that says a frame was presented: the
/// retain of the surface as the displayed one and the display pipe's present
/// event, which is the guest's frame-done edge. The guest therefore sees the
/// command complete and no present happen for it.
///
/// A present published after the horizon, or on another incarnation of the
/// FIFO, runs: it shows the resident and writes no guest memory.
pub fn refuse_swap(
    state: &mut DeviceState,
    channel: u32,
    base_pfn: u32,
    head: u32,
    mapping: Option<u32>,
) -> bool {
    if !guard_enabled() {
        return false;
    }
    let Some(mapping) = mapping else {
        return false;
    };
    let internal = state.mappings.get(&mapping).map(|m| m.mapping_internal);
    let ww = &mut state.wire_watch;
    let generation = ww.fifo_generation(channel);
    let Some(w) = ww.watched.get_mut(&mapping) else {
        return false;
    };
    if internal != Some(w.internal) {
        return false;
    }
    let Some(c) = w.condemned.as_mut() else {
        return false;
    };
    let Some(at) = c.horizon.iter().position(|h| h.channel == channel) else {
        return false;
    };
    let h = c.horizon[at];
    if h.generation != generation || h.base_pfn != base_pfn {
        c.horizon.remove(at);
        crate::observe::off(format!(
            "wire_guard_horizon_retired mapping={mapping} ch={channel} reason=ring_changed \
             gen={}->{generation} pfn={:#x}->{base_pfn:#x}",
            h.generation, h.base_pfn
        ));
        return false;
    }
    if !h.holds(head) {
        return false;
    }
    let seq = c.seq;
    ww.cost.swaps_refused += 1;
    crate::observe::fail(format!(
        "wire_guard_swap_refused mapping={mapping} ch={channel} head={head:#x} \
         horizon={:#x}..{:#x} gen={generation} condemned_seq={seq} t_us={} (published before \
         the unwire was seen; not shown and no present event, its stamp is written as the \
         command fence)",
        h.head,
        h.tail,
        now_us()
    ));
    true
}

/// What a display present packet named, as the drain decoded it.
pub struct SwapPacket<'a> {
    pub channel: u32,
    pub head: u32,
    /// The stamp slot this channel's completions are written to.
    pub slot: u32,
    pub opcode: u16,
    pub stamp: u32,
    pub waits: &'a [(u32, u32)],
    pub payload: &'a [u8],
    pub mapping: Option<u32>,
}

/// Before a display present packet runs: a sample of the swap buffers.
pub fn before_swap<H: HostMemory + HostOps>(state: &mut DeviceState, host: &H) {
    if !enabled() || state.wire_watch.off() {
        return;
    }
    state.wire_watch.present_in_swap = None;
    with_watch(state, host, |ww, rd, state| {
        sample_where(ww, rd, state, "swap_begin", |w| w.display);
    });
}

/// After a display present packet ran, will run again, or was completed without
/// running.
pub fn after_swap<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    packet: SwapPacket<'_>,
    started_us: u64,
    how: SwapHow,
) {
    if !enabled() || state.wire_watch.off() {
        return;
    }
    let mapping = packet.mapping.unwrap_or(0);
    let present = state.wire_watch.present_in_swap.take().unwrap_or("-");
    with_watch(state, host, |ww, rd, state| {
        ww.held_seen
            .retain(|&(ch, head, _)| (ch, head) != (packet.channel, packet.head));
        let event = ww.push(Ev::Swap {
            channel: packet.channel,
            head: packet.head,
            opcode: packet.opcode,
            stamp: packet.stamp,
            slot: packet.slot,
            mapping,
            how,
            present,
        });
        let words: Vec<String> = packet
            .payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| format!("{:x}", u32::from_le_bytes(*w)))
            .collect();
        let waits: Vec<String> = packet
            .waits
            .iter()
            .map(|(i, v)| format!("{i}:{v:#x}"))
            .collect();
        crate::observe::off(format!(
            "wire_watch_ev {event} waits=[{}] words=[{}] dur_us={} hold_active={}",
            waits.join(","),
            words.join(","),
            event.t_us.saturating_sub(started_us),
            ww.hold_since_us.is_some() as u8
        ));
        ww.last_swap = Some(event.clone());
        if how != SwapHow::Deferred {
            if ww.unstamped.len() == UNSTAMPED_MAX {
                ww.unstamped.pop_front();
                ww.cost.unstamped_dropped += 1;
            }
            ww.unstamped.push_back(Unstamped {
                seq: event.seq,
                slot: packet.slot,
                stamp: packet.stamp,
                mapping,
            });
        }
        if let Some(w) = ww.watched.get_mut(&mapping) {
            w.display = true;
        }
        sample_where(ww, rd, state, "swap_end", |w| w.display);
    });
}

/// A display present packet the drain will not run yet, seen at its FIFO's head.
pub fn swap_held(state: &mut DeviceState, cause: &'static str, channel: u32, head: u32) {
    swap_held_packet(state, cause, channel, head, 0, 0, 0);
}

fn swap_held_packet(
    state: &mut DeviceState,
    cause: &'static str,
    channel: u32,
    head: u32,
    opcode: u16,
    stamp: u32,
    mapping: u32,
) {
    if !enabled() || state.wire_watch.off() {
        return;
    }
    let ww = &mut state.wire_watch;
    // A buffer waiting to be shown is sampled at the display cadence from now.
    if let Some(w) = ww.watched.get_mut(&mapping) {
        w.display = true;
    }
    if !ww.held_seen.insert((channel, head, stamp)) {
        return;
    }
    let event = ww.push(Ev::SwapHeld {
        cause,
        channel,
        head,
        opcode,
        stamp,
        mapping,
    });
    crate::observe::off(format!("wire_watch_ev {event}"));
    ww.last_swap = Some(event);
}

/// Refresh-class present signals the VBL path raised. That path can run
/// without the device state, so it is a count every event carries rather than
/// an event of its own: at the display's refresh rate it would push everything
/// else out of the history.
static REFRESH_SIGNALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The VBL path set the present bit for the transaction class.
pub fn note_refresh_signal() {
    REFRESH_SIGNALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The display pipe's present signal, raised inside a swap.
pub fn on_present(state: &mut DeviceState, kind: &'static str) {
    if !enabled() || state.wire_watch.off() {
        return;
    }
    let ww = &mut state.wire_watch;
    ww.present_in_swap = Some(kind);
    ww.push(Ev::Present { kind });
}

/// A completion stamp written (or queued to be written) to `slot`. One that
/// completes display swaps is logged, and the swap buffers are sampled right
/// after it.
pub fn on_stamp<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    slot: u32,
    value: u32,
    queued: bool,
) {
    if !enabled() || state.wire_watch.off() {
        return;
    }
    let ww = &mut state.wire_watch;
    if !ww.unstamped.iter().any(|u| u.slot == slot) {
        return;
    }
    let mut swaps = Vec::new();
    ww.unstamped.retain(|u| {
        let done = u.slot == slot && value.wrapping_sub(u.stamp) as i32 >= 0;
        if done {
            swaps.push((u.seq, u.mapping));
        }
        !done
    });
    if swaps.is_empty() {
        return;
    }
    let event = ww.push(Ev::Stamp {
        slot,
        value,
        queued,
        swaps,
    });
    crate::observe::off(format!("wire_watch_ev {event}"));
    ww.last_stamp = Some(event);
    with_watch(state, host, |ww, rd, state| {
        sample_where(ww, rd, state, "swap_stamped", |w| w.display);
    });
}

// MARK: - Hooks: holds

/// A FIFO newly held behind a translation.
pub fn on_hold(state: &mut DeviceState, held: u32, new: u32, producer: u32) {
    if !enabled() || state.wire_watch.off() {
        return;
    }
    let ww = &mut state.wire_watch;
    ww.hold_since_us.get_or_insert_with(now_us);
    let event = ww.push(Ev::Hold {
        held,
        new,
        producer,
    });
    crate::observe::off(format!("wire_watch_ev {event}"));
    // Read the held heads on the next tick rather than in a minute.
    ww.peeked_us = None;
}

/// Every translation hold released.
pub fn on_hold_release(state: &mut DeviceState, held: u32) {
    if !enabled() || state.wire_watch.off() {
        return;
    }
    let ww = &mut state.wire_watch;
    let since = ww.hold_since_us.take();
    let event = ww.push(Ev::HoldRelease {
        held,
        held_us: since.map_or(0, |s| now_us().saturating_sub(s)),
    });
    crate::observe::off(format!("wire_watch_ev {event}"));
}

// MARK: - Hooks: worker

/// The worker's periodic pass: swap buffers, every mapping, held FIFO heads,
/// and the cost line, each at its own cadence.
pub fn tick<H: HostMemory + HostOps>(state: &mut DeviceState, host: &H) {
    if !enabled() || state.wire_watch.off() {
        return;
    }
    let now = now_us();
    let ww = &state.wire_watch;
    let display = due(ww.display_sampled_us, now, DISPLAY_SAMPLE_US);
    let all = due(ww.all_sampled_us, now, ALL_SAMPLE_US);
    let backpressure = if state.present.backpressure_hold_active {
        1u32.checked_shl(state.present.backpressure_hold_channel)
            .unwrap_or(0)
    } else {
        0
    };
    // Bit 0 is the root FIFO, which carries no display present.
    let held =
        (state.translation_order_hold_mask | state.translation_deferred_mask | backpressure) & !1;
    let peek = held != 0 && due(ww.peeked_us, now, PEEK_US);
    // The first cost line one interval after the first tick, not at it.
    let cost = ww.cost_logged_us.is_some() && due(ww.cost_logged_us, now, COST_US);
    state.wire_watch.cost_logged_us.get_or_insert(now);
    if peek {
        state.wire_watch.peeked_us = Some(now);
        for channel in 1..crate::model::MAX_CHANNELS as u32 {
            if held & (1 << channel) == 0 {
                continue;
            }
            state.wire_watch.cost.peeks += 1;
            let Some(p) = crate::runtime::drain::peek_child_head(state, host, channel) else {
                continue;
            };
            if p.mapping.is_some() {
                let cause = if state.present.backpressure_hold_active
                    && state.present.backpressure_hold_channel == channel
                {
                    "backpressure"
                } else {
                    "translation_hold"
                };
                swap_held_packet(
                    state,
                    cause,
                    channel,
                    p.head,
                    p.opcode,
                    p.completion_stamp,
                    p.mapping.unwrap_or(0),
                );
            }
        }
    }
    if display || all {
        with_watch(state, host, |ww, rd, state| {
            if all {
                ww.all_sampled_us = Some(now);
                ww.display_sampled_us = Some(now);
                sample_where(ww, rd, state, "tick", |_| true);
            } else {
                ww.display_sampled_us = Some(now);
                sample_where(ww, rd, state, "tick", |w| w.display);
            }
        });
    }
    if cost {
        state.wire_watch.cost_logged_us = Some(now);
        crate::observe::off(state.wire_watch.cost_line(state.kernel_walk.boot));
    }
}

/// Whether a cadence last run at `last` is due again at `now`.
fn due(last: Option<u64>, now: u64, every: u64) -> bool {
    last.is_none_or(|last| now.saturating_sub(last) >= every)
}

/// At a device reset, before the state is replaced: what the watch cost.
pub fn on_reset(state: &DeviceState) {
    if !enabled() {
        return;
    }
    crate::observe::off(state.wire_watch.cost_line(state.kernel_walk.boot));
}

#[cfg(test)]
#[path = "wire_watch_tests.rs"]
mod tests;
