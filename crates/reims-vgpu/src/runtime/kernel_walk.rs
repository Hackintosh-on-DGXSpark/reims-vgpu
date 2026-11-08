//! Guest kernel virtual addresses translated in software, for the threads that
//! have no vCPU to ask.
//!
//! # Why this exists
//!
//! The mapper rail names a surface by its `MappingInternal`, a guest *kernel*
//! object, and re-reads it to learn whether the surface's page list is still
//! the one this device holds. `HostOps::read_kva` answers that only on a vCPU
//! thread: it translates through the live CPU state, and from any other thread
//! it answers `MemError::NoCpu`. Since the drain moved to a worker thread
//! (QEMU 0009) every Store and write-back runs there, so the re-read never
//! happens and the mapping rail writes through whatever page list the last vCPU
//! resolve adopted — stale if the guest re-wired the surface, and permanently
//! holed if it was not yet wired then.
//!
//! # What this does instead
//!
//! On the vCPU, when the guest publishes on the IOSurface mapper ring, the
//! device reads the EL1 translation state — `TTBR1_EL1`, `TCR_EL1`,
//! `SCTLR_EL1` — through the shim, which synchronizes it from KVM first
//! ([`capture`]). The worker then translates a kernel address with a stage-1
//! walk of its own over guest physical memory ([`translate`]), reading nothing
//! but that snapshot and guest RAM. It never touches a live CPU.
//!
//! The walk refuses whatever it does not implement rather than guessing: the
//! MMU off, big-endian descriptors, `TTBR1` walks disabled, the 64 KiB granule,
//! 52-bit addressing, a block where the architecture allows none.
//!
//! # Runtime validation, and what happens without it
//!
//! Every time a vCPU path reads a kernel address through the CPU, the same bytes
//! are read through this walk and compared ([`sample`]). A root serves a thread
//! without a vCPU only after it has collected several agreeing non-zero reads
//! over several pages, one of them across a page boundary.
//! Validation belongs to one root in one boot: a changed root or a reset starts
//! it again. A stable disagreement suspends the root until it validates again
//! from zero; a second in the same root's life revokes it until it changes.
//!
//! Without a validated root — a QEMU without the matching shim, a root still
//! validating, suspended or revoked — [`Mode::Strict`] (the default) refuses the
//! write and logs it with its time, and [`Mode::Compat`] writes through the
//! cached page list as before. `REIMS_VGPU_KVA_WALK` selects the mode.
//!
//! # What it does not do
//!
//! A fresh walk only ends the blindness. It does not hold the guest's pages:
//! the guest can still re-point a page between this walk and the write that
//! follows it, and a reused `MappingInternal` is still caught, or not, by the
//! generation checks that already exist.

use crate::runtime::host::{HostMemory, HostOps};

/// `HostOps::read_xreg` indices from which the vmapple shim answers EL1
/// translation state instead of an X register. A shim without that answers an
/// error, which reads as "no root".
pub const SYSREG_BASE: u32 = 0x1000;
const SYSREG_TTBR1_EL1: u32 = SYSREG_BASE;
const SYSREG_TCR_EL1: u32 = SYSREG_BASE + 1;
const SYSREG_SCTLR_EL1: u32 = SYSREG_BASE + 2;
const SYSREG_CPU_INDEX: u32 = SYSREG_BASE + 3;
const SYSREG_VBAR_EL1: u32 = SYSREG_BASE + 4;

/// One vCPU's EL1 translation state, read in one MMIO exit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelRoot {
    pub ttbr1: u64,
    pub tcr: u64,
    pub sctlr: u64,
    /// The vCPU it was read on. Carried for the log, not compared: every vCPU
    /// runs the kernel on the same `TTBR1`.
    pub cpu_index: u64,
    /// [`KernelWalkState::boot`] at the capture.
    pub boot: u64,
    /// `VBAR_EL1`, if the shim answers it: an entry into kernel text for the
    /// active probes. Not part of the translation, and not compared.
    pub vbar: Option<u64>,
}

impl KernelRoot {
    fn same_translation(&self, other: &Self) -> bool {
        self.ttbr1 == other.ttbr1 && self.tcr == other.tcr && self.sctlr == other.sctlr
    }
}

/// Why the walk would not translate an address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkRefusal {
    MmuOff,
    BigEndian,
    Ttbr1WalksDisabled,
    Granule64k,
    GranuleReserved,
    Lpa2,
    VaSize(u32),
    NotTtbr1Address,
    /// An invalid descriptor: the address is not mapped.
    Fault {
        level: u32,
    },
    /// A block descriptor at a level the granule allows none at.
    BlockAtLevel {
        level: u32,
    },
    /// A level-3 descriptor with the reserved encoding.
    Reserved {
        level: u32,
    },
    /// `TCR_EL1.IPS` holds the reserved encoding.
    IpsReserved,
    /// A table or output address wider than `TCR_EL1.IPS` allows, at this
    /// level; the root table is level `start - 1`. QEMU raises an address size
    /// fault for the same.
    AddressSize {
        level: i32,
    },
    /// Guest physical memory could not be read at this address.
    Read {
        gpa: u64,
    },
}

impl WalkRefusal {
    pub fn slug(self) -> &'static str {
        match self {
            Self::MmuOff => "kva_walk_mmu_off",
            Self::BigEndian => "kva_walk_big_endian",
            Self::Ttbr1WalksDisabled => "kva_walk_ttbr1_disabled",
            Self::Granule64k => "kva_walk_granule_64k",
            Self::GranuleReserved => "kva_walk_granule_reserved",
            Self::Lpa2 => "kva_walk_lpa2",
            Self::VaSize(_) => "kva_walk_va_size",
            Self::NotTtbr1Address => "kva_walk_not_ttbr1_address",
            Self::Fault { .. } => "kva_walk_fault",
            Self::BlockAtLevel { .. } => "kva_walk_block_at_level",
            Self::Reserved { .. } => "kva_walk_reserved",
            Self::IpsReserved => "kva_walk_ips_reserved",
            Self::AddressSize { .. } => "kva_walk_address_size",
            Self::Read { .. } => "kva_walk_read",
        }
    }
}

const SCTLR_M: u64 = 1;
const SCTLR_EE: u64 = 1 << 25;
const TCR_EPD1: u64 = 1 << 23;
const TCR_TBI1: u64 = 1 << 38;
const TCR_DS: u64 = 1 << 59;
/// Output address bits a descriptor can carry without 52-bit addressing.
const OA_BITS: u32 = 48;

/// The translation regime [`KernelRoot`] describes, decoded once per walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Regime {
    granule_shift: u32,
    /// Index bits per level below the first.
    stride: u32,
    va_bits: u32,
    start_level: u32,
    pa_bits: u32,
    top_byte_ignored: bool,
    base: u64,
}

impl Regime {
    fn of(root: &KernelRoot) -> Result<Self, WalkRefusal> {
        if root.sctlr & SCTLR_M == 0 {
            return Err(WalkRefusal::MmuOff);
        }
        if root.sctlr & SCTLR_EE != 0 {
            return Err(WalkRefusal::BigEndian);
        }
        let tcr = root.tcr;
        if tcr & TCR_EPD1 != 0 {
            return Err(WalkRefusal::Ttbr1WalksDisabled);
        }
        if tcr & TCR_DS != 0 {
            return Err(WalkRefusal::Lpa2);
        }
        let granule_shift = match (tcr >> 30) & 0b11 {
            0b01 => 14,
            0b10 => 12,
            0b11 => return Err(WalkRefusal::Granule64k),
            _ => return Err(WalkRefusal::GranuleReserved),
        };
        let t1sz = ((tcr >> 16) & 0x3f) as u32;
        let va_bits = 64 - t1sz;
        if !(25..=OA_BITS).contains(&va_bits) {
            return Err(WalkRefusal::VaSize(va_bits));
        }
        let stride = granule_shift - 3;
        let levels = (va_bits - granule_shift).div_ceil(stride);
        let start_level = 4 - levels;
        let pa_bits = match (tcr >> 32) & 0b111 {
            0 => 32,
            1 => 36,
            2 => 40,
            3 => 42,
            4 => 44,
            5 | 6 => OA_BITS,
            _ => return Err(WalkRefusal::IpsReserved),
        };
        let start_bits = va_bits - granule_shift - stride * (levels - 1);
        let table_bytes = 8u64 << start_bits;
        let base = root.ttbr1 & ((1u64 << OA_BITS) - 1) & !(table_bytes - 1);
        if base >> pa_bits != 0 {
            return Err(WalkRefusal::AddressSize {
                level: start_level as i32 - 1,
            });
        }
        Ok(Self {
            granule_shift,
            stride,
            va_bits,
            start_level,
            pa_bits,
            top_byte_ignored: tcr & TCR_TBI1 != 0,
            base,
        })
    }

    fn granule(&self) -> u64 {
        1 << self.granule_shift
    }
}

/// Translate one guest kernel address to the guest physical address it names.
pub fn translate<M: HostMemory>(mem: &M, root: &KernelRoot, va: u64) -> Result<u64, WalkRefusal> {
    let regime = Regime::of(root)?;
    translate_in(mem, &regime, va).map(|(pa, _)| pa)
}

/// The descriptor a walk ended at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leaf {
    /// A level-3 page descriptor.
    Page,
    /// A block descriptor at this level.
    Block { level: u32 },
}

fn translate_in<M: HostMemory>(mem: &M, r: &Regime, va: u64) -> Result<(u64, Leaf), WalkRefusal> {
    let checked = if r.top_byte_ignored {
        va | 0xff00_0000_0000_0000
    } else {
        va
    };
    if checked >> r.va_bits != u64::MAX >> r.va_bits {
        return Err(WalkRefusal::NotTtbr1Address);
    }
    let oa_mask = ((1u64 << OA_BITS) - 1) & !(r.granule() - 1);
    let mut table = r.base;
    for level in r.start_level..=3 {
        let shift = r.granule_shift + r.stride * (3 - level);
        let bits = if level == r.start_level {
            r.va_bits - shift
        } else {
            r.stride
        };
        let index = (va >> shift) & ((1u64 << bits) - 1);
        let at = table + index * 8;
        let mut raw = [0u8; 8];
        mem.read_gpa(at, &mut raw)
            .map_err(|_| WalkRefusal::Read { gpa: at })?;
        let desc = u64::from_le_bytes(raw);
        if desc & 1 == 0 {
            return Err(WalkRefusal::Fault { level });
        }
        let is_table_or_page = desc & 0b10 != 0;
        let span = 1u64 << shift;
        let (pa, leaf) = if level == 3 {
            if !is_table_or_page {
                return Err(WalkRefusal::Reserved { level });
            }
            ((desc & oa_mask) | (va & (span - 1)), Leaf::Page)
        } else if is_table_or_page {
            table = desc & oa_mask;
            if table >> r.pa_bits != 0 {
                return Err(WalkRefusal::AddressSize {
                    level: level as i32,
                });
            }
            continue;
        } else {
            // Blocks: level 2 for both granules, level 1 for 4 KiB. A 16 KiB
            // level-1 block exists only with 52-bit addressing, refused above.
            let allowed = level == 2 || (level == 1 && r.granule_shift == 12);
            if !allowed {
                return Err(WalkRefusal::BlockAtLevel { level });
            }
            (
                (desc & oa_mask & !(span - 1)) | (va & (span - 1)),
                Leaf::Block { level },
            )
        };
        if pa >> r.pa_bits != 0 {
            return Err(WalkRefusal::AddressSize {
                level: level as i32,
            });
        }
        return Ok((pa, leaf));
    }
    unreachable!("level 3 always returns")
}

/// What one read through the walk went through: for each granule page it
/// touched, the descriptor it ended at and the physical page it named.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Trace {
    pub leaves: Vec<(Leaf, u64)>,
}

/// Read `buf.len()` bytes at guest kernel address `va` through the walk, one
/// granule at a time.
pub fn read<M: HostMemory>(
    mem: &M,
    root: &KernelRoot,
    va: u64,
    buf: &mut [u8],
) -> Result<(), WalkRefusal> {
    read_traced(mem, root, va, buf).map(|_| ())
}

/// [`read`], saying which descriptors it went through.
pub fn read_traced<M: HostMemory>(
    mem: &M,
    root: &KernelRoot,
    va: u64,
    buf: &mut [u8],
) -> Result<Trace, WalkRefusal> {
    let regime = Regime::of(root)?;
    let granule = regime.granule();
    let mut trace = Trace::default();
    let mut done = 0usize;
    while done < buf.len() {
        let at = va.wrapping_add(done as u64);
        let (pa, leaf) = translate_in(mem, &regime, at)?;
        trace.leaves.push((leaf, pa & !(granule - 1)));
        let in_page = (granule - (at & (granule - 1))) as usize;
        let n = in_page.min(buf.len() - done);
        mem.read_gpa(pa, &mut buf[done..done + n])
            .map_err(|_| WalkRefusal::Read { gpa: pa })?;
        done += n;
    }
    Ok(trace)
}

// MARK: - Mode

/// What a thread without a vCPU does when a mapping's page list must be
/// re-read. From `REIMS_VGPU_KVA_WALK`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The default. Re-read through a validated walk, or refuse the write: the
    /// cached page list is never written through unwitnessed.
    Strict,
    /// `compat`: the cached page list is kept and written through, as before
    /// this module. For a QEMU without the matching shim, and for baselines.
    Compat,
}

pub fn mode() -> Mode {
    #[cfg(test)]
    if let Some(mode) = TEST_MODE.with(std::cell::Cell::get) {
        return mode;
    }
    static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let (_, value) = crate::env::read(crate::env::KVA_WALK);
        let mode = match value.as_deref().map(str::trim) {
            Some(v) if v.eq_ignore_ascii_case("compat") => Mode::Compat,
            _ => Mode::Strict,
        };
        crate::observe::off(format!(
            "kva_walk_mode mode={mode:?} value={}",
            value.unwrap_or_else(|| "<unset>".into())
        ));
        mode
    })
}

#[cfg(test)]
thread_local! {
    static TEST_MODE: std::cell::Cell<Option<Mode>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with [`mode`] answering `mode` on this thread, for a test of the
/// arm the environment does not select.
#[cfg(test)]
pub(crate) fn with_mode<R>(mode: Mode, f: impl FnOnce() -> R) -> R {
    TEST_MODE.with(|m| m.set(Some(mode)));
    let out = f();
    TEST_MODE.with(|m| m.set(None));
    out
}

// MARK: - Runtime validation

/// How much agreement a root must collect before a thread without a vCPU may
/// translate with it. Only samples with a non-zero byte that the walk resolved
/// through level-3 page descriptors count — the descriptors the mapper's own
/// objects live under; a block mapping exercises a different branch, and one
/// matching read of zeros proves nothing about the physical page it came from.
/// Of those: this many in all, this many from the mapper's own reads rather
/// than the kernel-text probes, over this many pages, and one sample crossing
/// between two level-3 leaves.
const AGREED_SAMPLES: u64 = 8;
const AGREED_MAPPER_SAMPLES: u64 = 4;
const AGREED_PAGES: usize = 3;
/// Distinct pages remembered for [`AGREED_PAGES`]; more add nothing.
const PAGES_KEPT: usize = 16;

/// Where a sample's kernel address came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// A read the mapper made of its own objects on a vCPU.
    Mapper,
    /// An active probe of kernel text, from `VBAR_EL1`.
    Text,
}

/// The result of reading the same kernel bytes through the CPU and the walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkCompare {
    Agreed {
        kva: u64,
        len: usize,
        source: Source,
        nonzero: bool,
        /// Every page it touched resolved through a level-3 page descriptor.
        via_pages: bool,
        /// It touched two pages.
        crossed: bool,
        /// ...whose physical pages are not adjacent.
        noncontiguous: bool,
    },
    Mismatch {
        kva: u64,
        len: usize,
        why: &'static str,
    },
    /// The CPU read different bytes before and after the walk: the guest wrote
    /// in between, and the sample says nothing either way.
    Inconclusive,
}

/// Stable disagreements that retire a root, counted over the root's whole life
/// and not reset by agreements in between. The first suspends it at once — it
/// serves nothing until it has validated again from zero, a full threshold of
/// fresh samples — and this many revoke it until the root changes.
const MISMATCHES_TO_REVOKE: u32 = 2;

/// Bytes a crossing probe reads on each side of a page boundary.
const CROSSING_HALF: u64 = 128;

/// Runtime validation of one root against the CPU.
#[derive(Clone, Debug, Default)]
struct Validation {
    /// Qualifying samples: non-zero, agreeing, through level-3 pages.
    samples: u64,
    mapper_samples: u64,
    pages: std::collections::BTreeSet<u64>,
    /// A qualifying sample crossed between two level-3 leaves.
    crossed: bool,
    crossed_noncontiguous: bool,
    /// Agreeing samples that do not qualify, kept for the log.
    via_blocks: u64,
    zeros: u64,
    inconclusive: u64,
    mismatches: u32,
}

impl Validation {
    fn revoked(&self) -> bool {
        self.mismatches >= MISMATCHES_TO_REVOKE
    }

    fn validated(&self) -> bool {
        !self.revoked()
            && self.samples >= AGREED_SAMPLES
            && self.mapper_samples >= AGREED_MAPPER_SAMPLES
            && self.pages.len() >= AGREED_PAGES
            && self.crossed
    }

    /// The state the log reports: every condition the gate asks, by name.
    fn describe(&self) -> String {
        format!(
            "l3_samples={}/{AGREED_SAMPLES} mapper_l3_samples={}/{AGREED_MAPPER_SAMPLES} \
             l3_pages={}/{AGREED_PAGES} l3_crossed={} noncontiguous={} via_blocks={} zeros={} \
             inconclusive={} mismatches={}",
            self.samples,
            self.mapper_samples,
            self.pages.len(),
            self.crossed,
            self.crossed_noncontiguous,
            self.via_blocks,
            self.zeros,
            self.inconclusive,
            self.mismatches
        )
    }

    /// Void what validated the root, keeping its disagreements.
    fn suspend(&mut self) {
        *self = Self {
            mismatches: self.mismatches,
            ..Self::default()
        };
    }
}

/// Why a thread without a vCPU has no walk to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateRefusal {
    NoRoot,
    Unvalidated,
    /// One stable disagreement: not serving until validated again from zero.
    Suspended,
    Mismatched,
}

/// Why a page list was not re-read on a thread without a vCPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refused {
    Gate(GateRefusal),
    Walk(WalkRefusal),
}

impl Refused {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Gate(GateRefusal::NoRoot) => "kva_walk_no_root",
            Self::Gate(GateRefusal::Unvalidated) => "kva_walk_unvalidated",
            Self::Gate(GateRefusal::Suspended) => "kva_walk_suspended",
            Self::Gate(GateRefusal::Mismatched) => "kva_walk_mismatched",
            Self::Walk(why) => why.slug(),
        }
    }
}

/// What a reader without a vCPU does with a kernel address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Serve {
    /// Answer `NoCpu`, as before: compat mode, or a reader that only runs on a
    /// vCPU.
    Off,
    Walk(KernelRoot),
    Refuse(GateRefusal),
}

/// One reader's view of the walk: what to validate against, and what to do
/// without a vCPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalkPolicy {
    pub root: Option<KernelRoot>,
    pub serve: Serve,
    /// Whether a boundary-crossing sample is still owed, so a reader on a vCPU
    /// knows to take one.
    pub wants_crossing: bool,
    /// Whether the root has passed validation, after which a reader only
    /// audits it: see [`Self::sample_limit`].
    pub validated: bool,
}

/// Samples one reader takes while its root is still validating. Enough for a
/// resolve to validate on its own, and few enough that a 9216-entry page
/// table is not read three times over on the vCPU.
const SAMPLES_PER_READER_VALIDATING: u32 = 16;
/// Samples one reader takes once its root has validated: an audit, not a
/// comparison of every read.
const SAMPLES_PER_READER_VALIDATED: u32 = 1;
/// While a root validates, a reader also samples the first read of each new
/// page it touches, up to this many pages, so a resolve contributes the
/// MappingInternal, its page object and its page table rather than sixteen
/// reads of one object.
pub const PAGES_PER_READER_VALIDATING: usize = 8;

impl WalkPolicy {
    /// A reader that only ever runs on a vCPU and takes no part.
    pub const OFF: Self = Self {
        root: None,
        serve: Serve::Off,
        wants_crossing: false,
        validated: false,
    };

    /// How many kernel reads one reader compares.
    pub fn sample_limit(&self) -> u32 {
        match self.validated {
            true => SAMPLES_PER_READER_VALIDATED,
            false => SAMPLES_PER_READER_VALIDATING,
        }
    }
}

/// What one reader observed, folded into [`KernelWalkState`] afterwards.
#[derive(Clone, Debug, Default)]
pub struct WalkLog {
    pub compares: Vec<WalkCompare>,
    /// Boundary-crossing probes taken, each one extra pair of CPU reads.
    pub probes: u64,
}

// MARK: - State

/// What the device knows about kernel translation this boot.
#[derive(Clone, Debug, Default)]
pub struct KernelWalkState {
    pub root: Option<KernelRoot>,
    /// Device resets seen, so a root names the boot it was read in.
    pub boot: u64,
    validation: Validation,
    /// The shim refused the translation-state read; said once a boot.
    unavailable_reported: bool,
    validated_reported: bool,
    /// Refusals logged per mapping, to bound the log.
    refusals_logged: std::collections::BTreeMap<u32, u32>,
    /// What validation cost this boot, reported at its end.
    cost: ValidationCost,
    /// Rounds of kernel-text probes taken for the current root.
    text_rounds: u32,
    /// When the validation state was last logged while still validating.
    state_logged_ms: u64,
}

/// Rounds of kernel-text probes per root: one per mapper publish while the
/// root has not validated, so one unreadable round does not end it.
const TEXT_ROUNDS: u32 = 8;
/// Kernel-text pages one round samples, from `VBAR_EL1`'s page on.
const TEXT_PAGES: u64 = 4;
/// How often the state of a validation still in progress is logged.
const STATE_LOG_INTERVAL_MS: u64 = 10_000;

/// Validation work done in one boot: what the A/B pays on the vCPU for it.
#[derive(Clone, Copy, Debug, Default)]
struct ValidationCost {
    samples: u64,
    probes: u64,
    inconclusive: u64,
    mismatches: u64,
}

/// How many refusals one mapping may log in a boot. Each carries its time,
/// which is the point; past this the count in `store_routes` carries on.
const REFUSALS_LOGGED_PER_MAPPING: u32 = 64;

impl KernelWalkState {
    /// The root a thread without a vCPU may translate with, if this root has
    /// passed runtime validation.
    pub fn gate(&self) -> Result<KernelRoot, GateRefusal> {
        let root = self.root.ok_or(GateRefusal::NoRoot)?;
        if self.validation.revoked() {
            return Err(GateRefusal::Mismatched);
        }
        if !self.validation.validated() {
            return Err(match self.validation.mismatches {
                0 => GateRefusal::Unvalidated,
                _ => GateRefusal::Suspended,
            });
        }
        Ok(root)
    }

    pub fn policy(&self, mode: Mode) -> WalkPolicy {
        let serve = match mode {
            Mode::Compat => Serve::Off,
            Mode::Strict => match self.gate() {
                Ok(root) => Serve::Walk(root),
                Err(why) => Serve::Refuse(why),
            },
        };
        WalkPolicy {
            root: self.root,
            serve,
            wants_crossing: self.root.is_some() && !self.validation.crossed,
            validated: self.gate().is_ok(),
        }
    }

    /// Forget everything at a device reset: the next boot's kernel has its own
    /// tables, and its own validation to pass. The boot's validation cost is
    /// reported first.
    pub fn reset(&mut self) {
        let c = self.cost;
        if c.samples + c.probes != 0 {
            crate::observe::off(format!(
                "kva_walk_cost boot={} samples={} probes={} inconclusive={} mismatches={} \
                 text_rounds={} gate={:?} {} (runtime validation work on the vCPU this boot, \
                 and where the current root's validation stood)",
                self.boot,
                c.samples,
                c.probes,
                c.inconclusive,
                c.mismatches,
                self.text_rounds,
                self.gate().map(|_| "validated"),
                self.validation.describe()
            ));
        }
        let boot = self.boot.wrapping_add(1);
        *self = Self {
            boot,
            ..Self::default()
        };
    }

    /// Fold what one reader on a vCPU path compared.
    pub fn fold(&mut self, log: WalkLog) {
        let samples = log.compares.len() as u64;
        if samples + log.probes != 0 {
            crate::runtime::drain::note_store_route_n("kva_walk_samples", samples);
            crate::runtime::drain::note_store_route_n("kva_walk_probes", log.probes);
            self.cost.samples += samples;
            self.cost.probes += log.probes;
        }
        for compare in log.compares {
            match compare {
                WalkCompare::Inconclusive => self.cost.inconclusive += 1,
                WalkCompare::Mismatch { .. } => self.cost.mismatches += 1,
                WalkCompare::Agreed { .. } => {}
            }
            self.note_compare(compare);
        }
        self.log_state_if_due(crate::observe::elapsed_ms() as u64);
    }

    /// While a root is still validating, say where it stands, at most once
    /// per [`STATE_LOG_INTERVAL_MS`]: which of the gate's conditions are met.
    fn log_state_if_due(&mut self, now_ms: u64) {
        let Some(root) = self.root else {
            return;
        };
        if self.gate().is_ok()
            || now_ms.saturating_sub(self.state_logged_ms) < STATE_LOG_INTERVAL_MS
        {
            return;
        }
        self.state_logged_ms = now_ms;
        crate::observe::off(format!(
            "kva_walk_validation_state ttbr1={:#x} boot={} gate={:?} text_rounds={} {}",
            root.ttbr1,
            root.boot,
            self.gate().map(|_| "validated"),
            self.text_rounds,
            self.validation.describe()
        ));
    }

    fn note_compare(&mut self, compare: WalkCompare) {
        let Some(root) = self.root else {
            return;
        };
        let v = &mut self.validation;
        match compare {
            WalkCompare::Agreed {
                kva,
                len,
                source,
                nonzero,
                via_pages,
                crossed,
                noncontiguous,
            } => {
                crate::runtime::drain::note_store_route("kva_walk_agree");
                if v.revoked() {
                    return;
                }
                if !nonzero {
                    v.zeros += 1;
                    return;
                }
                if !via_pages {
                    v.via_blocks += 1;
                    return;
                }
                v.samples += 1;
                if source == Source::Mapper {
                    v.mapper_samples += 1;
                }
                if crossed {
                    v.crossed = true;
                    v.crossed_noncontiguous |= noncontiguous;
                }
                if v.pages.len() < PAGES_KEPT {
                    let granule = Regime::of(&root).map_or(1 << 14, |r| r.granule());
                    v.pages.insert(kva & !(granule - 1));
                    let last = kva.wrapping_add(len.max(1) as u64 - 1);
                    v.pages.insert(last & !(granule - 1));
                }
                if v.validated() && !self.validated_reported {
                    self.validated_reported = true;
                    crate::observe::off(format!(
                        "kva_walk_validated ttbr1={:#x} tcr={:#x} boot={} text_rounds={} {} \
                         (runtime validation: the software walk agreed with the CPU; it now \
                         serves threads without a vCPU)",
                        root.ttbr1,
                        root.tcr,
                        root.boot,
                        self.text_rounds,
                        v.describe()
                    ));
                }
            }
            WalkCompare::Mismatch { kva, len, why } => {
                crate::runtime::drain::note_store_route("kva_walk_mismatch");
                let samples = v.samples;
                v.mismatches += 1;
                // Suspended: everything that validated it is void, and it serves
                // nothing until a full threshold of fresh samples has agreed.
                v.suspend();
                self.validated_reported = false;
                if v.mismatches <= MISMATCHES_TO_REVOKE {
                    crate::observe::fail(format!(
                        "kva_walk_mismatch kva={kva:#x} len={len} detail={why} samples={samples} \
                         mismatches={} ttbr1={:#x} state={} (runtime validation: the CPU read \
                         the same bytes before and after, and the software walk read others; \
                         the first suspends the walk until it validates again from zero, \
                         {MISMATCHES_TO_REVOKE} in this root's life revoke it until the root \
                         changes)",
                        v.mismatches,
                        root.ttbr1,
                        if v.revoked() { "revoked" } else { "suspended" },
                    ));
                }
            }
            WalkCompare::Inconclusive => {
                crate::runtime::drain::note_store_route("kva_walk_inconclusive");
                v.inconclusive += 1;
            }
        }
    }

    /// Log one refusal to re-read a page list without a vCPU.
    pub fn note_refusal(&mut self, mapping_id: u32, pages: usize, why: Refused, kva: u64) {
        crate::runtime::drain::note_store_route(why.slug());
        crate::runtime::drain::note_store_route("kva_walk_resolve_refused");
        let logged = self.refusals_logged.entry(mapping_id).or_default();
        if *logged >= REFUSALS_LOGGED_PER_MAPPING {
            return;
        }
        *logged += 1;
        crate::observe::fail(format!(
            "kva_walk_resolve_refused mapping={mapping_id} reason={} detail={why:?} kva={kva:#x} \
             pages=0..{pages} mode={:?} (the page list could not be re-read without a vCPU, so \
             the cached one is not written through; the write is refused and reported lost)",
            why.slug(),
            mode()
        ));
    }
}

/// One runtime-validation sample, read as a sandwich: `before` is what the CPU
/// read at `kva`, then the walk reads, then `cpu_again` reads through the CPU
/// once more. A guest store in between — on 13.6, the very page-table entries
/// being wired — moves `after` off `before`, and the sample is inconclusive
/// rather than a disagreement. Stable bytes on both sides make a disagreement
/// likely real, not proven: the guest can write and restore inside the window.
pub fn sample<M: HostMemory>(
    mem: &M,
    root: &KernelRoot,
    kva: u64,
    before: &[u8],
    cpu_again: impl FnOnce(&mut [u8]) -> bool,
    source: Source,
) -> WalkCompare {
    let mut walked = vec![0u8; before.len()];
    let walk = read_traced(mem, root, kva, &mut walked);
    let mut after = vec![0u8; before.len()];
    if !cpu_again(&mut after) || after != before {
        return WalkCompare::Inconclusive;
    }
    match walk {
        Ok(trace) if walked == before => {
            let granule = Regime::of(root).map_or(1 << 14, |r| r.granule());
            let crossed = trace.leaves.len() > 1;
            WalkCompare::Agreed {
                kva,
                len: before.len(),
                source,
                nonzero: before.iter().any(|&b| b != 0),
                via_pages: trace.leaves.iter().all(|(leaf, _)| *leaf == Leaf::Page),
                crossed,
                noncontiguous: crossed
                    && trace
                        .leaves
                        .windows(2)
                        .any(|w| w[1].1 != w[0].1.wrapping_add(granule)),
            }
        }
        Ok(_) => WalkCompare::Mismatch {
            kva,
            len: before.len(),
            why: "bytes",
        },
        Err(why) => WalkCompare::Mismatch {
            kva,
            len: before.len(),
            why: why.slug(),
        },
    }
}

/// The span a boundary-crossing sample reads, around the end of the granule
/// page holding `kva`. Wide, because a sample counts only if it holds a
/// non-zero byte, and the last bytes of a kernel page are often zeros.
pub fn crossing_probe(root: &KernelRoot, kva: u64) -> Option<(u64, usize)> {
    let granule = Regime::of(root).ok()?.granule();
    let end = (kva | (granule - 1)).checked_add(1)?;
    Some((end - CROSSING_HALF, 2 * CROSSING_HALF as usize))
}

/// One sandwiched sample of `len` bytes at `kva`, taken on a vCPU: `None` when
/// the CPU cannot read it at all.
pub fn sample_on_cpu<H: HostMemory + HostOps>(
    host: &H,
    root: &KernelRoot,
    kva: u64,
    len: usize,
    source: Source,
) -> Option<WalkCompare> {
    let mut before = vec![0u8; len];
    host.read_kva(kva, &mut before).ok()?;
    Some(sample(
        host,
        root,
        kva,
        &before,
        |buf| host.read_kva(kva, buf).is_ok(),
        source,
    ))
}

/// One round of active probes of kernel text, entered at `VBAR_EL1`: an
/// interior sample and a sample across the end of each of a few pages from
/// there. Nothing about those pages is assumed — not that they are mapped,
/// contiguous, or unchanging; a sample counts only through the same sandwich
/// as any other. Text mapped by blocks adds to the log, not to the threshold.
fn probe_text<H: HostMemory + HostOps>(root: &KernelRoot, host: &H) -> WalkLog {
    let mut log = WalkLog::default();
    let (Some(vbar), Ok(regime)) = (root.vbar, Regime::of(root)) else {
        return log;
    };
    let granule = regime.granule();
    let first = vbar & !(granule - 1);
    for page in (0..TEXT_PAGES).map(|k| first.wrapping_add(k * granule)) {
        let spans = [
            (page + 0x400, 64),
            (page + granule - CROSSING_HALF, 2 * CROSSING_HALF as usize),
        ];
        for (at, len) in spans {
            log.probes += 1;
            log.compares
                .extend(sample_on_cpu(host, root, at, len, Source::Text));
        }
    }
    log
}

/// Read the vCPU's EL1 translation state and keep it as the root.
///
/// Called on the vCPU thread, in the MMIO exit of a mapper ring publish, before
/// anything on that path reads a kernel address. The shim synchronizes the
/// state from KVM before answering, and the reads share one exit, so they
/// describe one moment. A changed root starts validation again. While the
/// root has not validated, each call also takes one round of kernel-text
/// probes, up to [`TEXT_ROUNDS`] per root.
pub fn capture<H: HostMemory + HostOps>(state: &mut KernelWalkState, host: &H) {
    capture_root(state, host);
    let Some(root) = state.root else {
        return;
    };
    if state.gate().is_ok() || state.text_rounds >= TEXT_ROUNDS || root.vbar.is_none() {
        return;
    }
    state.text_rounds += 1;
    let log = probe_text(&root, host);
    state.fold(log);
}

fn capture_root<H: HostOps>(state: &mut KernelWalkState, host: &H) {
    let read = |index| host.read_xreg(index);
    let regs = (|| {
        Some((
            read(SYSREG_TTBR1_EL1).ok()?,
            read(SYSREG_TCR_EL1).ok()?,
            read(SYSREG_SCTLR_EL1).ok()?,
            read(SYSREG_CPU_INDEX).ok()?,
        ))
    })();
    let Some((ttbr1, tcr, sctlr, cpu_index)) = regs else {
        // A snapshot that cannot be taken now says nothing about the one held:
        // it stops serving until a capture succeeds and validates again.
        if let Some(held) = state.root.take() {
            crate::observe::fail(format!(
                "kva_walk_root_lost was_ttbr1={:#x} boot={} (the translation state could \
                 not be read on this vCPU; the held root no longer serves)",
                held.ttbr1, state.boot
            ));
            state.validation = Validation::default();
            state.validated_reported = false;
        }
        if !state.unavailable_reported {
            state.unavailable_reported = true;
            crate::observe::off(format!(
                "kva_walk_root_unavailable mode={:?} (the shim does not answer EL1 translation \
                 state; without a vCPU, strict refuses mapping writes and compat writes through \
                 the cached page list)",
                mode()
            ));
        }
        return;
    };
    let root = KernelRoot {
        ttbr1,
        tcr,
        sctlr,
        cpu_index,
        boot: state.boot,
        // Optional: a shim that answers the translation state but not this
        // only loses the text probes.
        vbar: read(SYSREG_VBAR_EL1).ok(),
    };
    match state.root {
        Some(held) if held.same_translation(&root) => return,
        Some(held) => crate::observe::fail(format!(
            "kva_walk_root_changed was_ttbr1={:#x} was_tcr={:#x} was_sctlr={:#x} \
             ttbr1={ttbr1:#x} tcr={tcr:#x} sctlr={sctlr:#x} cpu={cpu_index} boot={} \
             (validation starts again for the new root)",
            held.ttbr1, held.tcr, held.sctlr, state.boot
        )),
        None => crate::observe::off(format!(
            "kva_walk_root ttbr1={ttbr1:#x} tcr={tcr:#x} sctlr={sctlr:#x} cpu={cpu_index} \
             vbar={:?} boot={} mode={:?} regime={:?}",
            root.vbar.map(|v| format!("{v:#x}")),
            state.boot,
            mode(),
            Regime::of(&root).map(|r| (r.granule_shift, r.va_bits, r.start_level))
        )),
    }
    state.root = Some(root);
    state.validation = Validation::default();
    state.validated_reported = false;
    state.text_rounds = 0;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::runtime::host::FakeHost;

    /// A 16 KiB-granule, 47-bit kernel regime: three levels, L1 to L3, wide enough
    /// for the `0xfffffe...` kernel addresses the mapper fixtures use.
    pub(crate) const TCR_16K_47: u64 = (17 << 16) | (0b01 << 30) | (0b101 << 32);
    pub(crate) const SCTLR_ON: u64 = SCTLR_M;
    pub(crate) const G: u64 = 1 << 14;
    /// Where the walk's own tables live in the fixture.
    pub(crate) const L1: u64 = 0x4000_0000;
    const L2: u64 = L1 + G;
    const L3: u64 = L1 + 2 * G;

    pub(crate) fn root(ttbr1: u64) -> KernelRoot {
        KernelRoot {
            ttbr1,
            tcr: TCR_16K_47,
            sctlr: SCTLR_ON,
            cpu_index: 0,
            boot: 0,
            vbar: None,
        }
    }

    fn put_desc(host: &mut FakeHost, at: u64, desc: u64) {
        host.map_range(at & !7, 8, 0);
        host.write_gpa(at, &desc.to_le_bytes()).unwrap();
    }

    /// A kernel page table rooted at [`L1`] that maps the 16 KiB page holding
    /// `va` to `pa` through one L2 table and one L3 table. Called again for
    /// another `va` in the same 32 MiB, it adds to the same L3 table.
    pub(crate) fn map_page(host: &mut FakeHost, va: u64, pa: u64) {
        let l1 = (va >> 36) & 0x7ff;
        let l2 = (va >> 25) & 0x7ff;
        let l3 = (va >> 14) & 0x7ff;
        put_desc(host, L1 + l1 * 8, L2 | 0b11);
        put_desc(host, L2 + l2 * 8, L3 | 0b11);
        put_desc(host, L3 + l3 * 8, (pa & !(G - 1)) | 0b11 | (1 << 10));
    }

    /// What a kernel VA in this regime looks like: the upper 17 bits set.
    pub(crate) fn kva(offset: u64) -> u64 {
        0xffff_8000_0000_0000 | offset
    }

    #[test]
    fn a_mapped_page_translates_with_its_offset() {
        let mut host = FakeHost::new();
        let va = kva(0x12_3456_4000);
        map_page(&mut host, va, 0x8_0000_0000);
        let r = root(L1);
        assert_eq!(translate(&host, &r, va + 0x123), Ok(0x8_0000_0123));
        assert_eq!(
            translate(&host, &r, va + G),
            Err(WalkRefusal::Fault { level: 3 }),
            "the next page has no descriptor"
        );
    }

    /// A 32 MiB level-2 block covers the address with the low 25 bits passed
    /// through, and a level-1 block does not exist on this granule.
    #[test]
    fn blocks_are_taken_where_the_granule_allows_them() {
        let mut host = FakeHost::new();
        let va = kva(0x40_0000_0000 + 0x1_2345);
        let l1 = (va >> 36) & 0x7ff;
        let l2 = (va >> 25) & 0x7ff;
        put_desc(&mut host, L1 + l1 * 8, L2 | 0b11);
        put_desc(&mut host, L2 + l2 * 8, 0x2_0000_0000 | 0b01);
        assert_eq!(
            translate(&host, &root(L1), va),
            Ok(0x2_0000_0000 + 0x1_2345)
        );

        put_desc(&mut host, L1 + l1 * 8, 0x2_0000_0000 | 0b01);
        assert_eq!(
            translate(&host, &root(L1), va),
            Err(WalkRefusal::BlockAtLevel { level: 1 })
        );
    }

    #[test]
    fn what_the_walk_does_not_implement_is_refused() {
        let host = FakeHost::new();
        let refuse = |tcr: u64, sctlr: u64| {
            translate(
                &host,
                &KernelRoot {
                    tcr,
                    sctlr,
                    ..root(L1)
                },
                kva(0),
            )
        };
        assert_eq!(refuse(TCR_16K_47, 0), Err(WalkRefusal::MmuOff));
        assert_eq!(
            refuse(TCR_16K_47, SCTLR_ON | SCTLR_EE),
            Err(WalkRefusal::BigEndian)
        );
        assert_eq!(
            refuse(TCR_16K_47 | TCR_EPD1, SCTLR_ON),
            Err(WalkRefusal::Ttbr1WalksDisabled)
        );
        assert_eq!(
            refuse(TCR_16K_47 | TCR_DS, SCTLR_ON),
            Err(WalkRefusal::Lpa2)
        );
        assert_eq!(
            refuse((TCR_16K_47 & !(0b11 << 30)) | (0b11 << 30), SCTLR_ON),
            Err(WalkRefusal::Granule64k)
        );
        assert_eq!(
            translate(&host, &root(L1), 0x0000_0080_0000_0000),
            Err(WalkRefusal::NotTtbr1Address),
            "a TTBR0 address is not the kernel's"
        );
    }

    /// A read crossing a page boundary takes each page from its own
    /// translation, including pages that are not physically adjacent.
    #[test]
    fn a_read_across_pages_follows_each_translation() {
        let mut host = FakeHost::new();
        let va = kva(0x10_0000_0000);
        map_page(&mut host, va, 0x9_0000_0000);
        map_page(&mut host, va + G, 0x7_0000_0000);
        host.map_range(0x9_0000_0000, G as usize, 0);
        host.map_range(0x7_0000_0000, G as usize, 0);
        host.write_gpa(0x9_0000_0000 + G - 4, b"tail").unwrap();
        host.write_gpa(0x7_0000_0000, b"head").unwrap();
        let mut got = [0u8; 8];
        read(&host, &root(L1), va + G - 4, &mut got).unwrap();
        assert_eq!(&got, b"tailhead");
    }

    fn agreed_from(source: Source, kva: u64, len: usize, crossed: bool) -> WalkCompare {
        WalkCompare::Agreed {
            kva,
            len,
            source,
            nonzero: true,
            via_pages: true,
            crossed,
            noncontiguous: crossed,
        }
    }

    fn agreed(kva: u64, len: usize, crossed: bool) -> WalkCompare {
        agreed_from(Source::Mapper, kva, len, crossed)
    }

    /// Runtime validation, as many agreeing non-zero reads through level-3
    /// pages over several pages, one of them across a boundary, as it takes and
    /// no fewer.
    fn validate(state: &mut KernelWalkState) {
        for i in 0..AGREED_SAMPLES {
            state.note_compare(agreed(kva(i * G), 8, i == 0));
        }
    }

    fn give_root(host: &mut FakeHost, ttbr1: u64) {
        host.set_xreg(SYSREG_TTBR1_EL1, ttbr1);
        host.set_xreg(SYSREG_TCR_EL1, TCR_16K_47);
        host.set_xreg(SYSREG_SCTLR_EL1, SCTLR_ON);
        host.set_xreg(SYSREG_CPU_INDEX, 2);
    }

    #[test]
    fn a_shim_that_does_not_answer_leaves_no_root() {
        let host = FakeHost::new();
        let mut state = KernelWalkState::default();
        capture(&mut state, &host);
        assert!(state.root.is_none());
        assert_eq!(state.gate(), Err(GateRefusal::NoRoot));
        assert_eq!(
            state.policy(Mode::Strict).serve,
            Serve::Refuse(GateRefusal::NoRoot)
        );
        assert_eq!(state.policy(Mode::Compat).serve, Serve::Off);
    }

    /// One agreeing read does not validate a root, an all-zero one counts for
    /// nothing, and one that never crossed a page is not enough either.
    #[test]
    fn validation_takes_several_non_zero_reads_over_pages_and_a_crossing() {
        let mut host = FakeHost::new();
        give_root(&mut host, L1);
        let mut state = KernelWalkState::default();
        capture(&mut state, &host);

        state.note_compare(agreed(kva(0), 8, true));
        assert_eq!(state.gate(), Err(GateRefusal::Unvalidated), "one read");

        for i in 0..2 * AGREED_SAMPLES {
            state.note_compare(WalkCompare::Agreed {
                kva: kva(i * G),
                len: 8,
                source: Source::Mapper,
                nonzero: false,
                via_pages: true,
                crossed: true,
                noncontiguous: true,
            });
        }
        assert_eq!(
            state.gate(),
            Err(GateRefusal::Unvalidated),
            "zeros prove nothing"
        );

        let mut blocks = KernelWalkState::default();
        capture(&mut blocks, &host);
        for i in 0..2 * AGREED_SAMPLES {
            blocks.note_compare(WalkCompare::Agreed {
                kva: kva(i * G),
                len: 8,
                source: Source::Mapper,
                nonzero: true,
                via_pages: false,
                crossed: true,
                noncontiguous: true,
            });
        }
        assert_eq!(
            blocks.gate(),
            Err(GateRefusal::Unvalidated),
            "samples through block descriptors do not fill the threshold"
        );

        let mut text_only = KernelWalkState::default();
        capture(&mut text_only, &host);
        for i in 0..2 * AGREED_SAMPLES {
            text_only.note_compare(agreed_from(Source::Text, kva(i * G), 8, true));
        }
        assert_eq!(
            text_only.gate(),
            Err(GateRefusal::Unvalidated),
            "kernel text alone cannot vouch for the mapper's own objects"
        );
        for i in 0..AGREED_MAPPER_SAMPLES {
            text_only.note_compare(agreed(kva(i * G), 8, false));
        }
        assert!(text_only.gate().is_ok(), "with enough of them it can");

        let mut uncrossed = KernelWalkState::default();
        capture(&mut uncrossed, &host);
        for i in 0..AGREED_SAMPLES {
            uncrossed.note_compare(agreed(kva(i * G), 8, false));
        }
        assert_eq!(
            uncrossed.gate(),
            Err(GateRefusal::Unvalidated),
            "no crossing"
        );

        validate(&mut state);
        assert_eq!(state.gate().map(|r| r.ttbr1), Ok(L1));
        assert_eq!(
            state.policy(Mode::Strict).serve,
            Serve::Walk(state.root.unwrap())
        );
    }

    /// Validation belongs to one root in one boot: a changed root, or a reset,
    /// starts it again.
    #[test]
    fn a_new_root_or_a_reset_starts_validation_again() {
        let mut host = FakeHost::new();
        give_root(&mut host, L1);
        let mut state = KernelWalkState::default();
        capture(&mut state, &host);
        validate(&mut state);
        assert!(state.gate().is_ok());

        capture(&mut state, &host);
        assert!(state.gate().is_ok(), "the same root keeps its validation");

        give_root(&mut host, L1 + 0x10_0000);
        capture(&mut state, &host);
        assert_eq!(state.root.map(|r| r.ttbr1), Some(L1 + 0x10_0000));
        assert_eq!(state.gate(), Err(GateRefusal::Unvalidated));

        validate(&mut state);
        state.reset();
        assert_eq!(state.gate(), Err(GateRefusal::NoRoot));
        assert_eq!(state.boot, 1);
    }

    fn stable_mismatch() -> WalkCompare {
        WalkCompare::Mismatch {
            kva: kva(0),
            len: 8,
            why: "bytes",
        }
    }

    /// The first stable disagreement suspends the walk at once; it serves again
    /// only after a full threshold of fresh samples, and a second disagreement in
    /// the root's life — agreements in between notwithstanding — revokes it until
    /// the root changes.
    #[test]
    fn a_disagreement_suspends_the_walk_and_a_second_revokes_it() {
        let mut host = FakeHost::new();
        give_root(&mut host, L1);
        let mut state = KernelWalkState::default();
        capture(&mut state, &host);
        validate(&mut state);
        state.note_compare(stable_mismatch());
        assert_eq!(state.gate(), Err(GateRefusal::Suspended));
        assert_eq!(
            state.policy(Mode::Strict).serve,
            Serve::Refuse(GateRefusal::Suspended)
        );
        state.note_compare(agreed(kva(0), 8, true));
        assert_eq!(
            state.gate(),
            Err(GateRefusal::Suspended),
            "one agreement does not lift it"
        );
        validate(&mut state);
        assert!(state.gate().is_ok(), "a full fresh threshold does");

        state.note_compare(stable_mismatch());
        validate(&mut state);
        assert_eq!(state.gate(), Err(GateRefusal::Mismatched));

        give_root(&mut host, L1 + 0x10_0000);
        capture(&mut state, &host);
        validate(&mut state);
        assert!(state.gate().is_ok(), "a new root starts clean");
    }

    /// A table address the output size cannot express is an address size fault
    /// at that level, before anything is read from it — even when the leaf it
    /// would lead to is a low address. So is a root table the same.
    #[test]
    fn a_table_beyond_the_output_size_is_refused_whatever_the_leaf() {
        const IPS_32: u64 = (17 << 16) | (0b01 << 30);
        let r = KernelRoot {
            tcr: IPS_32,
            ..root(L1)
        };
        let mut host = FakeHost::new();
        let va = kva(0x30_0000_0000);
        let high_l2 = 0x1_0000_0000u64;
        let l3 = 0x5000_0000u64;
        put_desc(&mut host, L1 + ((va >> 36) & 0x7ff) * 8, high_l2 | 0b11);
        put_desc(&mut host, high_l2 + ((va >> 25) & 0x7ff) * 8, l3 | 0b11);
        put_desc(&mut host, l3 + ((va >> 14) & 0x7ff) * 8, 0x6000_0000 | 0b11);
        assert_eq!(
            translate(&host, &r, va),
            Err(WalkRefusal::AddressSize { level: 1 })
        );
        assert_eq!(
            translate(
                &host,
                &KernelRoot {
                    ttbr1: high_l2,
                    ..r
                },
                va
            ),
            Err(WalkRefusal::AddressSize { level: 0 }),
            "the root table itself"
        );
        assert_eq!(
            translate(
                &host,
                &KernelRoot {
                    tcr: IPS_32 | (0b111 << 32),
                    ..r
                },
                va
            ),
            Err(WalkRefusal::IpsReserved)
        );
    }

    /// A capture that fails takes the held root out of service: a stale
    /// snapshot must not keep serving. A later capture validates afresh.
    #[test]
    fn a_failed_capture_revokes_the_held_root() {
        let mut host = FakeHost::new();
        give_root(&mut host, L1);
        let mut state = KernelWalkState::default();
        capture(&mut state, &host);
        validate(&mut state);
        assert!(state.gate().is_ok());

        host.xregs.clear();
        capture(&mut state, &host);
        assert_eq!(state.gate(), Err(GateRefusal::NoRoot));

        give_root(&mut host, L1);
        capture(&mut state, &host);
        assert_eq!(state.gate(), Err(GateRefusal::Unvalidated));
        validate(&mut state);
        assert!(state.gate().is_ok());
    }

    /// A guest store between the CPU read and the walk — what wiring a page
    /// table does — makes the sample inconclusive: it neither validates nor
    /// revokes, however often it happens.
    #[test]
    fn a_concurrent_change_is_inconclusive_and_revokes_nothing() {
        let mut host = FakeHost::new();
        let va = kva(0x20_0000_0000);
        map_page(&mut host, va, 0x9_0000_0000);
        host.map_range(0x9_0000_0000, G as usize, 0);
        host.write_gpa(0x9_0000_0000, b"after!!!").unwrap();
        let r = root(L1);
        let moved = |buf: &mut [u8]| {
            buf.copy_from_slice(b"after!!!");
            true
        };
        assert_eq!(
            sample(&host, &r, va, b"before!!", moved, Source::Mapper),
            WalkCompare::Inconclusive
        );
        assert_eq!(
            sample(
                &host,
                &r,
                va,
                b"before!!",
                |_: &mut [u8]| false,
                Source::Mapper
            ),
            WalkCompare::Inconclusive,
            "a second CPU read that failed proves nothing either"
        );

        give_root(&mut host, L1);
        let mut state = KernelWalkState::default();
        capture(&mut state, &host);
        validate(&mut state);
        for _ in 0..10 {
            state.note_compare(WalkCompare::Inconclusive);
        }
        assert!(state.gate().is_ok());
    }

    /// Kernel text from `VBAR_EL1`, mapped page by page at scattered physical
    /// pages, for both the CPU and the guest's tables.
    fn text_fixture(host: &mut FakeHost) -> u64 {
        let text = kva(0x40_0000_0000);
        for k in 0..=TEXT_PAGES {
            let pa = 0x9_0000_0000 + k * 0x10_0000;
            map_page(host, text + k * G, pa);
            host.map_range(pa, G as usize, 0xd5);
            host.kva_cpu_pages.insert(text + k * G, pa);
        }
        give_root(host, L1);
        host.set_xreg(SYSREG_VBAR_EL1, text + 0x800);
        text
    }

    /// The probes taken at capture count toward validation through level-3
    /// pages, crossings between scattered leaves included, but cannot finish
    /// it: the mapper's own reads still have to agree. They run once per
    /// capture while the root validates, and stop after a bounded number.
    #[test]
    fn text_probes_help_validation_and_stop() {
        let mut host = FakeHost::new();
        text_fixture(&mut host);
        let mut state = KernelWalkState::default();
        capture(&mut state, &host);
        assert_eq!(state.text_rounds, 1);
        assert_eq!(state.validation.samples, 2 * TEXT_PAGES);
        assert!(state.validation.crossed && state.validation.crossed_noncontiguous);
        assert_eq!(state.gate(), Err(GateRefusal::Unvalidated));

        for _ in 0..2 * TEXT_ROUNDS {
            capture(&mut state, &host);
        }
        assert_eq!(state.text_rounds, TEXT_ROUNDS, "bounded per root");

        for i in 0..AGREED_MAPPER_SAMPLES {
            state.note_compare(agreed(kva(i * G), 8, false));
        }
        assert!(state.gate().is_ok());
        let rounds = state.text_rounds;
        capture(&mut state, &host);
        assert_eq!(state.text_rounds, rounds, "no probes once validated");
    }

    /// Text the guest maps with a block is sampled and logged, and adds
    /// nothing to the threshold.
    #[test]
    fn text_mapped_by_a_block_does_not_fill_the_threshold() {
        let mut host = FakeHost::new();
        let text = kva(0x50_0000_0000);
        let l1 = (text >> 36) & 0x7ff;
        let l2 = (text >> 25) & 0x7ff;
        let block = 0x6_0000_0000u64;
        put_desc(&mut host, L1 + l1 * 8, L2 | 0b11);
        put_desc(&mut host, L2 + l2 * 8, block | 0b01);
        host.map_range(block, 8 * G as usize, 0xd5);
        for k in 0..8 {
            host.kva_cpu_pages.insert(text + k * G, block + k * G);
        }
        give_root(&mut host, L1);
        host.set_xreg(SYSREG_VBAR_EL1, text);
        let mut state = KernelWalkState::default();
        capture(&mut state, &host);
        assert_eq!(state.validation.samples, 0);
        assert_eq!(state.validation.via_blocks, 2 * TEXT_PAGES);
    }

    /// A probe the CPU cannot read is skipped, not counted either way.
    #[test]
    fn an_unreadable_text_page_is_skipped() {
        let mut host = FakeHost::new();
        let text = text_fixture(&mut host);
        host.kva_cpu_pages.remove(&(text + G));
        let mut state = KernelWalkState::default();
        capture(&mut state, &host);
        let v = &state.validation;
        assert_eq!(v.mismatches, 0);
        assert!(
            v.samples < 2 * TEXT_PAGES && v.samples > 0,
            "{}",
            v.describe()
        );
    }

    /// The sample itself, with the CPU reading the same bytes on both sides:
    /// equal bytes agree, a different translation is a stable disagreement, and
    /// a read the CPU answered but the walk cannot is one too.
    #[test]
    fn a_stable_sample_names_agreement_by_bytes() {
        let mut host = FakeHost::new();
        let va = kva(0x20_0000_0000);
        map_page(&mut host, va, 0x9_0000_0000);
        host.map_range(0x9_0000_0000, G as usize, 0);
        host.write_gpa(0x9_0000_0000, b"kernel!!").unwrap();
        let r = root(L1);
        let steady = |bytes: &'static [u8]| {
            move |buf: &mut [u8]| {
                buf.copy_from_slice(bytes);
                true
            }
        };
        assert_eq!(
            sample(
                &host,
                &r,
                va,
                b"kernel!!",
                steady(b"kernel!!"),
                Source::Mapper
            ),
            WalkCompare::Agreed {
                kva: va,
                len: 8,
                source: Source::Mapper,
                nonzero: true,
                via_pages: true,
                crossed: false,
                noncontiguous: false,
            }
        );
        assert!(matches!(
            sample(
                &host,
                &r,
                va,
                b"elsewhre",
                steady(b"elsewhre"),
                Source::Mapper
            ),
            WalkCompare::Mismatch { why: "bytes", .. }
        ));
        assert!(matches!(
            sample(
                &host,
                &r,
                va + G,
                b"unmapped",
                steady(b"unmapped"),
                Source::Mapper
            ),
            WalkCompare::Mismatch {
                why: "kva_walk_fault",
                ..
            }
        ));
        assert_eq!(crossing_probe(&r, va + 0x10), Some((va + G - 128, 256)));
    }
}
