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
//! are read through this walk and compared ([`compare`]). A root serves a thread
//! without a vCPU only after it has collected several agreeing non-zero reads
//! over several pages, one of them across a page boundary, and no disagreement.
//! Validation belongs to one root in one boot: a changed root or a reset starts
//! it again, and one disagreement retires the root until it changes.
//!
//! Without a validated root — a QEMU without the matching shim, a root still
//! validating, a disagreement — [`Mode::Strict`] (the default) refuses the
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
    translate_in(mem, &regime, va)
}

fn translate_in<M: HostMemory>(mem: &M, r: &Regime, va: u64) -> Result<u64, WalkRefusal> {
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
        let pa = if level == 3 {
            if !is_table_or_page {
                return Err(WalkRefusal::Reserved { level });
            }
            (desc & oa_mask) | (va & (span - 1))
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
            (desc & oa_mask & !(span - 1)) | (va & (span - 1))
        };
        if pa >> r.pa_bits != 0 {
            return Err(WalkRefusal::AddressSize {
                level: level as i32,
            });
        }
        return Ok(pa);
    }
    unreachable!("level 3 always returns")
}

/// Read `buf.len()` bytes at guest kernel address `va` through the walk, one
/// granule at a time.
pub fn read<M: HostMemory>(
    mem: &M,
    root: &KernelRoot,
    va: u64,
    buf: &mut [u8],
) -> Result<(), WalkRefusal> {
    let regime = Regime::of(root)?;
    let granule = regime.granule();
    let mut done = 0usize;
    while done < buf.len() {
        let at = va.wrapping_add(done as u64);
        let pa = translate_in(mem, &regime, at)?;
        let in_page = (granule - (at & (granule - 1))) as usize;
        let n = in_page.min(buf.len() - done);
        mem.read_gpa(pa, &mut buf[done..done + n])
            .map_err(|_| WalkRefusal::Read { gpa: pa })?;
        done += n;
    }
    Ok(())
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
/// translate with it: reads with non-zero bytes, from several pages, and one
/// that crossed a page boundary. One matching read, all zeros, proves nothing
/// about the physical address it came from.
const AGREED_SAMPLES: u64 = 8;
const AGREED_PAGES: usize = 3;
/// Distinct pages remembered for [`AGREED_PAGES`]; more add nothing.
const PAGES_KEPT: usize = 16;

/// The result of reading the same kernel bytes through the CPU and the walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkCompare {
    Agreed {
        kva: u64,
        len: usize,
        nonzero: bool,
        crossed: bool,
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

/// Disagreements that retire a root. One can be noise the sandwich did not
/// catch; two are a pattern.
const MISMATCHES_TO_REVOKE: u32 = 2;

/// Bytes a crossing probe reads on each side of a page boundary.
const CROSSING_HALF: u64 = 128;

/// Runtime validation of one root against the CPU.
#[derive(Clone, Debug, Default)]
struct Validation {
    samples: u64,
    pages: std::collections::BTreeSet<u64>,
    crossed: bool,
    mismatches: u32,
}

impl Validation {
    fn revoked(&self) -> bool {
        self.mismatches >= MISMATCHES_TO_REVOKE
    }

    fn validated(&self) -> bool {
        !self.revoked()
            && self.samples >= AGREED_SAMPLES
            && self.pages.len() >= AGREED_PAGES
            && self.crossed
    }
}

/// Why a thread without a vCPU has no walk to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateRefusal {
    NoRoot,
    Unvalidated,
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
}

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
            return Err(GateRefusal::Unvalidated);
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
                 validated={} (runtime validation work on the vCPU this boot)",
                self.boot,
                c.samples,
                c.probes,
                c.inconclusive,
                c.mismatches,
                self.gate().is_ok()
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
                nonzero,
                crossed,
            } => {
                crate::runtime::drain::note_store_route("kva_walk_agree");
                if !nonzero || v.revoked() {
                    return;
                }
                v.samples += 1;
                v.crossed |= crossed;
                if v.pages.len() < PAGES_KEPT {
                    let granule = Regime::of(&root).map_or(1 << 14, |r| r.granule());
                    v.pages.insert(kva & !(granule - 1));
                    let last = kva.wrapping_add(len.max(1) as u64 - 1);
                    v.pages.insert(last & !(granule - 1));
                }
                if v.validated() && !self.validated_reported {
                    self.validated_reported = true;
                    crate::observe::off(format!(
                        "kva_walk_validated samples={} pages={} ttbr1={:#x} tcr={:#x} \
                         boot={} (runtime validation: the software walk agreed with the CPU; \
                         it now serves threads without a vCPU)",
                        v.samples,
                        v.pages.len(),
                        root.ttbr1,
                        root.tcr,
                        root.boot
                    ));
                }
            }
            WalkCompare::Mismatch { kva, len, why } => {
                crate::runtime::drain::note_store_route("kva_walk_mismatch");
                v.mismatches += 1;
                if v.mismatches <= MISMATCHES_TO_REVOKE {
                    crate::observe::fail(format!(
                        "kva_walk_mismatch kva={kva:#x} len={len} detail={why} samples={} \
                         mismatches={} ttbr1={:#x} revoked={} (runtime validation: the CPU \
                         read the same bytes before and after, and the software walk read \
                         others; at {MISMATCHES_TO_REVOKE} the walk serves nothing until the \
                         root changes)",
                        v.samples,
                        v.mismatches,
                        root.ttbr1,
                        v.revoked()
                    ));
                }
            }
            WalkCompare::Inconclusive => {
                crate::runtime::drain::note_store_route("kva_walk_inconclusive");
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
) -> WalkCompare {
    let mut walked = vec![0u8; before.len()];
    let walk = read(mem, root, kva, &mut walked);
    let mut after = vec![0u8; before.len()];
    if !cpu_again(&mut after) || after != before {
        return WalkCompare::Inconclusive;
    }
    match walk {
        Ok(()) if walked == before => {
            let granule = Regime::of(root).map_or(1 << 14, |r| r.granule());
            let last = kva.wrapping_add(before.len().max(1) as u64 - 1);
            WalkCompare::Agreed {
                kva,
                len: before.len(),
                nonzero: before.iter().any(|&b| b != 0),
                crossed: kva & !(granule - 1) != last & !(granule - 1),
            }
        }
        Ok(()) => WalkCompare::Mismatch {
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

/// Read the vCPU's EL1 translation state and keep it as the root.
///
/// Called on the vCPU thread, in the MMIO exit of a mapper ring publish, before
/// anything on that path reads a kernel address. The shim synchronizes the
/// state from KVM before answering, and the four reads share one exit, so they
/// describe one moment. A changed root starts validation again.
pub fn capture<H: HostOps>(state: &mut KernelWalkState, host: &H) {
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
             boot={} mode={:?} regime={:?}",
            state.boot,
            mode(),
            Regime::of(&root).map(|r| (r.granule_shift, r.va_bits, r.start_level))
        )),
    }
    state.root = Some(root);
    state.validation = Validation::default();
    state.validated_reported = false;
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

    fn agreed(kva: u64, len: usize, crossed: bool) -> WalkCompare {
        WalkCompare::Agreed {
            kva,
            len,
            nonzero: true,
            crossed,
        }
    }

    /// Runtime validation, as many agreeing non-zero reads over several
    /// pages, one of them across a boundary, as it takes and no fewer.
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

        for _ in 0..2 * AGREED_SAMPLES {
            state.note_compare(WalkCompare::Agreed {
                kva: kva(G),
                len: 8,
                nonzero: false,
                crossed: false,
            });
        }
        assert_eq!(
            state.gate(),
            Err(GateRefusal::Unvalidated),
            "zeros prove nothing"
        );

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

    /// One stable disagreement is logged and tolerated as possible noise; the
    /// second retires the root until it changes.
    #[test]
    fn two_stable_disagreements_retire_the_walk_until_the_root_changes() {
        let mut host = FakeHost::new();
        give_root(&mut host, L1);
        let mut state = KernelWalkState::default();
        capture(&mut state, &host);
        validate(&mut state);
        state.note_compare(stable_mismatch());
        assert!(state.gate().is_ok(), "one disagreement does not revoke");
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
            sample(&host, &r, va, b"before!!", moved),
            WalkCompare::Inconclusive
        );
        assert_eq!(
            sample(&host, &r, va, b"before!!", |_: &mut [u8]| false),
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
            sample(&host, &r, va, b"kernel!!", steady(b"kernel!!")),
            WalkCompare::Agreed {
                kva: va,
                len: 8,
                nonzero: true,
                crossed: false
            }
        );
        assert!(matches!(
            sample(&host, &r, va, b"elsewhre", steady(b"elsewhre")),
            WalkCompare::Mismatch { why: "bytes", .. }
        ));
        assert!(matches!(
            sample(&host, &r, va + G, b"unmapped", steady(b"unmapped")),
            WalkCompare::Mismatch {
                why: "kva_walk_fault",
                ..
            }
        ));
        assert_eq!(crossing_probe(&r, va + 0x10), Some((va + G - 128, 256)));
    }
}
