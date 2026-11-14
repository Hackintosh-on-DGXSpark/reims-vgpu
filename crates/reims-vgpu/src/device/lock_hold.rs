//! How long each holder keeps the device lock, how long a vCPU waited for it,
//! and the order blocked threads get it in.
//!
//! # Why
//!
//! A vCPU that traps into this device on the mapper ring blocks on the device
//! lock (`lock_device_for_vcpu`) for as long as the drain worker holds it, and
//! a vCPU stopped in an MMIO exit answers no interrupt — including the guest
//! kernel's cross-call to stop every CPU for its debugger. A panic reading
//! `DebuggerXCallEnter … synch pending on cpu N` is what that looks like from
//! the guest. `vcpu_lock_wait` already counts the waits once a second; this
//! names the holder that made them long, with a distribution rather than a sum.
//!
//! # Order
//!
//! Every release goes through [`Held`] and is fair (`unlock_fair`): when a
//! thread is parked on the lock, the lock passes straight to the one that
//! parked first instead of being left free for whichever thread reaches it
//! next. The threads that block on it — the drain worker, a vCPU writing a
//! mapper register, the console's scanout copy, device reset — are served in the
//! order they blocked. A vCPU waits for the hold in progress and the waiters
//! ahead of it, not for a worker that comes straight back for the lock; the
//! worker waits for the vCPUs ahead of it, not for every vCPU that arrives
//! while it waits; and a `try_lock` caller cannot take a lock that is being
//! handed over.
//!
//! An ordinary parking_lot release hands over like this only when its
//! fairness timer has run out, at most a millisecond apart. Any other release
//! leaves the lock free and wakes the first waiter to compete for it, and the
//! worker, which goes from one tranche to the next without parking, can take
//! it again first.
//!
//! Nothing here waits, backs off or takes a lock that was not taken before: a
//! release still returns at once, and what each holder does under the lock,
//! and in what order, is unchanged. So it cannot shorten a hold either. A vCPU
//! blocked behind one long tranche waits for all of it.
//!
//! # One wait
//!
//! A vCPU registers before it blocks ([`WaitTicket`]), and while it is
//! registered every acquisition and release of the lock is written into its
//! record: how many times the drain worker took the lock first, how many times
//! anyone else did, how much of the wait the lock was held, and the longest
//! hold that ended during it. The rest of the wait, with the lock free and the
//! vCPU not yet running, is time off the CPU; that is what tells a descheduled
//! vCPU from a starved one. A wait past [`REPORT_THRESHOLD_US`] writes its
//! record as one `device_lock_wait` line once the vCPU has released.
//!
//! A hold is counted before its release and a wait before its registration
//! ends, and while a vCPU is registered no release writes the interval's line,
//! so a wait and the hold that caused it are on the same `device_lock_hold`
//! line. They used to be a line apart: the worker's release wrote the line
//! before the vCPU it had released counted its wait.
//!
//! The record lives behind a small mutex of its own, a leaf: it is taken with
//! the device lock held or with nothing held, and nothing is taken under it.
//!
//! # Cost
//!
//! Two clock reads, four relaxed atomic adds and two atomic loads per
//! acquisition. While a vCPU is registered, each acquisition and release also
//! takes the record's mutex. A line is written at most once per
//! [`REPORT_INTERVAL_MS`], and only when some hold or wait in that interval
//! reached [`REPORT_THRESHOLD_US`]; a boot summary is written at device reset.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::{Mutex, MutexGuard};

/// Who took the device lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holder {
    /// The drain worker's tranche (`device_drain`).
    Drain,
    /// `device_reset`.
    Reset,
    /// A vCPU writing an IOSurface mapper register: the ring publish drains
    /// synchronously. Reads take no lock.
    VcpuIosfcWrite,
    /// A vCPU reading a GFX register under the lock (`try_lock` only).
    VcpuGfxRead,
    /// A vCPU applying a GFX register write directly (`try_lock` only).
    VcpuGfxWrite,
    /// The host console copying a presented surface.
    ScanoutCopy,
    /// The 4 ms poll (`try_lock` only).
    Poll,
    /// The main loop taking a queued host action (`try_lock` only).
    PopAction,
    /// The console feed, EFI console and cursor readers (`try_lock` only).
    Display,
}

const HOLDERS: usize = 9;

impl Holder {
    const ALL: [Holder; HOLDERS] = [
        Holder::Drain,
        Holder::Reset,
        Holder::VcpuIosfcWrite,
        Holder::VcpuGfxRead,
        Holder::VcpuGfxWrite,
        Holder::ScanoutCopy,
        Holder::Poll,
        Holder::PopAction,
        Holder::Display,
    ];

    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::Drain => "drain",
            Self::Reset => "reset",
            Self::VcpuIosfcWrite => "vcpu_iosfc_write",
            Self::VcpuGfxRead => "vcpu_gfx_read",
            Self::VcpuGfxWrite => "vcpu_gfx_write",
            Self::ScanoutCopy => "scanout_copy",
            Self::Poll => "poll",
            Self::PopAction => "pop_action",
            Self::Display => "display",
        }
    }

    /// The drain worker's tranche, or a reset taking the lock the same way.
    fn is_drain(self) -> bool {
        matches!(self, Self::Drain | Self::Reset)
    }
}

/// A vCPU's access to an IOSurface mapper register, timed from entry to
/// return. While QEMU dispatches the region under its BQL, which it does
/// unless the region is lockless, this is how long that vCPU held the BQL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IosfcCall {
    Read,
    Write,
}

const IOSFC_CALLS: usize = 2;

impl IosfcCall {
    const ALL: [IosfcCall; IOSFC_CALLS] = [IosfcCall::Read, IosfcCall::Write];

    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
        }
    }
}

/// A report is written at most this often.
pub const REPORT_INTERVAL_MS: u64 = 10_000;
/// ...and only when a hold or a wait in the interval reached this.
pub const REPORT_THRESHOLD_US: u64 = 50_000;
/// At most this many `device_lock_wait` lines per report interval.
const WAIT_LINES_PER_INTERVAL: u64 = 16;

/// Log2 buckets of microseconds: bucket `i` holds `[2^i, 2^(i+1))`, bucket 0
/// also holds 0. Thirty-two reach past an hour.
const BUCKETS: usize = 32;

fn bucket(us: u64) -> usize {
    (63 - us.max(1).leading_zeros() as usize).min(BUCKETS - 1)
}

// MARK: - vCPUs

/// vCPUs with a wait distribution of their own. The last slot is shared by
/// every thread whose name gives no index, and by vCPUs past it.
const CPU_SLOTS: usize = 17;
const OTHER_CPU: usize = CPU_SLOTS - 1;

fn cpu_name(cpu: usize) -> String {
    match cpu {
        OTHER_CPU => "vcpu_other".to_owned(),
        cpu => format!("vcpu{cpu}"),
    }
}

/// The index in a QEMU vCPU thread's name, `CPU 3/KVM`: the guest's CPU
/// number, the one a guest panic names.
pub fn vcpu_index(thread_name: &str) -> Option<usize> {
    let (index, _accelerator) = thread_name.strip_prefix("CPU ")?.split_once('/')?;
    index.parse().ok()
}

/// This thread's slot, read from its name the first time it is asked for.
fn this_cpu() -> usize {
    thread_local! {
        static CPU: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    }
    CPU.with(|slot| match slot.get() {
        Some(cpu) => cpu,
        None => {
            let cpu = std::fs::read_to_string("/proc/thread-self/comm")
                .ok()
                .and_then(|name| vcpu_index(name.trim_end()))
                .filter(|&cpu| cpu < OTHER_CPU)
                .unwrap_or(OTHER_CPU);
            slot.set(Some(cpu));
            cpu
        }
    })
}

// MARK: - Distributions

/// One distribution of durations.
#[derive(Debug)]
struct Durations {
    count: AtomicU64,
    max_us: AtomicU64,
    buckets: [AtomicU64; BUCKETS],
}

impl Default for Durations {
    fn default() -> Self {
        Self {
            count: AtomicU64::new(0),
            max_us: AtomicU64::new(0),
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

/// What one distribution held when it was read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub count: u64,
    pub max_us: u64,
    /// The upper bound of the log2 bucket holding the 99th percentile.
    pub p99_us: u64,
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.count, self.max_us, self.p99_us)
    }
}

impl Durations {
    fn note(&self, us: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.max_us.fetch_max(us, Ordering::Relaxed);
        self.buckets[bucket(us)].fetch_add(1, Ordering::Relaxed);
    }

    /// Read, and clear when `take`.
    fn summary(&self, take: bool) -> Summary {
        let read = |a: &AtomicU64| match take {
            true => a.swap(0, Ordering::Relaxed),
            false => a.load(Ordering::Relaxed),
        };
        let buckets: Vec<u64> = self.buckets.iter().map(read).collect();
        let count = read(&self.count);
        let max_us = read(&self.max_us);
        let total: u64 = buckets.iter().sum();
        let want = total - total / 100;
        let mut seen = 0;
        let mut p99_us = 0;
        for (i, n) in buckets.iter().enumerate() {
            seen += n;
            if *n != 0 && seen >= want {
                p99_us = (1u64 << (i + 1)).min(max_us.max(1));
                break;
            }
        }
        Summary {
            count,
            max_us,
            p99_us,
        }
    }
}

/// One period's distributions: a report interval, or a boot.
#[derive(Debug)]
struct Period {
    holds: [Durations; HOLDERS],
    vcpu_wait: Durations,
    vcpu_wait_by_cpu: [Durations; CPU_SLOTS],
    /// The drain worker's contended waits for the lock: its time without
    /// progress that this lock accounts for.
    drain_wait: Durations,
    /// Drain releases made while a vCPU waited, each handing the lock on
    /// rather than leaving it to be taken again.
    drain_handoffs: AtomicU64,
    /// Each vCPU's mapper register accesses, entry to return, by kind.
    iosfc_calls: [[Durations; CPU_SLOTS]; IOSFC_CALLS],
}

impl Default for Period {
    fn default() -> Self {
        Self {
            holds: std::array::from_fn(|_| Durations::default()),
            vcpu_wait: Durations::default(),
            vcpu_wait_by_cpu: std::array::from_fn(|_| Durations::default()),
            drain_wait: Durations::default(),
            drain_handoffs: AtomicU64::new(0),
            iosfc_calls: std::array::from_fn(|_| std::array::from_fn(|_| Durations::default())),
        }
    }
}

/// A period read out, ready to render.
struct PeriodSummary {
    holds: Vec<(Holder, Summary)>,
    vcpu_wait: Summary,
    vcpu_wait_by_cpu: Vec<(usize, Summary)>,
    drain_wait: Summary,
    drain_handoffs: u64,
    iosfc_calls: Vec<(IosfcCall, usize, Summary)>,
}

impl Period {
    fn summary(&self, take: bool) -> PeriodSummary {
        PeriodSummary {
            holds: Holder::ALL
                .iter()
                .map(|&h| (h, self.holds[h.index()].summary(take)))
                .collect(),
            vcpu_wait: self.vcpu_wait.summary(take),
            vcpu_wait_by_cpu: (0..CPU_SLOTS)
                .map(|cpu| (cpu, self.vcpu_wait_by_cpu[cpu].summary(take)))
                .collect(),
            drain_wait: self.drain_wait.summary(take),
            drain_handoffs: match take {
                true => self.drain_handoffs.swap(0, Ordering::Relaxed),
                false => self.drain_handoffs.load(Ordering::Relaxed),
            },
            iosfc_calls: IosfcCall::ALL
                .iter()
                .flat_map(|&call| {
                    (0..CPU_SLOTS).map(move |cpu| {
                        (call, cpu, self.iosfc_calls[call.index()][cpu].summary(take))
                    })
                })
                .collect(),
        }
    }
}

impl PeriodSummary {
    fn is_empty(&self) -> bool {
        self.vcpu_wait.count == 0
            && self.drain_wait.count == 0
            && self.holds.iter().all(|(_, s)| s.count == 0)
            && self.iosfc_calls.iter().all(|(_, _, s)| s.count == 0)
    }

    fn worst_us(&self) -> u64 {
        let hold = self.holds.iter().map(|(_, s)| s.max_us).max().unwrap_or(0);
        let call = self
            .iosfc_calls
            .iter()
            .map(|(_, _, s)| s.max_us)
            .max()
            .unwrap_or(0);
        hold.max(call)
            .max(self.vcpu_wait.max_us)
            .max(self.drain_wait.max_us)
    }

    /// Everything after the leading vCPU wait fields.
    fn render(&self) -> String {
        let mut fields: Vec<String> = self
            .holds
            .iter()
            .filter(|(_, s)| s.count != 0)
            .map(|(h, s)| format!("{}={s}", h.name()))
            .collect();
        fields.push(format!("drain_wait={}", self.drain_wait));
        fields.push(format!("drain_handoffs={}", self.drain_handoffs));
        fields.extend(
            self.vcpu_wait_by_cpu
                .iter()
                .filter(|(_, s)| s.count != 0)
                .map(|(cpu, s)| format!("{}_wait={s}", cpu_name(*cpu))),
        );
        fields.extend(
            self.iosfc_calls
                .iter()
                .filter(|(_, _, s)| s.count != 0)
                .map(|(call, cpu, s)| format!("{}_iosfc_{}_call={s}", cpu_name(*cpu), call.name())),
        );
        fields.join(" ")
    }
}

// MARK: - One wait

/// A vCPU blocked on the lock, and what happened while it waited.
#[derive(Clone, Copy, Debug)]
struct OpenWait {
    ticket: u64,
    since: Instant,
    drain_first: u64,
    others_first: u64,
    held_us: u64,
    longest_us: u64,
    longest: Option<Holder>,
}

/// What one vCPU wait came to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaitRecord {
    /// The vCPU's slot: its index, or the shared last slot.
    pub cpu: usize,
    pub wait_us: u64,
    /// Times the drain worker (or a reset) took the lock while this vCPU
    /// waited.
    pub drain_first: u64,
    /// Times anyone else did: another vCPU, the scanout copy, a `try_lock`
    /// caller.
    pub others_first: u64,
    /// The part of the wait in which someone held the lock.
    pub held_us: u64,
    /// The longest hold that ended during the wait, all of it, and its holder.
    pub longest_us: u64,
    pub longest: Option<Holder>,
}

impl WaitRecord {
    /// The part of the wait in which nobody held the lock: the vCPU had been
    /// handed it, or could have taken it, and was not running yet.
    pub fn free_us(&self) -> u64 {
        self.wait_us.saturating_sub(self.held_us)
    }

    fn line(&self, at: Holder) -> String {
        format!(
            "device_lock_wait {} at={} wait_us={} drain_first={} others_first={} held_us={} \
             free_us={} longest_hold_us={} longest_holder={} (one vCPU wait for the device lock \
             past 50 ms: drain_first/others_first count who took the lock while it waited, \
             held_us the part of the wait the lock was held, free_us the rest, the lock free and \
             this vCPU not yet running)",
            cpu_name(self.cpu),
            at.name(),
            self.wait_us,
            self.drain_first,
            self.others_first,
            self.held_us,
            self.free_us(),
            self.longest_us,
            self.longest.map_or("none", Holder::name),
        )
    }
}

/// A vCPU registered as blocked on the lock. Closing it counts the wait;
/// dropping it unclosed, on a panic or an early return, only deregisters it,
/// so the count of waiters cannot be left raised.
pub struct WaitTicket<'a> {
    census: &'a LockCensus,
    ticket: u64,
    cpu: usize,
    since: Instant,
    open: bool,
}

impl WaitTicket<'_> {
    /// The lock is held: count the wait, then deregister.
    fn close(mut self) -> WaitRecord {
        self.open = false;
        let wait_us = self.since.elapsed().as_micros() as u64;
        let wait = self.census.take_wait(self.ticket);
        let record = WaitRecord {
            cpu: self.cpu,
            wait_us,
            drain_first: wait.map_or(0, |w| w.drain_first),
            others_first: wait.map_or(0, |w| w.others_first),
            held_us: wait.map_or(0, |w| w.held_us),
            longest_us: wait.map_or(0, |w| w.longest_us),
            longest: wait.and_then(|w| w.longest),
        };
        self.census.note_vcpu_wait(self.cpu, wait_us);
        // After the wait is counted: a report that finds no waiter has it.
        self.census.vcpu_waiters.fetch_sub(1, Ordering::AcqRel);
        record
    }
}

impl Drop for WaitTicket<'_> {
    fn drop(&mut self) {
        if self.open {
            self.census.take_wait(self.ticket);
            self.census.vcpu_waiters.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

// MARK: - Census

/// Holds per holder, and vCPU waits, for the current report interval and for
/// the boot; and the vCPUs blocked right now.
#[derive(Debug)]
pub struct LockCensus {
    interval: Period,
    boot: Period,
    last_report_ms: AtomicU64,
    wait_lines: AtomicU64,
    vcpu_waiters: AtomicU64,
    open_waits: Mutex<Vec<OpenWait>>,
    next_ticket: AtomicU64,
}

impl Default for LockCensus {
    fn default() -> Self {
        Self {
            interval: Period::default(),
            boot: Period::default(),
            last_report_ms: AtomicU64::new(0),
            wait_lines: AtomicU64::new(0),
            vcpu_waiters: AtomicU64::new(0),
            open_waits: Mutex::new(Vec::new()),
            next_ticket: AtomicU64::new(0),
        }
    }
}

impl LockCensus {
    pub fn note_hold(&self, holder: Holder, us: u64) {
        self.interval.holds[holder.index()].note(us);
        self.boot.holds[holder.index()].note(us);
    }

    fn note_vcpu_wait(&self, cpu: usize, us: u64) {
        for period in [&self.interval, &self.boot] {
            period.vcpu_wait.note(us);
            period.vcpu_wait_by_cpu[cpu.min(OTHER_CPU)].note(us);
        }
    }

    fn note_drain_wait(&self, us: u64) {
        self.interval.drain_wait.note(us);
        self.boot.drain_wait.note(us);
    }

    fn note_drain_handoff(&self) {
        self.interval.drain_handoffs.fetch_add(1, Ordering::Relaxed);
        self.boot.drain_handoffs.fetch_add(1, Ordering::Relaxed);
    }

    fn note_iosfc_call(&self, call: IosfcCall, cpu: usize, us: u64) {
        for period in [&self.interval, &self.boot] {
            period.iosfc_calls[call.index()][cpu.min(OTHER_CPU)].note(us);
        }
    }

    fn vcpu_waiting(&self) -> bool {
        self.vcpu_waiters.load(Ordering::Acquire) != 0
    }

    /// Register a vCPU about to block.
    fn open_wait(&self, cpu: usize) -> WaitTicket<'_> {
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        let since = Instant::now();
        self.open_waits.lock().push(OpenWait {
            ticket,
            since,
            drain_first: 0,
            others_first: 0,
            held_us: 0,
            longest_us: 0,
            longest: None,
        });
        self.vcpu_waiters.fetch_add(1, Ordering::AcqRel);
        WaitTicket {
            census: self,
            ticket,
            cpu,
            since,
            open: true,
        }
    }

    fn take_wait(&self, ticket: u64) -> Option<OpenWait> {
        let mut open = self.open_waits.lock();
        let at = open.iter().position(|w| w.ticket == ticket)?;
        Some(open.swap_remove(at))
    }

    /// Someone took the lock while vCPUs waited.
    fn note_acquired(&self, holder: Holder) {
        for wait in self.open_waits.lock().iter_mut() {
            match holder.is_drain() {
                true => wait.drain_first += 1,
                false => wait.others_first += 1,
            }
        }
    }

    /// A hold of `us` ended at `now` while vCPUs waited.
    fn note_released(&self, holder: Holder, us: u64, now: Instant) {
        for wait in self.open_waits.lock().iter_mut() {
            let waited = now.saturating_duration_since(wait.since).as_micros() as u64;
            wait.held_us += us.min(waited);
            if us > wait.longest_us {
                wait.longest_us = us;
                wait.longest = Some(holder);
            }
        }
    }

    /// The interval's line, if an interval has passed and something in it
    /// reached the threshold. The interval is cleared whenever it is claimed,
    /// so a quiet interval does not carry into the next.
    ///
    /// While a vCPU is blocked the line waits for it: the vCPU counts its wait
    /// once it has the lock, and its own release writes the line with the hold
    /// that kept it waiting. Not past a second interval, so vCPUs that keep
    /// arriving cannot hold the line back for good.
    pub fn report(&self, now_ms: u64) -> Option<String> {
        let last = self.last_report_ms.load(Ordering::Relaxed);
        let since = now_ms.saturating_sub(last);
        if since < REPORT_INTERVAL_MS
            || (self.vcpu_waiting() && since < 2 * REPORT_INTERVAL_MS)
            || self
                .last_report_ms
                .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
        {
            return None;
        }
        self.wait_lines.store(0, Ordering::Relaxed);
        let period = self.interval.summary(true);
        if period.worst_us() < REPORT_THRESHOLD_US {
            return None;
        }
        Some(format!(
            "device_lock_hold win_ms={} vcpu_wait_max_us={} vcpu_wait_p99_us={} vcpu_waits={} {} \
             (holds of the device lock this interval, per holder: count/max_us/p99_us; a vCPU \
             waiting here answers no interrupt)",
            since,
            period.vcpu_wait.max_us,
            period.vcpu_wait.p99_us,
            period.vcpu_wait.count,
            period.render()
        ))
    }

    /// The boot's line, and the boot's distributions cleared for the next.
    pub fn boot_summary(&self) -> Option<String> {
        let period = self.boot.summary(true);
        if period.is_empty() {
            return None;
        }
        Some(format!(
            "device_lock_hold_boot vcpu_wait_max_us={} vcpu_wait_p99_us={} vcpu_waits={} {}",
            period.vcpu_wait.max_us,
            period.vcpu_wait.p99_us,
            period.vcpu_wait.count,
            period.render()
        ))
    }
}

pub static CENSUS: std::sync::LazyLock<LockCensus> = std::sync::LazyLock::new(LockCensus::default);

// MARK: - Holding

/// The device lock, held by a named holder and timed until it is released.
pub struct Held<'a, T> {
    guard: Option<MutexGuard<'a, T>>,
    holder: Holder,
    since: Instant,
    census: &'a LockCensus,
    /// Written once the lock is released: a long vCPU wait's own line.
    after_release: Option<String>,
}

impl<'a, T> Held<'a, T> {
    pub fn new(guard: MutexGuard<'a, T>, holder: Holder) -> Self {
        Self::counted(guard, holder, &CENSUS)
    }

    fn counted(guard: MutexGuard<'a, T>, holder: Holder, census: &'a LockCensus) -> Self {
        if census.vcpu_waiting() {
            census.note_acquired(holder);
        }
        Self {
            guard: Some(guard),
            holder,
            since: Instant::now(),
            census,
            after_release: None,
        }
    }
}

/// Take the lock, or not, without waiting.
pub fn try_hold<T>(mutex: &Mutex<T>, holder: Holder) -> Option<Held<'_, T>> {
    mutex.try_lock().map(|guard| Held::new(guard, holder))
}

/// Take the lock from a vCPU thread: at once if it is free, and otherwise
/// registered as a waiter for as long as it blocks. Returns the record of the
/// wait, if it had to.
pub fn hold_for_vcpu<T>(mutex: &Mutex<T>, holder: Holder) -> (Held<'_, T>, Option<WaitRecord>) {
    vcpu_hold(mutex, holder, &CENSUS)
}

fn vcpu_hold<'a, T>(
    mutex: &'a Mutex<T>,
    holder: Holder,
    census: &'a LockCensus,
) -> (Held<'a, T>, Option<WaitRecord>) {
    if let Some(guard) = mutex.try_lock() {
        return (Held::counted(guard, holder, census), None);
    }
    let ticket = census.open_wait(this_cpu());
    let guard = mutex.lock();
    let record = ticket.close();
    let mut held = Held::counted(guard, holder, census);
    if record.wait_us >= REPORT_THRESHOLD_US
        && census.wait_lines.fetch_add(1, Ordering::Relaxed) < WAIT_LINES_PER_INTERVAL
    {
        held.after_release = Some(record.line(holder));
    }
    (held, Some(record))
}

/// Take the lock for the drain worker, or a reset, recording how long the
/// worker was kept from it.
pub fn hold_for_drain<T>(mutex: &Mutex<T>, holder: Holder) -> Held<'_, T> {
    drain_hold(mutex, holder, &CENSUS)
}

fn drain_hold<'a, T>(mutex: &'a Mutex<T>, holder: Holder, census: &'a LockCensus) -> Held<'a, T> {
    if let Some(guard) = mutex.try_lock() {
        return Held::counted(guard, holder, census);
    }
    let since = Instant::now();
    let guard = mutex.lock();
    if holder == Holder::Drain {
        census.note_drain_wait(since.elapsed().as_micros() as u64);
    }
    Held::counted(guard, holder, census)
}

impl<T> std::ops::Deref for Held<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.guard.as_ref().expect("held until drop")
    }
}

impl<T> std::ops::DerefMut for Held<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.guard.as_mut().expect("held until drop")
    }
}

impl<T> Drop for Held<'_, T> {
    fn drop(&mut self) {
        let Some(guard) = self.guard.take() else {
            return;
        };
        let now = Instant::now();
        let us = now.saturating_duration_since(self.since).as_micros() as u64;
        // Counted while still held, so a vCPU this hold kept waiting finds it
        // in its own record and on the same line as its wait.
        self.census.note_hold(self.holder, us);
        if self.census.vcpu_waiting() {
            self.census.note_released(self.holder, us, now);
            if self.holder.is_drain() {
                self.census.note_drain_handoff();
            }
        }
        MutexGuard::unlock_fair(guard);
        // Released before anything is written, so the writing is never part of
        // the hold it describes.
        if let Some(line) = self.after_release.take() {
            crate::observe::off(line);
        }
        if let Some(line) = self.census.report(crate::observe::elapsed_ms() as u64) {
            crate::observe::off(line);
        }
    }
}

/// Count one mapper register access by this vCPU, from `started` to now.
pub fn note_iosfc_call(call: IosfcCall, started: Instant) {
    let us = started.elapsed().as_micros() as u64;
    CENSUS.note_iosfc_call(call, this_cpu(), us);
}

/// Write the boot's summary; called at device reset and destroy.
pub fn report_boot() {
    if let Some(line) = CENSUS.boot_summary() {
        crate::observe::off(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    fn spin(d: Duration) {
        let start = Instant::now();
        while start.elapsed() < d {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn a_distribution_reports_its_max_and_a_bucketed_p99() {
        let d = Durations::default();
        for _ in 0..990 {
            d.note(100);
        }
        for _ in 0..10 {
            d.note(70_000);
        }
        let s = d.summary(false);
        assert_eq!(s.count, 1000);
        assert_eq!(s.max_us, 70_000);
        assert_eq!(s.p99_us, 128, "the 99th percentile is in 100's bucket");
        d.note(70_000);
        assert_eq!(d.summary(false).p99_us, 70_000, "capped at the max");
        assert_eq!(d.summary(true).count, 1001);
        assert_eq!(d.summary(false), Summary::default(), "taking clears");
    }

    /// Quiet intervals write nothing; an interval with a long hold writes one
    /// line, at most once per interval, naming the holder.
    #[test]
    fn a_line_is_written_only_for_a_long_hold_and_only_once_an_interval() {
        let c = LockCensus::default();
        c.note_hold(Holder::Drain, 1_000);
        assert_eq!(c.report(REPORT_INTERVAL_MS), None, "nothing reached 50 ms");

        c.note_hold(Holder::Drain, 120_000);
        c.note_vcpu_wait(3, 119_000);
        assert_eq!(
            c.report(REPORT_INTERVAL_MS + 5),
            None,
            "inside the interval"
        );
        let line = c
            .report(2 * REPORT_INTERVAL_MS + 1)
            .expect("a 120 ms hold is reported");
        assert!(line.contains("drain=1/120000/120000"), "{line}");
        assert!(line.contains("vcpu_wait_max_us=119000"), "{line}");
        assert!(line.contains("vcpu3_wait=1/119000/119000"), "{line}");
        assert_eq!(c.report(4 * REPORT_INTERVAL_MS), None, "cleared");

        let boot = c.boot_summary().expect("the boot saw holds");
        assert!(boot.contains("drain=2/120000/"), "{boot}");
        assert!(boot.contains("vcpu_waits=1"), "{boot}");
        assert_eq!(c.boot_summary(), None, "cleared for the next boot");
    }

    #[test]
    fn a_held_lock_is_timed_until_it_is_dropped() {
        let c = LockCensus::default();
        let m = Mutex::new(5u32);
        {
            let mut held = Held::counted(m.lock(), Holder::VcpuIosfcWrite, &c);
            *held += 1;
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(m.try_lock().is_some(), "released at drop");
        // The boot distribution: a report on drop may already have taken the
        // interval.
        let s = c.boot.holds[Holder::VcpuIosfcWrite.index()].summary(false);
        assert_eq!(s.count, 1);
        assert!(s.max_us >= 2_000, "{s:?}");
        assert_eq!(*m.lock(), 6);
    }

    /// Mapper register accesses are counted per vCPU and per kind, and a long
    /// one is reason enough for a line.
    #[test]
    fn iosfc_calls_are_counted_per_vcpu_and_kind() {
        let c = LockCensus::default();
        c.note_iosfc_call(IosfcCall::Read, 3, 2);
        assert_eq!(c.report(REPORT_INTERVAL_MS), None, "nothing reached 50 ms");

        c.note_iosfc_call(IosfcCall::Write, 3, 70_000);
        c.note_iosfc_call(IosfcCall::Read, 3, 2);
        c.note_iosfc_call(IosfcCall::Read, 40, 5);
        let line = c
            .report(2 * REPORT_INTERVAL_MS + 1)
            .expect("a 70 ms mapper write is reported");
        assert!(
            line.contains("vcpu3_iosfc_write_call=1/70000/70000"),
            "{line}"
        );
        assert!(line.contains("vcpu3_iosfc_read_call=1/2/2"), "{line}");
        assert!(
            line.contains("vcpu_other_iosfc_read_call=1/5/5"),
            "past the slots: {line}"
        );

        let boot = c.boot_summary().expect("the boot saw accesses");
        assert!(boot.contains("vcpu3_iosfc_read_call=2/2/2"), "{boot}");
    }

    #[test]
    fn a_vcpu_thread_name_gives_its_index() {
        assert_eq!(vcpu_index("CPU 3/KVM"), Some(3));
        assert_eq!(vcpu_index("CPU 12/TCG"), Some(12));
        assert_eq!(vcpu_index("worker"), None);
        assert_eq!(vcpu_index("CPU x/KVM"), None);
        assert_eq!(vcpu_index("CPU 3"), None);
        #[cfg(target_os = "linux")]
        {
            let named = |name: &str| {
                std::thread::Builder::new()
                    .name(name.to_owned())
                    .spawn(this_cpu)
                    .unwrap()
                    .join()
                    .unwrap()
            };
            assert_eq!(named("CPU 4/KVM"), 4);
            assert_eq!(named("CPU 40/KVM"), OTHER_CPU, "past the slots");
            assert_eq!(named("reims-worker"), OTHER_CPU);
        }
    }

    /// The bug the old line had: the worker's release wrote the interval
    /// before the vCPU it released had counted its wait, so a 7.8 s wait was
    /// reported a line after the 8.3 s hold that caused it.
    #[test]
    fn a_wait_and_the_hold_that_caused_it_are_on_one_line() {
        let c = LockCensus::default();
        c.note_hold(Holder::Drain, 8_000_000);
        let ticket = c.open_wait(3);
        assert_eq!(
            c.report(REPORT_INTERVAL_MS),
            None,
            "a vCPU is still waiting"
        );
        ticket.close();
        let line = c
            .report(REPORT_INTERVAL_MS + 1)
            .expect("the vCPU has counted its wait");
        assert!(line.contains("drain=1/8000000/"), "{line}");
        assert!(line.contains("vcpu_waits=1"), "{line}");
        assert!(line.contains("vcpu3_wait=1/"), "{line}");

        // A vCPU that never gets the lock holds the line back one more
        // interval at most.
        let _stuck = c.open_wait(3);
        c.note_hold(Holder::Drain, 60_000);
        assert_eq!(c.report(2 * REPORT_INTERVAL_MS + 2), None);
        assert!(c.report(3 * REPORT_INTERVAL_MS + 2).is_some());
    }

    #[test]
    fn a_waiter_that_unwinds_is_deregistered() {
        let c = LockCensus::default();
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ticket = c.open_wait(1);
            panic!("unwinding out of a wait");
        }));
        assert!(unwound.is_err());
        assert!(!c.vcpu_waiting());
        assert!(c.open_waits.lock().is_empty());

        c.open_wait(1).close();
        assert_eq!(c.vcpu_waiters.load(Ordering::Relaxed), 0, "closed once");
        assert!(c.open_waits.lock().is_empty());
    }

    /// One wait, end to end: the record names the hold that blocked it, and
    /// the line is kept for after the release.
    #[test]
    fn a_wait_records_who_held_the_lock() {
        let c = LockCensus::default();
        let m = Mutex::new(());
        let (taken, wait_for_it) = std::sync::mpsc::channel();
        let record = std::thread::scope(|s| {
            s.spawn(|| {
                let _held = drain_hold(&m, Holder::Drain, &c);
                taken.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(60));
            });
            wait_for_it.recv().unwrap();
            let (held, record) = vcpu_hold(&m, Holder::VcpuIosfcWrite, &c);
            let record = record.expect("the worker had it");
            assert!(held.after_release.is_some(), "{record:?}");
            record
        });
        assert_eq!(record.drain_first, 0, "{record:?}");
        assert_eq!(record.others_first, 0, "{record:?}");
        assert_eq!(record.longest, Some(Holder::Drain), "{record:?}");
        assert!(record.longest_us >= 60_000, "{record:?}");
        assert!(record.held_us <= record.wait_us, "{record:?}");
        assert!(record.wait_us >= 50_000, "{record:?}");
    }

    /// The drain worker goes from one hold straight into the next, as it goes
    /// from one tranche to the next. A vCPU that blocks gets the lock when the
    /// hold in progress ends, or the one after if the worker went round while
    /// the vCPU was still on its way to parking: never after the worker has
    /// taken it again and again.
    #[test]
    fn a_vcpu_is_not_overtaken_by_a_worker_that_comes_straight_back() {
        const HOLD: Duration = Duration::from_micros(100);
        const WAITS: usize = 200;
        let c = LockCensus::default();
        let m = Mutex::new(0u64);
        let stop = AtomicBool::new(false);
        let records = std::thread::scope(|s| {
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    let mut held = drain_hold(&m, Holder::Drain, &c);
                    *held += 1;
                    spin(HOLD);
                }
            });
            let vcpu = std::thread::Builder::new()
                .name("CPU 2/KVM".to_owned())
                .spawn_scoped(s, || {
                    let mut records = Vec::new();
                    while records.len() < WAITS {
                        std::thread::sleep(Duration::from_micros(150));
                        let (_held, record) = vcpu_hold(&m, Holder::VcpuIosfcWrite, &c);
                        records.extend(record);
                    }
                    records
                })
                .unwrap();
            let records = vcpu.join().unwrap();
            stop.store(true, Ordering::Relaxed);
            records
        });
        let overtaken: Vec<u64> = records.iter().map(|r| r.drain_first).collect();
        let longest = records.iter().map(|r| r.wait_us).max().unwrap();
        let once_at_most = overtaken.iter().filter(|&&n| n <= 1).count();
        eprintln!(
            "vcpu waits={} wait_max_us={longest} drain_first_max={} drain_first<=1: {once_at_most}",
            records.len(),
            overtaken.iter().max().unwrap()
        );
        assert!(
            once_at_most * 100 >= records.len() * 95,
            "the worker took the lock again ahead of a blocked vCPU: {overtaken:?}"
        );
        assert!(
            overtaken.iter().all(|&n| n <= 8),
            "a vCPU waited through a run of worker holds: {overtaken:?}"
        );
        assert!(
            longest < 50_000,
            "a vCPU waited {longest} us behind 100 us holds"
        );
        #[cfg(target_os = "linux")]
        assert!(records.iter().all(|r| r.cpu == 2));
        let handoffs = c.boot.drain_handoffs.load(Ordering::Relaxed);
        assert!(handoffs > 0, "the worker handed the lock over");
    }

    /// The other side: vCPUs that never stop taking the lock do not starve
    /// the worker, or each other.
    #[test]
    fn the_worker_and_each_vcpu_take_turns_when_vcpus_never_stop() {
        const HOLD: Duration = Duration::from_micros(100);
        const TRANCHES: u64 = 300;
        let c = LockCensus::default();
        let m = Mutex::new(0u64);
        let stop = AtomicBool::new(false);
        let per_vcpu = std::thread::scope(|s| {
            let vcpus: Vec<_> = (0..2)
                .map(|cpu| {
                    let (c, m, stop) = (&c, &m, &stop);
                    std::thread::Builder::new()
                        .name(format!("CPU {cpu}/KVM"))
                        .spawn_scoped(s, move || {
                            let mut records = Vec::new();
                            while !stop.load(Ordering::Relaxed) {
                                let (_held, record) = vcpu_hold(m, Holder::VcpuIosfcWrite, c);
                                records.extend(record);
                                spin(HOLD);
                            }
                            records
                        })
                        .unwrap()
                })
                .collect();
            for _ in 0..TRANCHES {
                let mut held = drain_hold(&m, Holder::Drain, &c);
                *held += 1;
                spin(HOLD);
            }
            stop.store(true, Ordering::Relaxed);
            vcpus
                .into_iter()
                .map(|v| v.join().unwrap())
                .collect::<Vec<_>>()
        });
        let worker = c.boot.drain_wait.summary(false);
        eprintln!("worker waits {worker}");
        assert!(
            worker.max_us < 50_000,
            "the worker was kept out: {worker:?}"
        );
        for records in &per_vcpu {
            assert!(!records.is_empty(), "each vCPU waited and got the lock");
            let longest = records.iter().map(|r| r.wait_us).max().unwrap();
            let overtaken = records.iter().map(|r| r.others_first).max().unwrap();
            eprintln!(
                "vcpu waits={} wait_max_us={longest} others_first_max={overtaken}",
                records.len()
            );
            assert!(longest < 50_000, "a vCPU was kept out for {longest} us");
        }
    }
}
