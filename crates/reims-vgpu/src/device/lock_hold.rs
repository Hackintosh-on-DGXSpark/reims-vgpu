//! How long each holder keeps the device lock, and how long a vCPU waited for
//! it.
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
//! # Cost
//!
//! Two clock reads and four relaxed atomic adds per acquisition. A line is
//! written at most once per [`REPORT_INTERVAL_MS`], and only when some hold or
//! wait in that interval reached [`REPORT_THRESHOLD_US`]; a boot summary is
//! written at device reset.

use std::sync::atomic::{AtomicU64, Ordering};

/// Who took the device lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holder {
    /// The drain worker's tranche (`device_drain`).
    Drain,
    /// `device_reset`.
    Reset,
    /// A vCPU reading an IOSurface mapper register.
    VcpuIosfcRead,
    /// A vCPU writing one — the mapper ring publish, which drains synchronously.
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

const HOLDERS: usize = 10;

impl Holder {
    const ALL: [Holder; HOLDERS] = [
        Holder::Drain,
        Holder::Reset,
        Holder::VcpuIosfcRead,
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
            Self::VcpuIosfcRead => "vcpu_iosfc_read",
            Self::VcpuIosfcWrite => "vcpu_iosfc_write",
            Self::VcpuGfxRead => "vcpu_gfx_read",
            Self::VcpuGfxWrite => "vcpu_gfx_write",
            Self::ScanoutCopy => "scanout_copy",
            Self::Poll => "poll",
            Self::PopAction => "pop_action",
            Self::Display => "display",
        }
    }
}

/// A report is written at most this often.
pub const REPORT_INTERVAL_MS: u64 = 10_000;
/// ...and only when a hold or a wait in the interval reached this.
pub const REPORT_THRESHOLD_US: u64 = 50_000;

/// Log2 buckets of microseconds: bucket `i` holds `[2^i, 2^(i+1))`, bucket 0
/// also holds 0. Thirty-two reach past an hour.
const BUCKETS: usize = 32;

fn bucket(us: u64) -> usize {
    (63 - us.max(1).leading_zeros() as usize).min(BUCKETS - 1)
}

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

/// Holds per holder, and vCPU waits, for the current report interval and for
/// the boot.
#[derive(Debug)]
pub struct LockCensus {
    interval: [Durations; HOLDERS],
    boot: [Durations; HOLDERS],
    vcpu_wait_interval: Durations,
    vcpu_wait_boot: Durations,
    last_report_ms: AtomicU64,
}

impl Default for LockCensus {
    fn default() -> Self {
        Self {
            interval: std::array::from_fn(|_| Durations::default()),
            boot: std::array::from_fn(|_| Durations::default()),
            vcpu_wait_interval: Durations::default(),
            vcpu_wait_boot: Durations::default(),
            last_report_ms: AtomicU64::new(0),
        }
    }
}

impl LockCensus {
    pub fn note_hold(&self, holder: Holder, us: u64) {
        self.interval[holder.index()].note(us);
        self.boot[holder.index()].note(us);
    }

    pub fn note_vcpu_wait(&self, us: u64) {
        self.vcpu_wait_interval.note(us);
        self.vcpu_wait_boot.note(us);
    }

    /// The interval's line, if an interval has passed and something in it
    /// reached the threshold. The interval is cleared whenever it is claimed,
    /// so a quiet interval does not carry into the next.
    pub fn report(&self, now_ms: u64) -> Option<String> {
        let last = self.last_report_ms.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) < REPORT_INTERVAL_MS
            || self
                .last_report_ms
                .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
        {
            return None;
        }
        let holds: Vec<(Holder, Summary)> = Holder::ALL
            .iter()
            .map(|&h| (h, self.interval[h.index()].summary(true)))
            .collect();
        let wait = self.vcpu_wait_interval.summary(true);
        let worst = holds.iter().map(|(_, s)| s.max_us).max().unwrap_or(0);
        if worst.max(wait.max_us) < REPORT_THRESHOLD_US {
            return None;
        }
        Some(format!(
            "device_lock_hold win_ms={} vcpu_wait_max_us={} vcpu_wait_p99_us={} vcpu_waits={} {} \
             (holds of the device lock this interval, per holder: count/max_us/p99_us; a vCPU \
             waiting here answers no interrupt)",
            now_ms.saturating_sub(last),
            wait.max_us,
            wait.p99_us,
            wait.count,
            render(&holds)
        ))
    }

    /// The boot's line, and the boot's distributions cleared for the next.
    pub fn boot_summary(&self) -> Option<String> {
        let holds: Vec<(Holder, Summary)> = Holder::ALL
            .iter()
            .map(|&h| (h, self.boot[h.index()].summary(true)))
            .collect();
        let wait = self.vcpu_wait_boot.summary(true);
        if wait.count == 0 && holds.iter().all(|(_, s)| s.count == 0) {
            return None;
        }
        Some(format!(
            "device_lock_hold_boot vcpu_wait_max_us={} vcpu_wait_p99_us={} vcpu_waits={} {}",
            wait.max_us,
            wait.p99_us,
            wait.count,
            render(&holds)
        ))
    }
}

fn render(holds: &[(Holder, Summary)]) -> String {
    holds
        .iter()
        .filter(|(_, s)| s.count != 0)
        .map(|(h, s)| format!("{}={}/{}/{}", h.name(), s.count, s.max_us, s.p99_us))
        .collect::<Vec<_>>()
        .join(" ")
}

pub static CENSUS: std::sync::LazyLock<LockCensus> = std::sync::LazyLock::new(LockCensus::default);

/// The device lock, held by a named holder and timed until it is released.
pub struct Held<'a, T> {
    guard: Option<parking_lot::MutexGuard<'a, T>>,
    holder: Holder,
    since: std::time::Instant,
    census: &'a LockCensus,
}

impl<'a, T> Held<'a, T> {
    pub fn new(guard: parking_lot::MutexGuard<'a, T>, holder: Holder) -> Self {
        Self::counted(guard, holder, &CENSUS)
    }

    fn counted(
        guard: parking_lot::MutexGuard<'a, T>,
        holder: Holder,
        census: &'a LockCensus,
    ) -> Self {
        Self {
            guard: Some(guard),
            holder,
            since: std::time::Instant::now(),
            census,
        }
    }
}

/// Take the lock, or not, without waiting.
pub fn try_hold<T>(mutex: &parking_lot::Mutex<T>, holder: Holder) -> Option<Held<'_, T>> {
    mutex.try_lock().map(|guard| Held::new(guard, holder))
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
        let us = self.since.elapsed().as_micros() as u64;
        // Released before anything is recorded or written, so the report is
        // never part of the hold it describes.
        drop(self.guard.take());
        self.census.note_hold(self.holder, us);
        if let Some(line) = self.census.report(crate::observe::elapsed_ms() as u64) {
            crate::observe::off(line);
        }
    }
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
        c.note_vcpu_wait(119_000);
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
        assert_eq!(c.report(4 * REPORT_INTERVAL_MS), None, "cleared");

        let boot = c.boot_summary().expect("the boot saw holds");
        assert!(boot.contains("drain=2/120000/"), "{boot}");
        assert!(boot.contains("vcpu_waits=1"), "{boot}");
        assert_eq!(c.boot_summary(), None, "cleared for the next boot");
    }

    #[test]
    fn a_held_lock_is_timed_until_it_is_dropped() {
        let c = LockCensus::default();
        let m = parking_lot::Mutex::new(5u32);
        {
            let mut held = Held::counted(m.lock(), Holder::VcpuIosfcWrite, &c);
            *held += 1;
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(m.try_lock().is_some(), "released at drop");
        // The boot distribution: a report on drop may already have taken the
        // interval.
        let s = c.boot[Holder::VcpuIosfcWrite.index()].summary(false);
        assert_eq!(s.count, 1);
        assert!(s.max_us >= 2_000, "{s:?}");
        assert_eq!(*m.lock(), 6);
    }
}
