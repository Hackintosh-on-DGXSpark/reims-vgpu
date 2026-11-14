//! Graphics pipelines are built on their own thread, with no lock held.
//!
//! # Why
//!
//! `vkCreateGraphicsPipelines` is a shader compile inside the driver, and it
//! used to run inside the draw that first needed the pipeline — on the drain
//! thread, holding the device lock for the whole tranche and the engine lock for
//! the whole draw. On a macOS 27.2 boot one draw held both for 8.6 s compiling
//! the 368 062-word glass fragment shader (`pl_compile_us=8598624`), and NVIDIA
//! has been measured at 14–24 s for pipelines of that class. A vCPU entering the
//! IOSurface MMIO path holds QEMU's BQL and then waits for the device lock, so
//! for those seconds QEMU's main loop, interrupt delivery and the action BH
//! stopped with it, and the guest panicked on its own IPI timeouts.
//!
//! # Three steps
//!
//! 1. **Prepare**, on the drain thread under the engine lock:
//!    [`super::caches::ObjectCaches::resolve_graphics_pipeline`] answers a key
//!    already built, refuses one already refused, joins one already being
//!    built, and otherwise makes the device's checks, snapshots what the
//!    compile reads into a [`GraphicsCompileJob`], and returns a
//!    [`PipelineTicket`]. Nothing there waits on a driver compile.
//! 2. **Compile**, on this module's thread with no lock: validate and build
//!    both shader modules, build the pipeline, persist the pipeline cache.
//! 3. **Publish**, under the engine lock again:
//!    [`super::caches::ObjectCaches::publish_compiled`] files the result and
//!    answers the ticket.
//!
//! No creation of any kind happens under the lock, not even one the driver
//! would serve from its pipeline cache: `VK_EXT_pipeline_creation_cache_control`
//! only promises such a call does not compile, not that it does not block, and
//! the cache's internal lock is shared with this thread.
//!
//! # A draw waits by not running
//!
//! The draw that is told `Pending` returns before it records or acquires
//! anything. `runtime::exec` keeps its packet at the FIFO head with a
//! continuation, holds every other FIFO behind it the way a cold AIR
//! translation already does, and resumes at that draw once the ticket is
//! answered. Nothing holds a lock while it waits.
//!
//! # Each ticket is answered once
//!
//! A ticket's [`PipelineOutcome`] is written once and travels back into the
//! resumed draw. `Failed` is that draw's refusal even when the negative cache
//! declines to remember it — out-of-memory is about the instant, not the
//! request — so the waiting packet cannot loop into the same compile; a later,
//! independent draw may try again. `DeviceRebuilt` sends the draw to ask again
//! on the device that replaced the one it waited on.
//!
//! # Lifetimes
//!
//! A job holds the device through [`DeviceLife`], so a device loss in the
//! middle of a compile destroys nothing the driver is still reading; see that
//! type. The job's words are the runtime's own `Arc`s, not copies.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use ash::vk;

use super::caches::{
    create_graphics_pipeline, declare_image_capabilities, note_quarantined, refuse_invalid_module,
    GraphicsRecipe, PipelineKey, ShaderModuleSource, SpirvStorageCaps,
};
use super::context::DeviceLife;
use super::digest::Digest128;
use super::types::DrawError;
use super::vk_call::{VkCall, VkOp};

/// What a ticket's draw is told when its wait ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PipelineOutcome {
    /// Built and filed; the resumed draw finds it in the map.
    Ready,
    /// The compile refused, with the refusal the resumed draw reports.
    Failed(DrawError),
    /// The device the compile was for was replaced before its result could be
    /// used. The resumed draw asks again on the new one.
    DeviceRebuilt,
}

struct Flight {
    key: PipelineKey,
    submitted: Instant,
    /// The compiler is done with it and its result waits in the finished queue,
    /// or has already been published.
    compiled: AtomicBool,
    outcome: OnceLock<PipelineOutcome>,
}

/// One graphics pipeline being built, as the draws waiting for it hold it.
#[derive(Clone)]
pub struct PipelineTicket(Arc<Flight>);

impl std::fmt::Debug for PipelineTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineTicket")
            .field("vert", &self.0.key.vert)
            .field("frag", &self.0.key.frag)
            .field("compiled", &self.compiled())
            .field("outcome", &self.outcome())
            .finish()
    }
}

impl PipelineTicket {
    pub(crate) fn new(key: PipelineKey) -> Self {
        Self(Arc::new(Flight {
            key,
            submitted: Instant::now(),
            compiled: AtomicBool::new(false),
            outcome: OnceLock::new(),
        }))
    }

    pub(crate) fn key(&self) -> &PipelineKey {
        &self.0.key
    }

    /// Whether `other` is this same compile.
    pub(crate) fn is(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(crate) fn outcome(&self) -> Option<&PipelineOutcome> {
        self.0.outcome.get()
    }

    /// Answer the ticket. The first answer stands; a later one is ignored.
    pub(crate) fn settle(&self, outcome: PipelineOutcome) {
        let _ = self.0.outcome.set(outcome);
    }

    pub(crate) fn compiled(&self) -> bool {
        self.0.compiled.load(Ordering::Acquire)
    }

    /// The refusal this ticket ended in, when it was a compile of `key`.
    pub(crate) fn failure_for(&self, key: &PipelineKey) -> Option<DrawError> {
        if self.0.key != *key {
            return None;
        }
        match self.outcome() {
            Some(PipelineOutcome::Failed(error)) => Some(error.clone()),
            _ => None,
        }
    }
}

/// Everything one compile reads, owned or shared, so it needs no lock.
pub(crate) struct GraphicsCompileJob {
    pub(crate) ticket: PipelineTicket,
    pub(crate) life: Arc<DeviceLife>,
    pub(crate) recipe: GraphicsRecipe,
    pub(crate) vert: Arc<Vec<u32>>,
    pub(crate) frag: Arc<Vec<u32>>,
    pub(crate) caps: SpirvStorageCaps,
    /// The guest pipeline object the first draw named, held weakly so a
    /// compile does not keep a deleted object alive.
    pub(crate) pipeline_object: Option<(
        std::num::NonZeroU64,
        std::sync::Weak<super::types::PipelineObjectLife>,
    )>,
}

/// Why a compile produced no pipeline.
pub(crate) enum CompileFailure {
    /// The device was being replaced before the compile started, so it was not
    /// compiled.
    Stale,
    /// The compile refused. `module` names a shader module whose own creation
    /// or validation refused, so the draw path refuses that shader by digest
    /// from then on, as it did when it built modules itself.
    Refused {
        error: DrawError,
        module: Option<Digest128>,
    },
}

/// A finished compile waiting to be published.
pub(crate) struct CompiledPipeline {
    pub(crate) ticket: PipelineTicket,
    pub(crate) life: Arc<DeviceLife>,
    pub(crate) pipeline_object: Option<(
        std::num::NonZeroU64,
        std::sync::Weak<super::types::PipelineObjectLife>,
    )>,
    pub(crate) result: Result<vk::Pipeline, CompileFailure>,
}

struct Compiler {
    queue: Mutex<VecDeque<GraphicsCompileJob>>,
    wake: Condvar,
    finished: Mutex<Vec<CompiledPipeline>>,
    any_finished: AtomicBool,
    /// Jobs submitted and not yet finished.
    outstanding: AtomicU64,
}

static COMPILER: OnceLock<&'static Compiler> = OnceLock::new();

fn compiler() -> &'static Compiler {
    COMPILER.get_or_init(|| {
        let compiler: &'static Compiler = Box::leak(Box::new(Compiler {
            queue: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            finished: Mutex::new(Vec::new()),
            any_finished: AtomicBool::new(false),
            outstanding: AtomicU64::new(0),
        }));
        std::thread::Builder::new()
            .name("reims-pl-compile".into())
            .spawn(move || compiler.run())
            .expect("the pipeline compiler thread must start: no draw needing a new pipeline can run without it");
        compiler
    })
}

/// Whether graphics pipelines are built on the compiler thread, the default, or
/// inside the draw that first needs them under `REIMS_VGPU_PIPELINE_ASYNC=off`.
/// Read once; the boot line says which.
pub(crate) fn off_the_draw() -> bool {
    #[cfg(test)]
    if test_hooks::in_draw_forced() {
        return false;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let (state, value) = crate::env::read(crate::env::PIPELINE_ASYNC);
        let on = !matches!(state, crate::env::Switch::Off);
        crate::observe::off(format!(
            "pipeline_async on={on} switch={state:?} value={}",
            value.unwrap_or_else(|| "<unset>".into())
        ));
        on
    })
}

/// Build `job` here and now, on the calling thread and under whatever locks it
/// holds: the `REIMS_VGPU_PIPELINE_ASYNC=off` arm, which is how every graphics
/// pipeline was built before the compiler thread existed.
pub(crate) fn compile_in_draw(job: GraphicsCompileJob) -> CompiledPipeline {
    CENSUS.jobs.fetch_add(1, Ordering::Relaxed);
    CENSUS.in_draw.fetch_add(1, Ordering::Relaxed);
    let done = compile(job, super::driver_breadcrumb::DriverLane::Drain);
    done.ticket.0.compiled.store(true, Ordering::Release);
    done
}

/// Queue a compile. Never waits for one.
pub(crate) fn submit(job: GraphicsCompileJob) {
    let compiler = compiler();
    CENSUS.jobs.fetch_add(1, Ordering::Relaxed);
    compiler.outstanding.fetch_add(1, Ordering::AcqRel);
    compiler
        .queue
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push_back(job);
    compiler.wake.notify_one();
}

/// Wait until every submitted compile has finished.
///
/// For a reset that keeps the device but destroys the cache objects a compile
/// reads; nothing on a guest path calls it, because nothing on a guest path
/// may wait for a compile.
pub(crate) fn wait_idle() {
    let Some(compiler) = COMPILER.get() else {
        return;
    };
    while compiler.outstanding.load(Ordering::Acquire) != 0 {
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Every compile finished since the last call, for publishing.
pub(crate) fn take_finished() -> Vec<CompiledPipeline> {
    let Some(compiler) = COMPILER.get() else {
        return Vec::new();
    };
    if !compiler.any_finished.swap(false, Ordering::AcqRel) {
        return Vec::new();
    }
    std::mem::take(&mut *compiler.finished.lock().unwrap_or_else(|e| e.into_inner()))
}

impl Compiler {
    fn run(&'static self) {
        loop {
            let job = {
                let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if let Some(job) = queue.pop_front() {
                        break job;
                    }
                    queue = self.wake.wait(queue).unwrap_or_else(|e| e.into_inner());
                }
            };
            let ticket = job.ticket.clone();
            let done = compile(job, super::driver_breadcrumb::DriverLane::Compiler);
            self.finished
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(done);
            // After the push, so a ticket that reads `compiled` has its result
            // where the publisher will find it.
            ticket.0.compiled.store(true, Ordering::Release);
            self.any_finished.store(true, Ordering::Release);
            self.outstanding.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

fn compile(job: GraphicsCompileJob, lane: super::driver_breadcrumb::DriverLane) -> CompiledPipeline {
    let started = Instant::now();
    CENSUS.note_max(
        &CENSUS.queue_wait_max_ns,
        started.duration_since(job.ticket.0.submitted),
    );
    let result = if job.life.is_stale() {
        CENSUS.stale_skipped.fetch_add(1, Ordering::Relaxed);
        Err(CompileFailure::Stale)
    } else {
        #[cfg(test)]
        test_hooks::wait_at_gate();
        let built =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { build(&job, lane) }));
        built.unwrap_or_else(|panic| {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            let reason = super::reason::DrawReason::PipelineCompilerPanicked;
            crate::observe::Emit::decline("vk_pipeline_compiler", &reason)
                .field("vert_words", job.vert.len())
                .field("frag_words", job.frag.len())
                .field("panic", message.replace(char::is_whitespace, "_"))
                .fail();
            Err(CompileFailure::Refused {
                error: DrawError::Unsupported(reason),
                module: None,
            })
        })
    };
    let took = started.elapsed();
    CENSUS
        .compile_ns
        .fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
    CENSUS.note_max(&CENSUS.compile_max_ns, took);
    match &result {
        Ok(_) => CENSUS.built.fetch_add(1, Ordering::Relaxed),
        Err(CompileFailure::Refused { .. }) => CENSUS.refused.fetch_add(1, Ordering::Relaxed),
        Err(CompileFailure::Stale) => 0,
    };
    CompiledPipeline {
        ticket: job.ticket,
        life: job.life,
        pipeline_object: job.pipeline_object,
        result,
    }
}

unsafe fn build(
    job: &GraphicsCompileJob,
    lane: super::driver_breadcrumb::DriverLane,
) -> Result<vk::Pipeline, CompileFailure> {
    let vert = module(&job.life, &job.vert, job.caps, lane)?;
    let frag = module(&job.life, &job.frag, job.caps, lane)?;
    #[cfg(test)]
    if let Some(result) = test_hooks::forced_result() {
        return Err(CompileFailure::Refused {
            error: DrawError::VkCall(VkCall::new(VkOp::CachesCreateGraphicsPipelines, result)),
            module: None,
        });
    }
    let pipeline = create_graphics_pipeline(
        lane,
        &job.life.device,
        job.life.pipeline_cache,
        &job.recipe,
        ShaderModuleSource {
            module: vert,
            spirv: &job.vert,
        },
        ShaderModuleSource {
            module: frag,
            spirv: &job.frag,
        },
    )
    .map_err(|error| CompileFailure::Refused {
        error,
        module: None,
    })?;
    // A fresh compile grew the VkPipelineCache; persist it so the next boot
    // warm-starts. Off every lock here, which is where its serialize belongs.
    job.life.persist_pipeline_cache();
    Ok(pipeline)
}

/// The module `words` builds on `life`'s device, made once per digest.
///
/// Kept with the device rather than in the engine's caches because only this
/// thread builds graphics modules, and they die with the device they were made
/// on. A refusal is kept beside them so a second pipeline using the same bad
/// shader does not validate it again — except out-of-memory, which is about
/// the instant, as everywhere else in this engine.
unsafe fn module(
    life: &DeviceLife,
    words: &[u32],
    caps: SpirvStorageCaps,
    lane: super::driver_breadcrumb::DriverLane,
) -> Result<vk::ShaderModule, CompileFailure> {
    let declared =
        declare_image_capabilities(words, caps).map_err(|error| CompileFailure::Refused {
            error,
            module: Some(Digest128::of_u32_words(words)),
        })?;
    let digest = Digest128::of_u32_words(&declared);
    let known = life
        .modules
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&digest)
        .cloned();
    let refused = |error| CompileFailure::Refused {
        error,
        module: Some(digest),
    };
    if let Some(known) = known {
        return known.map_err(refused);
    }
    let started = Instant::now();
    let built = (|| {
        refuse_invalid_module(&declared, digest)?;
        let breadcrumb = super::driver_breadcrumb::DriverBreadcrumb::arm(
            lane,
            "create_shader_module",
            &[("module", &declared)],
        )
        .map_err(|hit| note_quarantined("create_shader_module", &hit))?;
        let created = life
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&declared), None);
        breadcrumb.disarm();
        created.map_err(|e| DrawError::VkCall(VkCall::new(VkOp::CachesCreateShaderModule, e)))
    })();
    CENSUS
        .module_ns
        .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if !matches!(&built, Err(error) if error.out_of_memory()) {
        life.modules
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(digest, built.clone());
    }
    built.map_err(refused)
}

// MARK: - Census

/// What the compiler and the draws around it did over one census window.
struct Census {
    jobs: AtomicU64,
    /// Of `jobs`, the ones built inside their draw (`REIMS_VGPU_PIPELINE_ASYNC=off`).
    in_draw: AtomicU64,
    built: AtomicU64,
    refused: AtomicU64,
    stale_skipped: AtomicU64,
    compile_ns: AtomicU64,
    compile_max_ns: AtomicU64,
    module_ns: AtomicU64,
    queue_wait_max_ns: AtomicU64,
    published_stale: AtomicU64,
    failures_delivered: AtomicU64,
    compute_compiles: AtomicU64,
    compute_compile_ns: AtomicU64,
    compute_compile_max_ns: AtomicU64,
}

impl Census {
    fn note_max(&self, slot: &AtomicU64, took: Duration) {
        slot.fetch_max(took.as_nanos() as u64, Ordering::Relaxed);
    }
}

static CENSUS: Census = Census {
    jobs: AtomicU64::new(0),
    in_draw: AtomicU64::new(0),
    built: AtomicU64::new(0),
    refused: AtomicU64::new(0),
    stale_skipped: AtomicU64::new(0),
    compile_ns: AtomicU64::new(0),
    compile_max_ns: AtomicU64::new(0),
    module_ns: AtomicU64::new(0),
    queue_wait_max_ns: AtomicU64::new(0),
    published_stale: AtomicU64::new(0),
    failures_delivered: AtomicU64::new(0),
    compute_compiles: AtomicU64::new(0),
    compute_compile_ns: AtomicU64::new(0),
    compute_compile_max_ns: AtomicU64::new(0),
};

/// A finished result was dropped because its device had been replaced.
pub(crate) fn note_published_stale() {
    CENSUS.published_stale.fetch_add(1, Ordering::Relaxed);
}

/// A resumed draw was handed the refusal its wait ended in.
pub(crate) fn note_failure_delivered() {
    CENSUS.failures_delivered.fetch_add(1, Ordering::Relaxed);
}

/// A compute pipeline was compiled on the drain thread, under both locks.
pub(crate) fn note_compute_compile(took: Duration) {
    CENSUS.compute_compiles.fetch_add(1, Ordering::Relaxed);
    CENSUS
        .compute_compile_ns
        .fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
    CENSUS.note_max(&CENSUS.compute_compile_max_ns, took);
}

/// The `pipeline_compile` line for the window just ended, or `None` when
/// nothing was compiled or published.
///
/// `compile_us` is graphics compile time spent on the compiler thread with no
/// lock held, and `compile_max_us` the longest one; `module_us` is the part of
/// it spent validating and building shader modules. `cp_compile_us` is the
/// compute compile time still spent on the drain thread under both locks.
pub(crate) fn take_census(win_ms: u64) -> Option<String> {
    let take = |slot: &AtomicU64| slot.swap(0, Ordering::Relaxed);
    let us = |ns: u64| ns / 1000;
    let jobs = take(&CENSUS.jobs);
    let in_draw = take(&CENSUS.in_draw);
    let built = take(&CENSUS.built);
    let refused = take(&CENSUS.refused);
    let stale = take(&CENSUS.stale_skipped);
    let compile_ns = take(&CENSUS.compile_ns);
    let compile_max_ns = take(&CENSUS.compile_max_ns);
    let module_ns = take(&CENSUS.module_ns);
    let queue_wait_max_ns = take(&CENSUS.queue_wait_max_ns);
    let published_stale = take(&CENSUS.published_stale);
    let delivered = take(&CENSUS.failures_delivered);
    let cp = take(&CENSUS.compute_compiles);
    let cp_ns = take(&CENSUS.compute_compile_ns);
    let cp_max_ns = take(&CENSUS.compute_compile_max_ns);
    if jobs + built + refused + stale + published_stale + delivered + cp == 0 {
        return None;
    }
    Some(format!(
        "pipeline_compile win_ms={win_ms} jobs={jobs} in_draw={in_draw} built={built} refused={refused} \
         stale_skipped={stale} compile_us={} compile_max_us={} module_us={} \
         queue_wait_max_us={} published_stale={published_stale} \
         failures_delivered={delivered} cp_compiles={cp} cp_compile_us={} cp_compile_max_us={}",
        us(compile_ns),
        us(compile_max_ns),
        us(module_ns),
        us(queue_wait_max_ns),
        us(cp_ns),
        us(cp_max_ns),
    ))
}

#[cfg(test)]
mod tests;

// MARK: - Test hooks

#[cfg(test)]
pub(crate) mod test_hooks {
    use std::sync::{Condvar, Mutex};
    use std::time::{Duration, Instant};

    struct Gate {
        closed: bool,
        waiting: usize,
    }

    static GATE: Mutex<Gate> = Mutex::new(Gate {
        closed: false,
        waiting: 0,
    });
    static GATE_CHANGED: Condvar = Condvar::new();
    static FORCED: Mutex<Option<ash::vk::Result>> = Mutex::new(None);

    /// Stops the compiler at the start of every compile until dropped — a
    /// compile as slow as the test wants, without a shader that takes that long.
    pub(crate) struct HeldCompiler(());

    impl HeldCompiler {
        pub(crate) fn hold() -> Self {
            GATE.lock().unwrap().closed = true;
            Self(())
        }

        /// Wait until `n` compiles are stopped at the gate.
        pub(crate) fn wait_for_waiting(&self, n: usize, limit: Duration) -> bool {
            let deadline = Instant::now() + limit;
            let mut gate = GATE.lock().unwrap();
            while gate.waiting < n {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return false;
                }
                gate = GATE_CHANGED.wait_timeout(gate, left).unwrap().0;
            }
            true
        }
    }

    impl Drop for HeldCompiler {
        fn drop(&mut self) {
            GATE.lock().unwrap().closed = false;
            GATE_CHANGED.notify_all();
        }
    }

    pub(super) fn wait_at_gate() {
        let mut gate = GATE.lock().unwrap();
        if !gate.closed {
            return;
        }
        gate.waiting += 1;
        GATE_CHANGED.notify_all();
        while gate.closed {
            gate = GATE_CHANGED.wait(gate).unwrap();
        }
        gate.waiting -= 1;
        GATE_CHANGED.notify_all();
    }

    /// Makes every pipeline create report `result` instead of calling the
    /// driver, until dropped.
    pub(crate) struct ForcedResult(());

    impl ForcedResult {
        pub(crate) fn new(result: ash::vk::Result) -> Self {
            *FORCED.lock().unwrap() = Some(result);
            Self(())
        }
    }

    impl Drop for ForcedResult {
        fn drop(&mut self) {
            *FORCED.lock().unwrap() = None;
        }
    }

    pub(super) fn forced_result() -> Option<ash::vk::Result> {
        *FORCED.lock().unwrap()
    }

    static IN_DRAW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// Builds every pipeline inside its draw, as `REIMS_VGPU_PIPELINE_ASYNC=off`
    /// does, until dropped.
    pub(crate) struct InDraw(());

    impl InDraw {
        pub(crate) fn force() -> Self {
            IN_DRAW.store(true, std::sync::atomic::Ordering::Release);
            Self(())
        }
    }

    impl Drop for InDraw {
        fn drop(&mut self) {
            IN_DRAW.store(false, std::sync::atomic::Ordering::Release);
        }
    }

    pub(super) fn in_draw_forced() -> bool {
        IN_DRAW.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Wait until `ticket`'s compile has finished on the compiler thread.
    pub(crate) fn wait_compiled(ticket: &super::PipelineTicket, limit: Duration) -> bool {
        let deadline = Instant::now() + limit;
        while !ticket.compiled() {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }
}
