//! The pipeline compiler against a real device: a cold pipeline is built off
//! the engine lock, asks for one being built join it, a device rebuilt
//! mid-compile destroys nothing the compile is using, and a failed compile is
//! the waiting draw's answer.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ash::vk;

use super::test_hooks::{wait_compiled, ForcedResult, HeldCompiler};
use super::{PipelineOutcome, PipelineTicket, CENSUS};
use crate::backend::vulkan::engine::{
    execute_draw_request, pipeline_ticket_settled, DrawAnswer, DrawError, DrawRequest,
    PrimitiveTopology,
};

/// Written for these tests, not taken from a guest: this GLSL, compiled by
/// `glslangValidator -V --target-env vulkan1.2`.
///
/// ```glsl
/// #version 450
/// void main() {
///     vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
///     gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
/// }
/// ```
const TRI_VERT: [u32; 284] = [
    0x07230203, 0x00010500, 0x0008000b, 0x0000002b, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000c, 0x0000001d, 0x00030003,
    0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000, 0x00030005, 0x00000009,
    0x00000070, 0x00060005, 0x0000000c, 0x565f6c67, 0x65747265, 0x646e4978, 0x00007865, 0x00060005,
    0x0000001b, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x0000001b, 0x00000000,
    0x505f6c67, 0x7469736f, 0x006e6f69, 0x00070006, 0x0000001b, 0x00000001, 0x505f6c67, 0x746e696f,
    0x657a6953, 0x00000000, 0x00070006, 0x0000001b, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369,
    0x0065636e, 0x00070006, 0x0000001b, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369, 0x0065636e,
    0x00030005, 0x0000001d, 0x00000000, 0x00040047, 0x0000000c, 0x0000000b, 0x0000002a, 0x00030047,
    0x0000001b, 0x00000002, 0x00050048, 0x0000001b, 0x00000000, 0x0000000b, 0x00000000, 0x00050048,
    0x0000001b, 0x00000001, 0x0000000b, 0x00000001, 0x00050048, 0x0000001b, 0x00000002, 0x0000000b,
    0x00000003, 0x00050048, 0x0000001b, 0x00000003, 0x0000000b, 0x00000004, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007,
    0x00000006, 0x00000002, 0x00040020, 0x00000008, 0x00000007, 0x00000007, 0x00040015, 0x0000000a,
    0x00000020, 0x00000001, 0x00040020, 0x0000000b, 0x00000001, 0x0000000a, 0x0004003b, 0x0000000b,
    0x0000000c, 0x00000001, 0x0004002b, 0x0000000a, 0x0000000e, 0x00000001, 0x0004002b, 0x0000000a,
    0x00000010, 0x00000002, 0x00040017, 0x00000017, 0x00000006, 0x00000004, 0x00040015, 0x00000018,
    0x00000020, 0x00000000, 0x0004002b, 0x00000018, 0x00000019, 0x00000001, 0x0004001c, 0x0000001a,
    0x00000006, 0x00000019, 0x0006001e, 0x0000001b, 0x00000017, 0x00000006, 0x0000001a, 0x0000001a,
    0x00040020, 0x0000001c, 0x00000003, 0x0000001b, 0x0004003b, 0x0000001c, 0x0000001d, 0x00000003,
    0x0004002b, 0x0000000a, 0x0000001e, 0x00000000, 0x0004002b, 0x00000006, 0x00000020, 0x40000000,
    0x0004002b, 0x00000006, 0x00000022, 0x3f800000, 0x0004002b, 0x00000006, 0x00000025, 0x00000000,
    0x00040020, 0x00000029, 0x00000003, 0x00000017, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000008, 0x00000009, 0x00000007, 0x0004003d,
    0x0000000a, 0x0000000d, 0x0000000c, 0x000500c4, 0x0000000a, 0x0000000f, 0x0000000d, 0x0000000e,
    0x000500c7, 0x0000000a, 0x00000011, 0x0000000f, 0x00000010, 0x0004006f, 0x00000006, 0x00000012,
    0x00000011, 0x0004003d, 0x0000000a, 0x00000013, 0x0000000c, 0x000500c7, 0x0000000a, 0x00000014,
    0x00000013, 0x00000010, 0x0004006f, 0x00000006, 0x00000015, 0x00000014, 0x00050050, 0x00000007,
    0x00000016, 0x00000012, 0x00000015, 0x0003003e, 0x00000009, 0x00000016, 0x0004003d, 0x00000007,
    0x0000001f, 0x00000009, 0x0005008e, 0x00000007, 0x00000021, 0x0000001f, 0x00000020, 0x00050050,
    0x00000007, 0x00000023, 0x00000022, 0x00000022, 0x00050083, 0x00000007, 0x00000024, 0x00000021,
    0x00000023, 0x00050051, 0x00000006, 0x00000026, 0x00000024, 0x00000000, 0x00050051, 0x00000006,
    0x00000027, 0x00000024, 0x00000001, 0x00070050, 0x00000017, 0x00000028, 0x00000026, 0x00000027,
    0x00000025, 0x00000022, 0x00050041, 0x00000029, 0x0000002a, 0x0000001d, 0x0000001e, 0x0003003e,
    0x0000002a, 0x00000028, 0x000100fd, 0x00010038,
];

/// Written for these tests, not taken from a guest: this GLSL, compiled the
/// same way.
///
/// ```glsl
/// #version 450
/// layout(location = 0) out vec4 color;
/// void main() { color = vec4(1.0, 0.5, 0.25, 1.0); }
/// ```
const TRI_FRAG: [u32; 91] = [
    0x07230203, 0x00010500, 0x0008000b, 0x0000000e, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
    0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
    0x00040005, 0x00000009, 0x6f6c6f63, 0x00000072, 0x00040047, 0x00000009, 0x0000001e, 0x00000000,
    0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006, 0x00000020,
    0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003, 0x00000007,
    0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x0004002b, 0x00000006, 0x0000000a, 0x3f800000,
    0x0004002b, 0x00000006, 0x0000000b, 0x3f000000, 0x0004002b, 0x00000006, 0x0000000c, 0x3e800000,
    0x0007002c, 0x00000007, 0x0000000d, 0x0000000a, 0x0000000b, 0x0000000c, 0x0000000a, 0x00050036,
    0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x0003003e, 0x00000009,
    0x0000000d, 0x000100fd, 0x00010038,
];

/// `TRI_FRAG` with its blue channel nudged by `seed`, so each test asks for a
/// pipeline no other test has built.
fn frag_variant(seed: u32) -> Arc<Vec<u32>> {
    let quarter = 0.25f32.to_bits();
    let mut words = TRI_FRAG.to_vec();
    let at: Vec<usize> = (0..words.len()).filter(|&i| words[i] == quarter).collect();
    assert_eq!(at.len(), 1, "the fragment shader's one 0.25 constant");
    words[at[0]] = (0.25f32 + seed as f32 / 4096.0).to_bits();
    Arc::new(words)
}

fn request(frag: &Arc<Vec<u32>>) -> DrawRequest {
    DrawRequest {
        vert_spirv: Arc::new(TRI_VERT.to_vec()),
        frag_spirv: Arc::clone(frag),
        width: 8,
        height: 8,
        vertex_count: 3,
        instance_count: Some(1),
        primitive_topology: PrimitiveTopology::Triangle,
        ..Default::default()
    }
}

enum Asked {
    Pending(PipelineTicket),
    Drawn(Vec<u8>),
    NoDevice,
}

fn ask(req: &DrawRequest) -> Asked {
    match execute_draw_request(req) {
        Ok(DrawAnswer::PipelinePending(ticket)) => Asked::Pending(ticket),
        Ok(DrawAnswer::Drawn(out)) => Asked::Drawn(out.pixels),
        Err(DrawError::Init(e)) => {
            eprintln!("SKIP pipeline compiler: no device ({e})");
            Asked::NoDevice
        }
        Err(e) => panic!("draw refused: {e}"),
    }
}

fn settled(ticket: &PipelineTicket) -> PipelineOutcome {
    assert!(
        wait_compiled(ticket, Duration::from_secs(60)),
        "the compile never finished"
    );
    assert!(
        pipeline_ticket_settled(ticket),
        "a finished compile is answered"
    );
    ticket.outcome().cloned().expect("answered")
}

fn assert_drawn_color(pixels: &[u8], seed: u32) {
    let blue = ((0.25f32 + seed as f32 / 4096.0) * 255.0).round() as i32;
    let center = (4 * 8 + 4) * 4;
    let px = &pixels[center..center + 4];
    let near = |got: u8, want: i32| (i32::from(got) - want).abs() <= 1;
    assert!(
        near(px[0], 255) && near(px[1], 128) && near(px[2], blue) && near(px[3], 255),
        "drew {px:?}"
    );
}

/// A cold pipeline is not built inside the draw, and the engine lock is free
/// for as long as its compile runs.
///
/// The compile is stopped at the compiler's gate for 300 ms — a compile as
/// slow as the test likes. The draw returns `PipelinePending` without waiting
/// for it; while it is stopped, every attempt to take the engine lock gets it
/// at once, and the same draw asked again joins the same compile instead of
/// starting a second. Released, the compile is published and the draw draws.
#[test]
fn a_cold_pipeline_compiles_off_the_engine_lock_and_a_second_ask_joins_it() {
    crate::observe::redirect_logs_for_tests();
    let seed = 1;
    let frag = frag_variant(seed);
    let req = request(&frag);
    // Bring the device and its pools up first, with a pipeline of its own, so
    // the timed ask below is the pipeline lookup and nothing else.
    match super::super::execute_draw_request_blocking(&request(&frag_variant(100 + seed))) {
        Ok(_) => {}
        Err(DrawError::Init(e)) => {
            eprintln!("SKIP pipeline compiler: no device ({e})");
            return;
        }
        Err(e) => panic!("warm-up draw refused: {e}"),
    }
    let held = HeldCompiler::hold();
    let jobs_before = CENSUS.jobs.load(Ordering::Relaxed);
    let asked = Instant::now();
    let ticket = match ask(&req) {
        Asked::Pending(ticket) => ticket,
        Asked::Drawn(_) => panic!("a cold pipeline was built inside the draw"),
        Asked::NoDevice => return,
    };
    let ask_took = asked.elapsed();
    assert!(
        held.wait_for_waiting(1, Duration::from_secs(10)),
        "the compile reached the compiler thread"
    );

    let mut worst_lock = Duration::ZERO;
    let window = Instant::now();
    while window.elapsed() < Duration::from_millis(300) {
        let asked = Instant::now();
        let guard = super::super::ENGINE.try_lock();
        assert!(
            guard.is_some(),
            "the engine lock was held while a compile ran"
        );
        drop(guard);
        worst_lock = worst_lock.max(asked.elapsed());
        std::thread::sleep(Duration::from_millis(5));
    }
    let joins_before = super::super::lock_engine()
        .counters
        .pipeline_flight_joins
        .load(Ordering::Relaxed);
    let again = Instant::now();
    match ask(&req) {
        Asked::Pending(second) => assert!(second.is(&ticket), "a second compile was started"),
        _ => panic!("the pipeline was answered while its compile was stopped"),
    }
    let join_took = again.elapsed();
    assert_eq!(
        super::super::lock_engine()
            .counters
            .pipeline_flight_joins
            .load(Ordering::Relaxed),
        joins_before + 1
    );
    assert_eq!(
        CENSUS.jobs.load(Ordering::Relaxed),
        jobs_before + 1,
        "one compile"
    );
    assert_eq!(
        ticket.outcome(),
        None,
        "not answered before the compile finished"
    );

    drop(held);
    let compile_started = Instant::now();
    assert_eq!(settled(&ticket), PipelineOutcome::Ready);
    let compile_took = compile_started.elapsed();
    match ask(&req) {
        Asked::Drawn(pixels) => assert_drawn_color(&pixels, seed),
        _ => panic!("a published pipeline was not drawn with"),
    }
    eprintln!(
        "PASYNC_MEASURE engine cold_ask_us={} join_ask_us={} engine_try_lock_max_us={} \
         compile_after_release_us={}",
        ask_took.as_micros(),
        join_took.as_micros(),
        worst_lock.as_micros(),
        compile_took.as_micros()
    );
    assert!(
        ask_took < Duration::from_millis(50),
        "the cold ask held the engine lock for {ask_took:?}"
    );
}

/// A device rebuilt while a compile is inside the driver answers the waiting
/// draw at once, keeps the old device alive until the compile returns, and
/// then destroys the result and the device together.
#[test]
fn a_device_rebuilt_mid_compile_answers_the_wait_and_keeps_the_device_until_it_returns() {
    crate::observe::redirect_logs_for_tests();
    let seed = 2;
    let frag = frag_variant(seed);
    let req = request(&frag);
    let held = HeldCompiler::hold();
    let ticket = match ask(&req) {
        Asked::Pending(ticket) => ticket,
        Asked::Drawn(_) => panic!("a cold pipeline was built inside the draw"),
        Asked::NoDevice => return,
    };
    assert!(held.wait_for_waiting(1, Duration::from_secs(10)));
    let old_life = Arc::downgrade(
        &super::super::lock_engine()
            .owner
            .ctx
            .as_ref()
            .expect("a device")
            .life,
    );

    super::super::test_poison_and_flush();
    assert_eq!(
        ticket.outcome(),
        Some(&PipelineOutcome::DeviceRebuilt),
        "the waiting draw is answered when its device goes, not when the compile returns"
    );
    // The next draw brings the new device up and asks again on it.
    let renewed = match ask(&req) {
        Asked::Pending(renewed) => renewed,
        _ => panic!("the draw on the rebuilt device found a pipeline it never compiled"),
    };
    assert!(!renewed.is(&ticket));
    assert!(
        old_life.upgrade().is_some(),
        "the old device was destroyed under a compile still using it"
    );

    drop(held);
    assert!(wait_compiled(&ticket, Duration::from_secs(60)));
    assert_eq!(settled(&renewed), PipelineOutcome::Ready);
    assert!(
        old_life.upgrade().is_none(),
        "the old device outlived the compile that was holding it"
    );
    assert_eq!(
        ticket.outcome(),
        Some(&PipelineOutcome::DeviceRebuilt),
        "the first answer stands"
    );
    match ask(&req) {
        Asked::Drawn(pixels) => assert_drawn_color(&pixels, seed),
        _ => panic!("the pipeline compiled on the new device was not drawn with"),
    }
}

/// A compile that ran out of memory is the waiting draw's refusal, once: the
/// draw resumed with its ticket gets that refusal and starts no compile,
/// although the negative cache does not remember it. A later, independent
/// draw may compile again.
#[test]
fn an_out_of_memory_compile_is_delivered_to_its_draw_once_and_only_later_draws_retry() {
    crate::observe::redirect_logs_for_tests();
    let seed = 3;
    let frag = frag_variant(seed);
    let req = request(&frag);
    let forced = ForcedResult::new(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
    let ticket = match ask(&req) {
        Asked::Pending(ticket) => ticket,
        Asked::Drawn(_) => panic!("a cold pipeline was built inside the draw"),
        Asked::NoDevice => return,
    };
    let PipelineOutcome::Failed(oom) = settled(&ticket) else {
        panic!("a forced out-of-memory compile did not fail");
    };
    assert!(oom.out_of_memory());

    let jobs_before = CENSUS.jobs.load(Ordering::Relaxed);
    let mut resumed = request(&frag);
    resumed.pipeline_wait = Some(ticket.clone());
    match execute_draw_request(&resumed) {
        Err(e) => assert_eq!(e, oom, "the resumed draw gets its compile's refusal"),
        Ok(_) => panic!("the resumed draw did not get the refusal its wait ended in"),
    }
    assert_eq!(
        CENSUS.jobs.load(Ordering::Relaxed),
        jobs_before,
        "the waiting packet started the same compile again"
    );

    drop(forced);
    let retried = match ask(&req) {
        Asked::Pending(retried) => retried,
        _ => panic!("an out-of-memory refusal was remembered"),
    };
    assert!(!retried.is(&ticket));
    assert_eq!(CENSUS.jobs.load(Ordering::Relaxed), jobs_before + 1);
    assert_eq!(settled(&retried), PipelineOutcome::Ready);
    match ask(&req) {
        Asked::Drawn(pixels) => assert_drawn_color(&pixels, seed),
        _ => panic!("the retried pipeline was not drawn with"),
    }
}

/// A compile refused for a reason about the request is remembered like every
/// other refused create: the next draw of the same key is refused at once,
/// without a compile.
#[test]
fn a_refused_compile_is_remembered_and_not_compiled_again() {
    crate::observe::redirect_logs_for_tests();
    let frag = frag_variant(4);
    let req = request(&frag);
    let forced = ForcedResult::new(vk::Result::ERROR_INITIALIZATION_FAILED);
    let ticket = match ask(&req) {
        Asked::Pending(ticket) => ticket,
        Asked::Drawn(_) => panic!("a cold pipeline was built inside the draw"),
        Asked::NoDevice => return,
    };
    let PipelineOutcome::Failed(refused) = settled(&ticket) else {
        panic!("a forced refusal did not fail");
    };
    drop(forced);
    let jobs_before = CENSUS.jobs.load(Ordering::Relaxed);
    match execute_draw_request(&req) {
        Err(e) => assert_eq!(e, refused),
        Ok(_) => panic!("a refused pipeline was asked for again"),
    }
    assert_eq!(CENSUS.jobs.load(Ordering::Relaxed), jobs_before);
}
