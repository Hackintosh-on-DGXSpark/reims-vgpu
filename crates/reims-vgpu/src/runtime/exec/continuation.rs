//! An `EXEC_INDIRECT2` packet that stopped at a draw waiting for its pipeline.
//!
//! # Why a packet can stop part-way
//!
//! A draw learns its Vulkan pipeline key only once its request is built — the
//! attachment formats, the layout variant, the topology and the viewport count
//! are all draw state — so whether the pipeline exists cannot be asked before
//! the packet starts, the way the AIR preflight asks about translations. When
//! the engine answers `PipelinePending`, the blits, dispatches and draws ahead
//! of that draw have already run and must not run again.
//!
//! So the packet keeps everything it needs to continue — the streams it read,
//! the stream and draw it stopped at, the chain state the draws after it read —
//! and stays at its FIFO head under the same scheduling the AIR deferral uses:
//! `translation_deferred_mask`, no stamp, every other FIFO held behind it. The
//! 4 ms poll retries it; a retry whose ticket is unanswered costs two atomic
//! loads. Once the ticket is answered the packet continues at the draw that
//! waited, which receives the ticket so that a compile that failed is its
//! answer rather than a reason to compile again.
//!
//! # A continuation is for exactly one ring entry
//!
//! It records the entry's channel, next head, completion stamp and payload,
//! and is only ever resumed by that entry. A retry that finds anything else at
//! the head is refused rather than run from the top — its first half has
//! already executed, and running it again would repeat its blits, dispatches
//! and draws — and so is a continuation whose channel is redefined, freed or
//! reset before it finishes. Each of those is guest work lost, and says so.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{ChainCursor, ExecResult, ExecSlot, StreamAccum};

/// One stopped packet.
#[cfg_attr(
    not(feature = "backend-vulkan"),
    allow(
        dead_code,
        reason = "only the Vulkan rail builds pipelines off the drain"
    )
)]
pub(crate) struct ExecContinuation {
    slot: ExecSlot,
    task_id: u32,
    payload: Vec<u8>,
    streams: Vec<Vec<u8>>,
    next_stream: usize,
    acc: StreamAccum,
    cursor: ChainCursor,
    /// What the packet has done so far, so the summary that closes it covers
    /// the whole packet.
    out: ExecResult,
    #[cfg(feature = "backend-vulkan")]
    ticket: crate::backend::vulkan::engine::PipelineTicket,
    parked_at: std::time::Instant,
}

/// The stopped packets, at most one per channel: a channel's head is the only
/// packet of it that can be running.
#[derive(Default)]
pub struct ExecContinuations {
    by_channel: HashMap<u32, ExecContinuation>,
}

impl std::fmt::Debug for ExecContinuations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut channels: Vec<u32> = self.by_channel.keys().copied().collect();
        channels.sort_unstable();
        f.debug_struct("ExecContinuations")
            .field("channels", &channels)
            .finish()
    }
}

impl ExecContinuations {
    pub fn len(&self) -> usize {
        self.by_channel.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_channel.is_empty()
    }

    /// Whether `channel`'s head packet stopped at a draw waiting for its
    /// pipeline.
    pub fn holds(&self, channel: u32) -> bool {
        self.by_channel.contains_key(&channel)
    }

    pub(super) fn take(&mut self, channel: u32) -> Option<ExecContinuation> {
        self.by_channel.remove(&channel)
    }

    /// Give up `channel`'s stopped packet, if it has one, because `why`. What
    /// it had left to run is lost, and the line says so.
    pub fn drop_channel(&mut self, channel: u32, why: &'static str) {
        let Some(parked) = self.by_channel.remove(&channel) else {
            return;
        };
        CENSUS.dropped.fetch_add(1, Ordering::Relaxed);
        crate::observe::fail(format!(
            "exec_continuation_dropped reason=exec_continuation_{why} ch={channel} task={} \
             stream={}/{} draw={} waited_ms={} (a packet that stopped at a draw waiting for its \
             pipeline will not finish; the work after that draw is lost)",
            parked.task_id,
            parked.next_stream,
            parked.streams.len(),
            parked.cursor.draw_index,
            parked.parked_at.elapsed().as_millis()
        ));
    }
}

/// Keep `stop`'s packet at its channel head, to continue where it stopped.
#[cfg(feature = "backend-vulkan")]
pub(super) fn park(
    state: &mut crate::model::DeviceState,
    slot: ExecSlot,
    payload: &[u8],
    task_id: u32,
    streams: Vec<Vec<u8>>,
    stop: super::StreamStop,
    out: &mut ExecResult,
) {
    CENSUS.deferrals.fetch_add(1, Ordering::Relaxed);
    out.deferred = Some(super::ExecDeferral::PipelineCompiling);
    let super::StreamStop {
        stream,
        acc,
        suspension,
    } = stop;
    state.exec_continuations.by_channel.insert(
        slot.channel,
        ExecContinuation {
            slot,
            task_id,
            payload: payload.to_vec(),
            streams,
            next_stream: stream,
            acc,
            cursor: suspension.cursor,
            out: out.clone(),
            ticket: suspension.ticket,
            parked_at: std::time::Instant::now(),
        },
    );
}

/// Continue the packet `parked` stopped, if `slot` and `payload` are that
/// packet and its pipeline has been answered.
#[cfg(feature = "backend-vulkan")]
pub(super) fn resume<M: crate::runtime::host::HostMemory + crate::runtime::host::HostOps>(
    state: &mut crate::model::DeviceState,
    host: &mut M,
    slot: ExecSlot,
    payload: &[u8],
    parked: ExecContinuation,
    started: std::time::Instant,
) -> ExecResult {
    if parked.slot != slot || parked.payload != payload {
        CENSUS.mismatches.fetch_add(1, Ordering::Relaxed);
        crate::observe::fail(format!(
            "exec_continuation_mismatch reason=exec_continuation_identity ch={} \
             next_head={}/{} stamp={:#x}/{:#x} len={}/{} task={} draw={} (the packet at the \
             channel head is not the one that stopped at a draw waiting for its pipeline; that \
             packet's first half already ran, so this one is refused rather than run from the \
             start)",
            slot.channel,
            slot.next_head,
            parked.slot.next_head,
            slot.completion_stamp,
            parked.slot.completion_stamp,
            payload.len(),
            parked.payload.len(),
            parked.task_id,
            parked.cursor.draw_index
        ));
        let mut out = parked.out;
        out.deferred = None;
        out.total_us = super::elapsed_us(started);
        return out;
    }
    if !crate::backend::vulkan::engine::pipeline_ticket_settled(&parked.ticket) {
        CENSUS.retries.fetch_add(1, Ordering::Relaxed);
        let out = ExecResult {
            task_id: parked.task_id,
            deferred: Some(super::ExecDeferral::PipelineCompiling),
            ..Default::default()
        };
        state
            .exec_continuations
            .by_channel
            .insert(slot.channel, parked);
        return out;
    }
    let waited = parked.parked_at.elapsed();
    CENSUS.resumes.fetch_add(1, Ordering::Relaxed);
    CENSUS
        .deferral_ns
        .fetch_add(waited.as_nanos() as u64, Ordering::Relaxed);
    CENSUS
        .deferral_max_ns
        .fetch_max(waited.as_nanos() as u64, Ordering::Relaxed);
    let ExecContinuation {
        task_id,
        streams,
        next_stream,
        acc,
        mut cursor,
        mut out,
        ticket,
        ..
    } = parked;
    out.deferred = None;
    cursor.pipeline_wait = Some(ticket);
    let mut measured_ns = 0u64;
    let stopped = super::run_streams(
        state,
        host,
        task_id,
        &streams,
        next_stream,
        Some((acc, cursor)),
        &mut out,
        &mut measured_ns,
    );
    if let Some(stop) = stopped {
        park(state, slot, payload, task_id, streams, stop, &mut out);
    }
    super::note_exec_header(started, measured_ns);
    out.total_us = super::elapsed_us(started);
    out
}

/// What stopped packets did over one census window.
struct Census {
    deferrals: AtomicU64,
    retries: AtomicU64,
    resumes: AtomicU64,
    deferral_ns: AtomicU64,
    deferral_max_ns: AtomicU64,
    mismatches: AtomicU64,
    dropped: AtomicU64,
}

static CENSUS: Census = Census {
    deferrals: AtomicU64::new(0),
    retries: AtomicU64::new(0),
    resumes: AtomicU64::new(0),
    deferral_ns: AtomicU64::new(0),
    deferral_max_ns: AtomicU64::new(0),
    mismatches: AtomicU64::new(0),
    dropped: AtomicU64::new(0),
};

/// The `exec_pipeline_wait` line for the window just ended, or `None` when no
/// packet stopped, waited or resumed.
///
/// `deferrals` is draws that stopped their packet at their pipeline;
/// `retries` the polls that found the ticket unanswered; `resumes` the packets
/// that continued; `deferral_us` and `deferral_max_us` how long the packets
/// that continued this window had been stopped, summed and at most. A stopped
/// packet holds every other FIFO behind it, so `deferral_max_us` is also the
/// longest any guest GPU work waited on a pipeline compile.
pub fn take_census(win_ms: u64) -> Option<String> {
    let take = |slot: &AtomicU64| slot.swap(0, Ordering::Relaxed);
    let deferrals = take(&CENSUS.deferrals);
    let retries = take(&CENSUS.retries);
    let resumes = take(&CENSUS.resumes);
    let deferral_ns = take(&CENSUS.deferral_ns);
    let deferral_max_ns = take(&CENSUS.deferral_max_ns);
    let mismatches = take(&CENSUS.mismatches);
    let dropped = take(&CENSUS.dropped);
    if deferrals + retries + resumes + mismatches + dropped == 0 {
        return None;
    }
    Some(format!(
        "exec_pipeline_wait win_ms={win_ms} deferrals={deferrals} retries={retries} \
         resumes={resumes} deferral_us={} deferral_max_us={} mismatches={mismatches} \
         dropped={dropped}",
        deferral_ns / 1000,
        deferral_max_ns / 1000
    ))
}
