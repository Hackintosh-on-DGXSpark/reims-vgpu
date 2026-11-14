//! A stand-in for the engine behind a pass's draw list, for tests of how a
//! packet stops at a draw and continues.
//!
//! While installed, a draw list resolves its attachment set to a fixed request
//! and each draw takes the next scripted answer instead of encoding. What each
//! draw was handed back — the ticket it waited on, if any — is recorded, so a
//! test can see which draws ran, in what order, and how many times.

use std::cell::RefCell;
use std::collections::VecDeque;

use crate::backend::vulkan::engine::{PipelineOutcome, PipelineTicket};
use crate::runtime::draw::{ColorRtRequest, DrawEncodeRequest, EncodeStatus};

pub(crate) enum Scripted {
    /// The draw ran.
    Drawn,
    /// The draw's pipeline is being built; the draw ran nothing.
    Pending(PipelineTicket),
    /// What the engine answers a draw that carries a ticket back: the refusal
    /// the ticket ended in, or a draw.
    AsEngine,
}

#[derive(Clone, Debug)]
pub(crate) struct Ran {
    pub(crate) pipeline_ref: u32,
    /// The ticket the draw carried back in, if it was resumed.
    pub(crate) carried: Option<PipelineTicket>,
    pub(crate) answer: &'static str,
}

struct Script {
    answers: VecDeque<Scripted>,
    ran: Vec<Ran>,
    requests: usize,
}

thread_local! {
    static SCRIPT: RefCell<Option<Script>> = const { RefCell::new(None) };
}

/// The installed script, until dropped.
pub(crate) struct ScriptGuard(());

pub(crate) fn install(answers: impl IntoIterator<Item = Scripted>) -> ScriptGuard {
    SCRIPT.with(|script| {
        *script.borrow_mut() = Some(Script {
            answers: answers.into_iter().collect(),
            ran: Vec::new(),
            requests: 0,
        })
    });
    ScriptGuard(())
}

impl ScriptGuard {
    /// Every draw that took an answer, in order.
    pub(crate) fn ran(&self) -> Vec<Ran> {
        SCRIPT.with(|script| {
            script
                .borrow()
                .as_ref()
                .map(|s| s.ran.clone())
                .unwrap_or_default()
        })
    }

    /// How many times a draw list resolved its attachment set.
    pub(crate) fn requests(&self) -> usize {
        SCRIPT.with(|script| script.borrow().as_ref().map_or(0, |s| s.requests))
    }
}

impl Drop for ScriptGuard {
    fn drop(&mut self) {
        SCRIPT.with(|script| *script.borrow_mut() = None);
    }
}

pub(super) fn request(pipeline_ref: u32) -> Option<DrawEncodeRequest> {
    SCRIPT.with(|script| {
        let mut script = script.borrow_mut();
        let script = script.as_mut()?;
        script.requests += 1;
        Some(DrawEncodeRequest {
            pipeline_ref,
            colors: vec![ColorRtRequest {
                width: 4,
                height: 4,
                ..Default::default()
            }],
            ..Default::default()
        })
    })
}

pub(super) fn encode(req: &mut DrawEncodeRequest) -> Option<(EncodeStatus, Option<Vec<u8>>)> {
    SCRIPT.with(|script| {
        let mut script = script.borrow_mut();
        let script = script.as_mut()?;
        let carried = req.pipeline_wait.take();
        let answer = script
            .answers
            .pop_front()
            .expect("a draw ran that the script has no answer for");
        let failed = carried.as_ref().and_then(|ticket| match ticket.outcome() {
            Some(PipelineOutcome::Failed(_)) => Some(()),
            _ => None,
        });
        let (status, name) = match answer {
            Scripted::Pending(ticket) => {
                req.pipeline_wait = Some(ticket);
                (EncodeStatus::PipelinePending, "pending")
            }
            Scripted::AsEngine if failed.is_some() => {
                (EncodeStatus::BadArgs("scripted_pipeline_failed"), "refused")
            }
            Scripted::Drawn | Scripted::AsEngine => {
                req.chain_resident_established = true;
                (EncodeStatus::Ok, "drawn")
            }
        };
        script.ran.push(Ran {
            pipeline_ref: req.pipeline_ref,
            carried,
            answer: name,
        });
        Some((status, None))
    })
}
