//! [`ParallelSeam`] — the N-slot concurrent generation engine behind `infr serve --parallel N`.
//!
//! # What this is (and what it is not)
//!
//! It is a hybrid concurrent engine. Every sequence owns one KV slot. Compatible Qwen3.8 prefill
//! requests divide the configured ubatch and run as one layer-synchronous activation batch; their
//! decode frontiers then continue in one forward at `m = n_active`. Stateless work and paged
//! expert traffic are shared while every sequence retains its own positions, KV/recurrent state
//! and sampler. Other architectures and incompatible QSA phases keep the established interleaved
//! path through [`crate::sampling::StepGate`].
//!
//! Qwen3.8 cohorts admit newly arrived work at token boundaries. A ready decode joins the next
//! aggregated forward; a short prefill can share its final fitting ubatch with ready decode rows,
//! while a longer prefill temporarily takes the cohort through layer-synchronous prefill first.
//! Incompatible QSA phases split cleanly back into separate cohorts. The worker still releases the
//! GPU baton after every graph, so unsupported or fallback work can continue through `StepGate`.
//!
//! # VRAM: how `-np` interacts with `--ctx`
//!
//! N slots means N independent KV/recurrent states. The fit solver prices all N states plus one
//! shared runtime workspace and returns the maximum per-slot context, so raising `-np` cannot OOM
//! a device that `-np 1` fit. (It is NOT the same footprint — when the trained window is below the
//! fit, `-np 4` may allocate more total state than `-np 1`; what it cannot do is exceed the budget.
//! The visible cost can be a smaller per-request window.) An
//! explicit `--ctx C` is used verbatim per slot, and the Vulkan alloc-time budget guard is left to
//! fail it cleanly if `N * C` truly doesn't fit.
//!
//! Slots are forked EAGERLY at startup (weights are shared through `Arc<SeamWeights>`; a fork costs
//! only its own KV + IO buffers). That means a VRAM refusal happens at boot with a clear message,
//! never halfway through serving.
//!
//! When `kv.session_cache_dir` is explicitly set, a background worker streams idle dynamic Q8 KV
//! state to checksummed files and releases its physical 32K segments. A later prefix match restores
//! that state into any free resident slot. This extends the number of retained conversations; it
//! does not alter the decode cohort or fallback scheduling described above.

use crate::sampling::{ParallelSampler, RequestCtx, StepGate};
use crate::seam::SeamKv;
use crate::session_cache::SessionCache;
use crate::{Config, GenStats, SeamModel};
use anyhow::{anyhow, Context, Result};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const DECODE_BATCH_WAIT: Duration = Duration::from_secs(1);
const PREFILL_BATCH_WAIT: Duration = Duration::from_millis(10);
const MAX_DECODE_BATCH: usize = 8;

/// One projector result handed to the text model. Kept backend-neutral so `infr-llama` does not
/// depend on the optional vision crate.
pub struct MultimodalEmbedding {
    pub values: Vec<f32>,
    pub grid_nx: usize,
    pub grid_ny: usize,
}

fn expand_multimodal_prompt(
    tokens: &[u32],
    image_pad_id: u32,
    images: Vec<MultimodalEmbedding>,
    n_embd: usize,
) -> Result<(Vec<u32>, crate::seam::MropePlan)> {
    let mut expanded = Vec::new();
    let mut positions4 = Vec::new();
    let mut spans = Vec::with_capacity(images.len());
    let mut images = images.into_iter();
    let mut used = 0usize;
    let mut cursor = 0i32;

    for &token in tokens {
        if token != image_pad_id {
            expanded.push(token);
            positions4.extend_from_slice(&[cursor, cursor, cursor, 0]);
            cursor = cursor
                .checked_add(1)
                .ok_or_else(|| anyhow!("multimodal position overflow"))?;
            continue;
        }

        let image = images.next().ok_or_else(|| {
            anyhow!("rendered prompt contains more <|image_pad|> markers than image payloads")
        })?;
        used += 1;
        let n_tokens = image
            .grid_nx
            .checked_mul(image.grid_ny)
            .ok_or_else(|| anyhow!("image #{used} token grid overflows"))?;
        let expected_values = n_tokens
            .checked_mul(n_embd)
            .ok_or_else(|| anyhow!("image #{used} embedding size overflows"))?;
        if n_tokens == 0 || image.values.len() != expected_values {
            return Err(anyhow!(
                "image #{used} projector output has {} values for a {}x{} grid; expected {}",
                image.values.len(),
                image.grid_nx,
                image.grid_ny,
                expected_values
            ));
        }
        let start = expanded.len();
        for index in 0..n_tokens {
            let y = i32::try_from(index / image.grid_nx)
                .map_err(|_| anyhow!("image #{used} row exceeds i32"))?;
            let x = i32::try_from(index % image.grid_nx)
                .map_err(|_| anyhow!("image #{used} column exceeds i32"))?;
            expanded.push(image_pad_id);
            positions4.extend_from_slice(&[
                cursor,
                cursor
                    .checked_add(y)
                    .ok_or_else(|| anyhow!("image #{used} row position overflow"))?,
                cursor
                    .checked_add(x)
                    .ok_or_else(|| anyhow!("image #{used} column position overflow"))?,
                0,
            ]);
        }
        let extent = i32::try_from(image.grid_nx.max(image.grid_ny))
            .map_err(|_| anyhow!("image #{used} grid exceeds i32"))?;
        cursor = cursor
            .checked_add(extent)
            .ok_or_else(|| anyhow!("image #{used} position overflow"))?;
        spans.push(crate::seam::ImageSpanEmbeds {
            start,
            n_tokens,
            embeds: Arc::new(image.values),
        });
    }
    if images.next().is_some() {
        return Err(anyhow!(
            "request carries more image payloads than rendered <|image_pad|> markers"
        ));
    }
    Ok((
        expanded,
        crate::seam::MropePlan {
            prompt_pos4: positions4,
            spans,
            decode_base: cursor,
        },
    ))
}

/// Pure continuation-slot selection (the "this conversation continuing" case of [`checkout`], and
/// the twin of `seam::model::SlotPool::pick`'s first arm). Given `(slot_idx, prefix_score,
/// cached_len)` for each candidate free slot and the `prompt_len`, pick the qualifying slot with
/// the LONGEST reusable prefix — a slot qualifies when the prompt EXTENDS its cache (`score ==
/// cached_len`) or EQUALS it (`score == prompt_len`), and its score is positive. Returns the
/// winning `slot_idx`, or `None` if no slot qualifies.
///
/// Split out as a pure fn so this decision is unit-testable without a live Vulkan backend / KV
/// slots (the lock-drop and device-side `seed_from` around it stay integration-only).
fn pick_continuation(
    candidates: impl IntoIterator<Item = (usize, usize, usize)>,
    prompt_len: usize,
) -> Option<usize> {
    candidates
        .into_iter()
        .filter(|&(_, score, cached)| score > 0 && (score == cached || score == prompt_len))
        .max_by_key(|&(_, score, _)| score)
        .map(|(idx, _, _)| idx)
}

fn merge_stats(mut total: GenStats, tail: GenStats) -> GenStats {
    total.n_prompt += tail.n_prompt;
    total.n_cached += tail.n_cached;
    total.prompt_secs += tail.prompt_secs;
    total.n_gen += tail.n_gen;
    total.decode_secs += tail.decode_secs;
    total
}

/// One KV slot: its cache (moved OUT while a request holds it, so the generation gets the
/// `&mut Option<SeamKv>` the runner wants without holding the pool lock), plus the bookkeeping the
/// prefix-match/LRU policy needs.
struct Slot {
    /// `None` while checked out by an in-flight request, or before the slot is initialized.
    kv: Option<SeamKv>,
    busy: bool,
    /// LRU stamp.
    tick: u64,
    /// Set only when the opt-in cold session cache is active. Ordinary serving does not read the
    /// wall clock during slot selection or return.
    idle_since: Option<Instant>,
}

/// The N-slot pool. Deliberately a plain `Mutex` + `Condvar` rather than the sequential
/// [`crate::seam::model`] `SlotPool`: checkout has to MOVE the `SeamKv` out (a generation holds it
/// for its whole lifetime, which is far too long to hold a lock over) and has to consider only the
/// slots that are actually free.
struct Pool {
    slots: Vec<Slot>,
    tick: u64,
}

/// A checked-out slot. Returns its KV to the pool on drop — including on error or panic, so a
/// failed request can never permanently burn a slot.
enum BatchEvent {
    Token {
        id: u32,
        progress: infr_core::GenerationProgress,
    },
    Complete {
        kv: SeamKv,
        stats: GenStats,
    },
    Fallback {
        kv: SeamKv,
        prompt: Vec<u32>,
        max_new: usize,
        stats: GenStats,
        prefilled: bool,
        turn_checkpoint: Option<crate::seam::TurnCheckpoint>,
    },
    Failed {
        kv: SeamKv,
        error: String,
    },
}

impl BatchEvent {
    fn into_kv(self) -> Option<SeamKv> {
        match self {
            Self::Token { .. } => None,
            Self::Complete { kv, .. } | Self::Fallback { kv, .. } | Self::Failed { kv, .. } => {
                Some(kv)
            }
        }
    }
}

struct BatchWork {
    slot: usize,
    kv: Option<SeamKv>,
    prompt: Vec<u32>,
    max_new: usize,
    generated: usize,
    stats: GenStats,
    prefilled: bool,
    turn_checkpoint: Option<crate::seam::TurnCheckpoint>,
    sampler: Option<ParallelSampler>,
    channels: Option<BatchChannels>,
}

struct BatchChannels {
    events: SyncSender<BatchEvent>,
    acknowledgements: Receiver<bool>,
}

#[derive(Default)]
struct DecodeBatchQueue {
    running: bool,
    /// Batch-eligible requests registered before checkout/prefill but not yet ready for decode.
    prefilling: usize,
    waiting: VecDeque<BatchWork>,
    prefill_waiting: VecDeque<BatchWork>,
}

fn should_wait_for_decode_peers(prefilling: usize, ready_peers: usize) -> bool {
    prefilling > 0 && ready_peers + 1 < MAX_DECODE_BATCH
}

struct BatchPrefillRegistration<'a> {
    engine: &'a ParallelSeam,
    active: bool,
}

impl<'a> BatchPrefillRegistration<'a> {
    fn new(engine: &'a ParallelSeam) -> Self {
        engine
            .decode_batch
            .lock()
            .expect("decode batch queue poisoned")
            .prefilling += 1;
        Self {
            engine,
            active: true,
        }
    }

    fn arrive(&mut self, queue: &mut DecodeBatchQueue) {
        debug_assert!(self.active);
        queue.prefilling = queue.prefilling.saturating_sub(1);
        self.active = false;
    }
}

impl Drop for BatchPrefillRegistration<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut queue = match self.engine.decode_batch.lock() {
            Ok(queue) => queue,
            Err(error) => error.into_inner(),
        };
        queue.prefilling = queue.prefilling.saturating_sub(1);
        drop(queue);
        self.engine.decode_ready.notify_all();
    }
}

struct SlotGuard<'a> {
    engine: &'a ParallelSeam,
    idx: usize,
    kv: Option<SeamKv>,
    detached: bool,
}

impl SlotGuard<'_> {
    fn detach(&mut self) -> SeamKv {
        self.detached = true;
        self.kv.take().expect("checked-out slot has KV")
    }

    fn reattach(&mut self, kv: SeamKv) {
        self.kv = Some(kv);
        self.detached = false;
    }

    fn abandon_detached(&mut self) {
        self.kv = None;
        self.detached = false;
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if self.detached {
            return;
        }
        let mut p = match self.engine.pool.lock() {
            Ok(p) => p,
            Err(e) => e.into_inner(),
        };
        p.tick += 1;
        let tick = p.tick;
        let s = &mut p.slots[self.idx];
        s.kv = self.kv.take();
        s.busy = false;
        s.tick = tick;
        if self.engine.session_cache.is_some() {
            s.idle_since = Some(Instant::now());
        }
        drop(p);
        if self.engine.session_cache.is_some() {
            // Both a queued request and the deadline worker may be asleep on this condition.
            self.engine.freed.notify_all();
        } else {
            self.engine.freed.notify_one();
        }
    }
}

struct ColdWorker {
    stop: Arc<AtomicBool>,
    wake: Arc<Condvar>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for ColdWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.wake.notify_all();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::warn!("cold KV maintenance thread panicked while shutting down");
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cold_session_worker(
    pool: Arc<Mutex<Pool>>,
    wake: Arc<Condvar>,
    cache: Arc<Mutex<SessionCache>>,
    backend: Arc<infr_vulkan::VulkanBackend>,
    gate: Option<Arc<StepGate>>,
    model_cfg: Config,
    idle_for: Duration,
    stop: Arc<AtomicBool>,
) {
    loop {
        let selected = {
            let mut pool_guard = match pool.lock() {
                Ok(pool) => pool,
                Err(poisoned) => poisoned.into_inner(),
            };
            loop {
                if stop.load(Ordering::Acquire) {
                    break None;
                }
                let now = Instant::now();
                let mut next_wait: Option<Duration> = None;
                let mut candidate: Option<usize> = None;
                for (index, slot) in pool_guard.slots.iter().enumerate() {
                    let Some(idle_since) = slot.idle_since else {
                        continue;
                    };
                    if slot.busy || slot.kv.as_ref().is_none_or(|kv| kv.cached_len() == 0) {
                        continue;
                    }
                    let elapsed = now.saturating_duration_since(idle_since);
                    if elapsed >= idle_for {
                        if candidate.is_none_or(|old| slot.tick < pool_guard.slots[old].tick) {
                            candidate = Some(index);
                        }
                    } else {
                        let remaining = idle_for - elapsed;
                        next_wait = Some(next_wait.map_or(remaining, |old| old.min(remaining)));
                    }
                }
                if let Some(index) = candidate {
                    pool_guard.slots[index].busy = true;
                    pool_guard.slots[index].idle_since = None;
                    let kv = pool_guard.slots[index]
                        .kv
                        .take()
                        .expect("cold candidate has a resident KV slot");
                    break Some((index, kv));
                }
                pool_guard = match next_wait {
                    Some(timeout) => {
                        wake.wait_timeout(pool_guard, timeout)
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .0
                    }
                    None => wake
                        .wait(pool_guard)
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                };
            }
        };
        let Some((index, mut kv)) = selected else {
            break;
        };

        if !stop.load(Ordering::Acquire) {
            let _gate = gate.as_deref().map(StepGate::enter);
            if !stop.load(Ordering::Acquire) {
                let mut cache = match cache.lock() {
                    Ok(cache) => cache,
                    Err(poisoned) => poisoned.into_inner(),
                };
                match cache.spill(&mut kv, backend.as_ref(), &model_cfg) {
                    Ok(true) => {}
                    Ok(false) => {}
                    Err(error) => tracing::warn!(
                        slot = index,
                        "cold KV idle spill failed; keeping the resident state: {error}"
                    ),
                }
                if let Err(error) = cache.gc() {
                    tracing::warn!("cold KV cache maintenance failed: {error}");
                }
            }
        }

        let mut pool_guard = match pool.lock() {
            Ok(pool) => pool,
            Err(poisoned) => poisoned.into_inner(),
        };
        let slot = &mut pool_guard.slots[index];
        slot.kv = Some(kv);
        slot.busy = false;
        slot.idle_since = Some(Instant::now());
        drop(pool_guard);
        wake.notify_all();
    }
}

/// The concurrent seam engine. `Sync`: `&self` is all a request needs, so N of them run at once.
pub struct ParallelSeam {
    /// Declared first so its `Drop` joins the maintenance thread before model/backend fields drop.
    cold_worker: Option<ColdWorker>,
    model: SeamModel,
    vk: Arc<infr_vulkan::VulkanBackend>,
    pool: Arc<Mutex<Pool>>,
    /// Signalled when a slot is returned — a queued request waits here.
    freed: Arc<Condvar>,
    decode_batch: Mutex<DecodeBatchQueue>,
    /// Wakes a cohort leader when a registered request reaches (or abandons) decode.
    decode_ready: Condvar,
    /// Set only when an active decode worker has new work to admit. The runner polls it once per
    /// aggregated token; an empty steady-state decode takes no queue lock.
    batch_interrupt: AtomicBool,
    /// The GPU baton. `None` when `n_slots == 1`: a lone sequence must not pay even a mutex per
    /// token, and single-request decode speed is a hard non-regression requirement.
    gate: Option<Arc<StepGate>>,
    /// Opt-in disk-backed conversation catalog. `None` keeps the pre-existing checkout path
    /// byte-for-byte isolated from file I/O and cache locking.
    session_cache: Option<Arc<Mutex<SessionCache>>>,
    session_idle: Duration,
    max_ctx: usize,
    /// This model's OWN placement pins (pinned prefill chunk / auto-q8 KV — see
    /// [`crate::seam::PlacementPins`]). Per-engine so a multi-model host (`infr multi` runs N of
    /// these concurrently) never leaks one model's ladder decision to another. Entered as the
    /// current [`crate::seam::PlacementScope`] around placement (construction/warmup) and every
    /// request's decode; concurrent requests on THIS engine all point at this one shared cell.
    pins: Arc<crate::seam::PlacementPins>,
}

impl ParallelSeam {
    fn batch_decode_progress(
        &self,
        prompt_tokens: usize,
        cached_prompt_tokens: usize,
        completion_tokens: usize,
    ) -> infr_core::GenerationProgress {
        let cached_prompt_tokens = cached_prompt_tokens.min(prompt_tokens);
        infr_core::GenerationProgress {
            phase: infr_core::GenerationPhase::Decode,
            prompt_tokens: prompt_tokens as u64,
            cached_prompt_tokens: cached_prompt_tokens as u64,
            prefill_tokens: prompt_tokens.saturating_sub(cached_prompt_tokens) as u64,
            completion_tokens: completion_tokens as u64,
            context_tokens: prompt_tokens.saturating_add(completion_tokens) as u64,
            context_limit: self.max_ctx as u64,
        }
    }

    /// Build an N-slot engine: upload the weights once (via a warmup generation on slot 0, which is
    /// also what compiles every lazily-built pipeline), then fork N-1 sibling slots off it.
    ///
    /// `want_ctx` is the `--ctx` / `INFR_CTX` spec (token count or `%` of the free-VRAM KV
    /// capacity); `None` derives the per-slot window. See [`SeamModel::vulkan_slot_ctx`].
    pub fn new(
        model: SeamModel,
        n_slots: usize,
        want_ctx: Option<infr_core::SizeSpec>,
    ) -> Result<Self> {
        Self::new_on(None, model, n_slots, want_ctx)
    }

    /// [`new`](Self::new) pinned to physical device `dev`: `Some(idx)` binds `VulkanN`
    /// ([`infr_vulkan::VulkanBackend::new_on_with`], bypassing `device.dev`/the discrete-default
    /// rule), `None` is the default device (byte-identical to `new`). This is what lets `infr multi` host
    /// several concurrent-slot engines side by side, each on its own GPU: the whole engine (weights,
    /// N KV slots, recorder) lives on the ONE backend this constructs, so nothing crosses devices.
    pub fn new_on(
        dev: Option<usize>,
        model: SeamModel,
        n_slots: usize,
        want_ctx: Option<infr_core::SizeSpec>,
    ) -> Result<Self> {
        let n_slots = n_slots.max(1);
        let ecfg = model.cfg().clone();
        let vk = match dev {
            Some(idx) => infr_vulkan::VulkanBackend::new_on_with(idx, ecfg)
                .map_err(|e| anyhow!("vulkan init (Vulkan{idx}): {e}"))?,
            None => infr_vulkan::VulkanBackend::new_with(ecfg)
                .map_err(|e| anyhow!("vulkan init: {e}"))?,
        };
        Self::new_with_backend(model, n_slots, want_ctx, vk)
    }

    /// Build on a caller-owned Vulkan backend. Used by the unified service path so auxiliary
    /// engines can later derive clients from the exact device and elastic arena initialized by
    /// this LLM warmup.
    pub fn new_with_backend(
        model: SeamModel,
        n_slots: usize,
        want_ctx: Option<infr_core::SizeSpec>,
        vk: infr_vulkan::VulkanBackend,
    ) -> Result<Self> {
        let n_slots = n_slots.max(1);
        // This engine's own placement pins, entered as the current scope for the whole
        // placement phase (the clamp inside `vulkan_slot_ctx` + the `init_slots` warmup, which is
        // where the binder pins the prefill chunk / auto-q8 KV). See `PlacementPins`.
        let pins = Arc::new(crate::seam::PlacementPins::for_slots(n_slots));
        let scope = crate::seam::PlacementScope::enter(pins.clone());
        let max_ctx = model.vulkan_slot_ctx(&vk, n_slots, want_ctx)?;
        let session_idle = Duration::from_secs(model.engine_cfg().kv.session_idle_secs);
        let mut engine = Self {
            cold_worker: None,
            model,
            vk: Arc::new(vk),
            pool: Arc::new(Mutex::new(Pool {
                slots: Vec::new(),
                tick: 0,
            })),
            freed: Arc::new(Condvar::new()),
            decode_batch: Mutex::new(DecodeBatchQueue::default()),
            decode_ready: Condvar::new(),
            batch_interrupt: AtomicBool::new(false),
            // A 1-slot server has nothing to take turns with — keep it on the exact uncontended
            // path `infr run` takes (see `RequestCtx::gate_pass`: `None` constructs nothing).
            gate: (n_slots > 1).then(|| Arc::new(StepGate::new())),
            session_cache: None,
            session_idle,
            max_ctx,
            pins,
        };
        engine.init_slots(n_slots)?;
        engine.init_session_cache()?;
        drop(scope);
        engine.start_cold_worker()?;
        Ok(engine)
    }

    fn init_session_cache(&mut self) -> Result<()> {
        if self.model.engine_cfg().kv.session_cache_dir.is_none() {
            return Ok(());
        }
        let (supports_release, meta) = {
            let pool = self.pool.lock().expect("fresh pool");
            let kv = pool
                .slots
                .first()
                .and_then(|slot| slot.kv.as_ref())
                .ok_or_else(|| anyhow!("cold KV cache initialized before slot 0"))?;
            (kv.can_release_session_state(), kv.session_state_meta())
        };
        if !supports_release {
            return Err(anyhow!(
                "kv.session_cache_dir requires dynamic segmented KV; use Qwen3.5/3.6/3.8 with Q8 KV and leave kv.dynamic enabled"
            ));
        }
        self.session_cache = SessionCache::open(self.model.engine_cfg(), self.model.gguf(), &meta)?
            .map(|cache| Arc::new(Mutex::new(cache)));
        if self.session_cache.is_some() {
            let now = Instant::now();
            for slot in &mut self.pool.lock().expect("fresh pool").slots {
                slot.idle_since = Some(now);
            }
        }
        Ok(())
    }

    fn start_cold_worker(&mut self) -> Result<()> {
        let Some(cache) = self.session_cache.as_ref().map(Arc::clone) else {
            return Ok(());
        };
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let pool = Arc::clone(&self.pool);
        let wake = Arc::clone(&self.freed);
        let backend = Arc::clone(&self.vk);
        let gate = self.gate.as_ref().map(Arc::clone);
        let model_cfg = self.model.config().clone();
        let idle = self.session_idle;
        let thread = std::thread::Builder::new()
            .name("infr-kv-cold".into())
            .spawn(move || {
                cold_session_worker(
                    pool,
                    wake,
                    cache,
                    backend,
                    gate,
                    model_cfg,
                    idle,
                    worker_stop,
                )
            })
            .context("start cold KV maintenance thread")?;
        self.cold_worker = Some(ColdWorker {
            stop,
            wake: Arc::clone(&self.freed),
            thread: Some(thread),
        });
        Ok(())
    }

    pub fn fork_embedding_backend(&self) -> Result<infr_vulkan::VulkanBackend> {
        self.vk
            .fork_embedding_client()
            .map_err(|error| anyhow!("derive unified Embedding backend: {error}"))
    }

    pub fn fork_vision_backend(&self) -> Result<infr_vulkan::VulkanBackend> {
        self.vk
            .fork_vision_client()
            .map_err(|error| anyhow!("derive unified Vision backend: {error}"))
    }

    pub fn unified_vram_stats(&self) -> Option<infr_vulkan::unified::UnifiedVramStats> {
        self.vk.unified_vram().map(|pool| pool.stats())
    }

    /// Materialize slot 0 (weights + KV + pipelines) with a throwaway generation, then fork the
    /// rest off it. `&mut self` — this runs at startup, before the engine is shared.
    fn init_slots(&mut self, n_slots: usize) -> Result<()> {
        self.vk.defer_session_finalization(true);
        let initialized = self.init_slots_before_host_import(n_slots);
        if let Err(error) = initialized {
            self.vk.defer_session_finalization(false);
            return Err(error);
        }
        self.vk
            .finish_deferred_session_allocations()
            .map_err(|error| anyhow!("finalize Vulkan session allocations: {error}"))
    }

    /// Build every persistent slot before optional WDDM Host DMA aliases are admitted. The warmup
    /// still sees the proportionally preloaded RAM tier; only its faster Vulkan alias is deferred.
    fn init_slots_before_host_import(&mut self, n_slots: usize) -> Result<()> {
        let t0 = std::time::Instant::now();
        // The warmup generation both uploads the weights and compiles every lazily-built pipeline,
        // so the first REAL request pays neither. INFR_PROF_OPS is suppressed for it (recorders read
        // it at construction; warmup submits would pollute a later bench's per-op aggregate) via the
        // shared `with_profiling_suppressed` helper.
        let mut slot0: Option<SeamKv> = None;
        crate::with_profiling_suppressed(|| {
            crate::seam::generate_dense_vulkan_session(
                &self.vk,
                self.model.gguf(),
                self.model.config(),
                self.model.engine_cfg(),
                self.model.embd(),
                self.model.per_layer_embd(),
                &[1u32],
                2,
                |_| {},
                &mut slot0,
                self.max_ctx,
                Some(crate::seam::TurnCheckpoint::Enable),
                None, // constraint
                None, // req: startup, not a request — env sampling, no gate
                None, // multimodal plan
            )
        })?;
        let mut slot0 = slot0.ok_or_else(|| anyhow!("warmup did not initialize a KV slot"))?;
        // The warmup is what loads the weights, so it is also where the cold init re-clamps the
        // context against the memory the device reports free once they are resident. Take the
        // window that was actually allocated: every slot forked below is sized from it, and it is
        // what the server advertises and admits requests against.
        self.max_ctx = slot0.max_ctx();
        // Drop the warmup tokens so the first real prompt prefills a clean slot from row 0 instead
        // of forking off a garbage prefix.
        slot0.reset();

        let preallocated = slot0.take_preallocated_siblings();
        if !preallocated.is_empty() && preallocated.len() != n_slots.saturating_sub(1) {
            return Err(anyhow!(
                "startup materialized {} sibling slots, expected {}",
                preallocated.len(),
                n_slots.saturating_sub(1),
            ));
        }
        let mut preallocated = preallocated.into_iter();
        let mut slots = Vec::with_capacity(n_slots);
        for i in 1..n_slots {
            // A fork shares the `Arc<SeamWeights>` — it costs only its own KV + IO buffers. If VRAM
            // refuses, say so HERE, at boot, with the two knobs that fix it. Never mid-request.
            let kv = match preallocated.next() {
                Some(kv) => kv,
                None => slot0
                    .fork(
                        self.vk.as_ref(),
                        self.model.config(),
                        self.model.engine_cfg(),
                    )
                    .map_err(|e| {
                        anyhow!(
                            "could not allocate KV slot {}/{n_slots} at ctx {}: {e}\n\
                         lower --parallel, or lower --ctx (each slot owns a full context of KV cache)",
                            i + 1,
                            self.max_ctx,
                        )
                    })?,
            };
            slots.push(Slot {
                kv: Some(kv),
                busy: false,
                tick: 0,
                idle_since: None,
            });
        }
        slots.insert(
            0,
            Slot {
                kv: Some(slot0),
                busy: false,
                tick: 0,
                idle_since: None,
            },
        );
        self.pool.lock().expect("fresh pool").slots = slots;
        tracing::info!(
            "slots: {n_slots} x {} ctx ready in {:.1}s",
            self.max_ctx,
            t0.elapsed().as_secs_f32()
        );
        Ok(())
    }

    pub fn n_slots(&self) -> usize {
        self.pool.lock().expect("pool poisoned").slots.len()
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// The physical device this engine's backend bound (e.g. the discrete GPU or the iGPU). Used by
    /// `infr multi` to print the model→device routing table — two engines pinned to different
    /// indices report different names.
    pub fn device_name(&self) -> String {
        use infr_core::backend::Backend;
        self.vk.capabilities().name
    }

    pub fn model(&self) -> &SeamModel {
        &self.model
    }

    /// A fresh per-sequence context wired to this engine's baton — one per request.
    pub fn request_ctx(&self, sampling: crate::sampling::RequestSampling) -> RequestCtx {
        match &self.gate {
            Some(g) => RequestCtx::with_gate(sampling, g.clone()),
            None => RequestCtx::new(sampling),
        }
    }

    /// Take a slot for `prompt`, blocking until one is free.
    ///
    /// Slot choice preserves the cross-request KV prefix cache: a prompt that EXTENDS (or equals) a
    /// free slot's cached tokens continues that slot — this is the persistent prefix cache that
    /// makes a repeated system prompt ~7x cheaper on TTFT, and it is why the pick runs BEFORE the
    /// generation rather than round-robining blindly. Otherwise the least-recently-used free slot is
    /// recycled, seeded (device-side KV copy) from whichever free slot shares the longest prefix.
    ///
    /// Only FREE slots are considered: a busy slot's KV is checked out and cannot be read or
    /// recycled. So under load the prefix cache degrades gracefully (fewer candidate slots) rather
    /// than corrupting an in-flight sequence.
    fn checkout(&self, prompt: &[u32], req: &RequestCtx) -> Result<SlotGuard<'_>> {
        if self.session_cache.is_some() {
            self.checkout_with_cold_cache(prompt, req)
        } else {
            self.checkout_resident(prompt, req)
        }
    }

    /// Original resident-only checkout path. Keeping it separate means an unset cold-cache path
    /// does not add a file-cache mutex or wall-clock query to ordinary serving.
    fn checkout_resident(&self, prompt: &[u32], req: &RequestCtx) -> Result<SlotGuard<'_>> {
        /// Seeding shorter prefixes than this isn't worth the copy submit.
        const MIN_SEED: usize = 16;
        let cfg: &Config = self.model.config();
        let ec = self.model.engine_cfg();
        let mut p = self.pool.lock().expect("pool poisoned");
        loop {
            let free: Vec<usize> = (0..p.slots.len()).filter(|&i| !p.slots[i].busy).collect();
            if free.is_empty() {
                // Every slot is generating. The server's admission semaphore normally prevents this
                // (it bounds in-flight requests to n_slots), so this is the belt-and-braces path.
                p = self.freed.wait(p).expect("pool poisoned");
                continue;
            }
            let score = |s: &Slot| s.kv.as_ref().map_or(0, |k| k.prefix_score(prompt));
            // 1. This conversation continuing: the free slot with the LONGEST reusable prefix among
            //    those the prompt extends (or equals) — not merely the first such slot (which would
            //    re-prefill more suffix). Pure decision, unit-tested via `pick_continuation`.
            let cont = pick_continuation(
                free.iter().filter_map(|&i| {
                    p.slots[i].kv.as_ref().and_then(|k| {
                        k.continuation_prefix_len(prompt)
                            .map(|prefix| (i, prefix, prefix))
                    })
                }),
                prompt.len(),
            );
            // 2. Otherwise the LRU free slot, preferring an already-empty one (nothing to lose).
            let target = match cont {
                Some(i) => i,
                None => *free
                    .iter()
                    .min_by_key(|&&i| {
                        let empty = p.slots[i].kv.as_ref().is_none_or(|k| k.cached_len() == 0);
                        (!empty, p.slots[i].tick)
                    })
                    .expect("free is non-empty"),
            };
            // Seed the target with the best shared prefix among the other FREE slots (a common
            // system prompt), via a device-side KV copy instead of re-prefilling it.
            if cont.is_none() {
                let best = free
                    .iter()
                    .copied()
                    .filter(|&i| i != target)
                    .max_by_key(|&i| score(&p.slots[i]));
                if let Some(best) = best {
                    let best_s = score(&p.slots[best]);
                    if best_s >= MIN_SEED && best_s > score(&p.slots[target]) {
                        // `seed_from` is a device-side KV copy — it RECORDS, so it takes a turn on
                        // the baton like any other GPU submit. The module doc forbids holding the
                        // pool `Mutex` across a submit (it would serialize every other request's
                        // checkout/drop behind this copy). So: RESERVE both slots (mark them busy so
                        // no concurrent checkout can select them), take their KV out, DROP the lock
                        // across `seed_from`, then re-acquire to put them back.
                        p.slots[best].busy = true;
                        p.slots[target].busy = true;
                        let src = p.slots[best].kv.take().expect("scored slot is Some");
                        let mut dst = p.slots[target].kv.take();
                        drop(p);
                        let r = {
                            let _gp = req.gate_pass();
                            match dst.as_mut() {
                                Some(dst) => dst.seed_from(self.vk.as_ref(), cfg, ec, &src, best_s),
                                None => Ok(()),
                            }
                        };
                        p = self.pool.lock().expect("pool poisoned");
                        // Return the source slot's KV and release its reservation; `target` stays
                        // reserved (busy) — it is checked out just below.
                        p.slots[best].kv = Some(src);
                        p.slots[best].busy = false;
                        p.slots[target].kv = dst;
                        // A failed seed costs only the prefix reuse — the slot re-prefills from
                        // scratch and the answer is identical. Never fail the request for it.
                        if let Err(e) = r {
                            tracing::warn!(
                                "kv slots: prefix seed failed ({e}); re-prefilling instead"
                            );
                        }
                        // `best` just went free again — wake a waiter that may want it.
                        self.freed.notify_one();
                    }
                }
            }
            p.tick += 1;
            let tick = p.tick;
            p.slots[target].busy = true;
            p.slots[target].tick = tick;
            let kv = p.slots[target].kv.take();
            return Ok(SlotGuard {
                engine: self,
                idx: target,
                kv,
                detached: false,
            });
        }
    }

    /// Disk-extended slot checkout. Cold storage is deliberately serialized independently of the
    /// slot pool: a target is reserved under `pool`, then all GPU/file work runs after that lock is
    /// dropped. The maintenance thread handles timeout eviction independently.
    fn checkout_with_cold_cache(&self, prompt: &[u32], req: &RequestCtx) -> Result<SlotGuard<'_>> {
        let cache_mutex = self
            .session_cache
            .as_ref()
            .expect("cold checkout requires a session cache");
        let cfg: &Config = self.model.config();
        let mut pool = self.pool.lock().expect("pool poisoned");
        loop {
            let free = (0..pool.slots.len())
                .filter(|&index| !pool.slots[index].busy)
                .collect::<Vec<_>>();
            if free.is_empty() {
                pool = self.freed.wait(pool).expect("pool poisoned");
                continue;
            }
            let continuation = pick_continuation(
                free.iter().filter_map(|&index| {
                    pool.slots[index].kv.as_ref().and_then(|kv| {
                        kv.continuation_prefix_len(prompt)
                            .map(|prefix| (index, prefix, prefix))
                    })
                }),
                prompt.len(),
            );
            let resident_prefix = continuation
                .and_then(|index| {
                    pool.slots[index]
                        .kv
                        .as_ref()
                        .and_then(|kv| kv.continuation_prefix_len(prompt))
                })
                .unwrap_or(0);
            let target = continuation.unwrap_or_else(|| {
                *free
                    .iter()
                    .min_by_key(|&&index| {
                        let slot = &pool.slots[index];
                        let empty = slot.kv.as_ref().is_none_or(|kv| kv.cached_len() == 0);
                        (!empty, slot.tick)
                    })
                    .expect("free is non-empty")
            });

            pool.tick += 1;
            let tick = pool.tick;
            pool.slots[target].busy = true;
            pool.slots[target].tick = tick;
            pool.slots[target].idle_since = None;
            let mut target_kv = pool.slots[target].kv.take();
            drop(pool);

            {
                // One pass owns both the inference baton and the cache catalog. No other request
                // can submit against a state buffer while it is being downloaded or restored.
                let _gate = req.gate_pass();
                let mut cache = match cache_mutex.lock() {
                    Ok(cache) => cache,
                    Err(poisoned) => poisoned.into_inner(),
                };

                let cold_prefix = cache.best_continuation_len(prompt);
                let cold = (cold_prefix > resident_prefix && target_kv.is_some())
                    .then(|| cache.take_best_continuation(prompt))
                    .flatten();
                if let (Some(entry), Some(kv)) = (cold, target_kv.as_mut()) {
                    let mut released = false;
                    if kv.cached_len() != 0 {
                        match cache.spill(kv, self.vk.as_ref(), cfg) {
                            Ok(true) => released = true,
                            Ok(false) => {}
                            Err(error) => {
                                tracing::warn!(
                                    slot = target,
                                    "cold KV replacement spill failed; recycling the slot: {error}"
                                );
                                kv.reset();
                            }
                        }
                    }
                    if !released {
                        released = kv.release_session_state(self.vk.as_ref(), cfg).is_ok()
                            || kv.release_session_state(self.vk.as_ref(), cfg).is_ok();
                    }
                    if released {
                        match cache.restore(entry, kv, self.vk.as_ref(), cfg) {
                            Ok(()) => {}
                            Err(error) => tracing::warn!(
                                slot = target,
                                "cold KV restore failed; re-prefilling the request: {error}"
                            ),
                        }
                    } else {
                        tracing::warn!(
                            slot = target,
                            "could not release the target KV slot for cold restore; re-prefilling"
                        );
                        cache.return_entry(entry);
                        kv.reset();
                    }
                } else if continuation.is_none() {
                    if let Some(kv) = target_kv.as_mut().filter(|kv| kv.cached_len() != 0) {
                        if let Err(error) = cache.spill(kv, self.vk.as_ref(), cfg) {
                            tracing::warn!(
                                slot = target,
                                "cold KV replacement spill failed; forgetting the old conversation: {error}"
                            );
                            kv.reset();
                        }
                    }
                }
                if let Err(error) = cache.gc() {
                    tracing::warn!("cold KV cache maintenance failed: {error}");
                }
            }

            return Ok(SlotGuard {
                engine: self,
                idx: target,
                kv: target_kv,
                detached: false,
            });
        }
    }

    /// Take an LRU free slot without prefix seeding. Image payload identity is not represented by
    /// the repeated image-pad token ids, so token-prefix reuse would be unsound until slots carry
    /// an image fingerprint.
    fn checkout_fresh(&self, req: &RequestCtx) -> SlotGuard<'_> {
        if self.session_cache.is_some() {
            self.checkout_fresh_with_cold_cache(req)
        } else {
            self.checkout_fresh_resident()
        }
    }

    fn checkout_fresh_resident(&self) -> SlotGuard<'_> {
        let mut pool = self.pool.lock().expect("pool poisoned");
        loop {
            if let Some(target) = (0..pool.slots.len())
                .filter(|&index| !pool.slots[index].busy)
                .min_by_key(|&index| pool.slots[index].tick)
            {
                pool.tick += 1;
                let tick = pool.tick;
                pool.slots[target].busy = true;
                pool.slots[target].tick = tick;
                let kv = pool.slots[target].kv.take();
                return SlotGuard {
                    engine: self,
                    idx: target,
                    kv,
                    detached: false,
                };
            }
            pool = self.freed.wait(pool).expect("pool poisoned");
        }
    }

    fn checkout_fresh_with_cold_cache(&self, req: &RequestCtx) -> SlotGuard<'_> {
        let cache_mutex = self
            .session_cache
            .as_ref()
            .expect("cold checkout requires a session cache");
        let mut pool = self.pool.lock().expect("pool poisoned");
        loop {
            if let Some(target) = (0..pool.slots.len())
                .filter(|&index| !pool.slots[index].busy)
                .min_by_key(|&index| pool.slots[index].tick)
            {
                pool.tick += 1;
                let tick = pool.tick;
                pool.slots[target].busy = true;
                pool.slots[target].tick = tick;
                pool.slots[target].idle_since = None;
                let mut kv = pool.slots[target].kv.take();
                drop(pool);
                if let Some(kv) = kv.as_mut().filter(|kv| kv.cached_len() != 0) {
                    let _gate = req.gate_pass();
                    let mut cache = match cache_mutex.lock() {
                        Ok(cache) => cache,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    if let Err(error) = cache.spill(kv, self.vk.as_ref(), self.model.config()) {
                        tracing::warn!(
                            slot = target,
                            "cold KV spill before multimodal reuse failed; forgetting the old conversation: {error}"
                        );
                        kv.reset();
                    }
                    if let Err(error) = cache.gc() {
                        tracing::warn!("cold KV cache maintenance failed: {error}");
                    }
                }
                return SlotGuard {
                    engine: self,
                    idx: target,
                    kv,
                    detached: false,
                };
            }
            pool = self.freed.wait(pool).expect("pool poisoned");
        }
    }

    /// Render an OpenAI conversation through the model's own chat template.
    /// Return a KV whose request thread no longer owns it while a decode cohort is active.
    fn return_detached_slot(&self, index: usize, kv: SeamKv) {
        let mut pool = match self.pool.lock() {
            Ok(pool) => pool,
            Err(error) => error.into_inner(),
        };
        pool.tick += 1;
        let tick = pool.tick;
        let Some(slot) = pool.slots.get_mut(index) else {
            tracing::error!(slot = index, "decode batch returned an unknown KV slot");
            return;
        };
        if !slot.busy || slot.kv.is_some() {
            tracing::error!(
                slot = index,
                busy = slot.busy,
                has_kv = slot.kv.is_some(),
                "decode batch found an inconsistent detached KV slot"
            );
            return;
        }
        slot.kv = Some(kv);
        slot.busy = false;
        slot.tick = tick;
        if self.session_cache.is_some() {
            slot.idle_since = Some(Instant::now());
        }
        drop(pool);
        if self.session_cache.is_some() {
            self.freed.notify_all();
        } else {
            self.freed.notify_one();
        }
    }

    fn deliver_batch_state(&self, slot: usize, tx: SyncSender<BatchEvent>, event: BatchEvent) {
        if let Err(mpsc::SendError(event)) = tx.send(event) {
            if let Some(kv) = event.into_kv() {
                self.return_detached_slot(slot, kv);
            }
        }
    }

    fn fallback_batch_work(&self, mut work: BatchWork) {
        let kv = work.kv.take().expect("queued decode work owns a KV slot");
        let Some(channels) = work.channels.take() else {
            self.return_detached_slot(work.slot, kv);
            return;
        };
        let remaining = work.max_new.saturating_sub(work.generated);
        self.deliver_batch_state(
            work.slot,
            channels.events,
            BatchEvent::Fallback {
                kv,
                prompt: work.prompt,
                max_new: remaining,
                stats: work.stats,
                prefilled: work.prefilled,
                turn_checkpoint: work.turn_checkpoint,
            },
        );
    }

    fn fail_batch_work(&self, mut work: BatchWork, error: &str) {
        let kv = work.kv.take().expect("active decode work owns a KV slot");
        let Some(channels) = work.channels.take() else {
            self.return_detached_slot(work.slot, kv);
            return;
        };
        self.deliver_batch_state(
            work.slot,
            channels.events,
            BatchEvent::Failed {
                kv,
                error: error.to_owned(),
            },
        );
    }

    fn complete_batch_work(&self, mut work: BatchWork) {
        let kv = work
            .kv
            .take()
            .expect("completed decode work owns a KV slot");
        let Some(channels) = work.channels.take() else {
            self.return_detached_slot(work.slot, kv);
            return;
        };
        self.deliver_batch_state(
            work.slot,
            channels.events,
            BatchEvent::Complete {
                kv,
                stats: work.stats,
            },
        );
    }

    /// End this cohort and release requests that arrived after its depth was fixed.
    fn close_decode_batch(&self) {
        let waiting = {
            let mut queue = self
                .decode_batch
                .lock()
                .expect("decode batch queue poisoned");
            queue.running = false;
            let mut waiting = queue.waiting.drain(..).collect::<Vec<_>>();
            waiting.extend(queue.prefill_waiting.drain(..));
            self.batch_interrupt.store(false, Ordering::Release);
            waiting
        };
        for work in waiting {
            self.fallback_batch_work(work);
        }
    }

    fn take_pending_batch_work(&self, active: &[BatchWork]) -> Vec<BatchWork> {
        if active.is_empty() || !self.batch_interrupt.swap(false, Ordering::AcqRel) {
            return Vec::new();
        }
        let sparse = self.qsa_sparse_at(active[0].prompt.len());
        let (accepted, fallback) = {
            let mut queue = self
                .decode_batch
                .lock()
                .expect("decode batch queue poisoned");
            let mut candidates = queue.waiting.drain(..).collect::<Vec<_>>();
            candidates.extend(queue.prefill_waiting.drain(..));
            let mut accepted = Vec::new();
            let mut fallback = Vec::new();
            for work in candidates {
                if active.len() + accepted.len() < MAX_DECODE_BATCH
                    && (!work.prefilled || self.qsa_sparse_at(work.prompt.len()) == sparse)
                {
                    accepted.push(work);
                } else {
                    fallback.push(work);
                }
            }
            (accepted, fallback)
        };
        if !accepted.is_empty() {
            tracing::debug!(
                lanes = accepted.len(),
                prefill_lanes = accepted.iter().filter(|work| !work.prefilled).count(),
                "admitting work into the active generation cohort"
            );
        }
        for work in fallback {
            self.fallback_batch_work(work);
        }
        accepted
    }

    fn batch_frontier_ready(&self, kv: &SeamKv, prompt: &[u32]) -> bool {
        self.model.config().qwen4exp && kv.cached_len() + 1 == prompt.len()
    }

    fn batch_qsa_sparse(&self, kv: &SeamKv) -> bool {
        self.qsa_sparse_at(kv.cached_len() + 1)
    }

    fn qsa_sparse_at(&self, visible_tokens: usize) -> bool {
        visible_tokens > self.qsa_threshold()
    }

    fn qsa_threshold(&self) -> usize {
        let cfg = self.model.config();
        let ratio = cfg
            .compress_ratios
            .iter()
            .copied()
            .max()
            .unwrap_or(4)
            .max(1);
        cfg.indexer_top_k + ratio - 1
    }

    fn retain_decode_mode(&self, active: Vec<BatchWork>) -> Vec<BatchWork> {
        let Some(first) = active.first() else {
            return active;
        };
        let sparse = self.qsa_sparse_at(first.prompt.len());
        let mut compatible = Vec::with_capacity(active.len());
        let mut fallback = Vec::new();
        for (lane, work) in active.into_iter().enumerate() {
            if lane == 0 || self.qsa_sparse_at(work.prompt.len()) == sparse {
                compatible.push(work);
            } else {
                fallback.push(work);
            }
        }
        if !fallback.is_empty() {
            tracing::debug!(
                kept = compatible.len(),
                deferred = fallback.len(),
                sparse,
                "split generation cohort at the QSA mode boundary"
            );
        }
        for work in fallback {
            self.fallback_batch_work(work);
        }
        compatible
    }

    fn decode_steps_before_qsa_boundary(&self, active: &[BatchWork]) -> usize {
        let threshold = self.qsa_threshold();
        active
            .iter()
            .filter(|work| !self.qsa_sparse_at(work.prompt.len()))
            .map(|work| {
                threshold
                    .saturating_sub(work.prompt.len())
                    .saturating_add(1)
            })
            .min()
            .unwrap_or(usize::MAX)
    }

    #[allow(clippy::too_many_arguments)]
    fn continue_after_prefill<F: FnMut(&str)>(
        &self,
        guard: &mut SlotGuard<'_>,
        prompt: &[u32],
        max_new: usize,
        req: &RequestCtx,
        acc: &mut Vec<u32>,
        printed: &mut usize,
        on_piece: &mut F,
        stats: GenStats,
        prompt_accounted: bool,
        turn_checkpoint: Option<crate::seam::TurnCheckpoint>,
    ) -> Result<GenStats> {
        if max_new == 0 || crate::sampling::abort_requested(Some(req)) {
            return Ok(stats);
        }
        let (_, tail) = crate::seam::generate_dense_vulkan_session(
            self.vk.as_ref(),
            self.model.gguf(),
            self.model.config(),
            self.model.engine_cfg(),
            self.model.embd(),
            self.model.per_layer_embd(),
            prompt,
            max_new,
            |id| crate::stream_token(self.model.tokenizer(), acc, printed, id, on_piece),
            &mut guard.kv,
            self.max_ctx,
            turn_checkpoint,
            None,
            Some(req),
            None,
        )?;
        if prompt_accounted {
            Ok(GenStats {
                n_gen: stats.n_gen.saturating_add(tail.n_gen),
                decode_secs: stats.decode_secs + tail.decode_secs,
                ..stats
            })
        } else {
            Ok(merge_stats(stats, tail))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn wait_for_decode_batch<F: FnMut(&str)>(
        &self,
        guard: &mut SlotGuard<'_>,
        events: Receiver<BatchEvent>,
        acknowledgements: SyncSender<bool>,
        req: &RequestCtx,
        acc: &mut Vec<u32>,
        printed: &mut usize,
        on_piece: &mut F,
    ) -> Result<GenStats> {
        loop {
            let event = match events.recv() {
                Ok(event) => event,
                Err(_) => {
                    guard.abandon_detached();
                    return Err(anyhow!(
                        "parallel decode worker stopped before returning its KV"
                    ));
                }
            };
            match event {
                BatchEvent::Token { id, progress } => {
                    req.report_progress(progress);
                    let keep_going = if crate::sampling::abort_requested(Some(req)) {
                        false
                    } else {
                        crate::stream_token(self.model.tokenizer(), acc, printed, id, on_piece);
                        !crate::sampling::abort_requested(Some(req))
                    };
                    let _ = acknowledgements.send(keep_going);
                }
                BatchEvent::Complete { kv, stats } => {
                    guard.reattach(kv);
                    return Ok(stats);
                }
                BatchEvent::Fallback {
                    kv,
                    prompt,
                    max_new,
                    stats,
                    prefilled,
                    turn_checkpoint,
                } => {
                    guard.reattach(kv);
                    return self.continue_after_prefill(
                        guard,
                        &prompt,
                        max_new,
                        req,
                        acc,
                        printed,
                        on_piece,
                        stats,
                        prefilled,
                        turn_checkpoint,
                    );
                }
                BatchEvent::Failed { kv, error } => {
                    guard.reattach(kv);
                    return Err(anyhow!(error));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_prefill_batch(
        &self,
        leader: BatchWork,
        peers: Vec<BatchWork>,
        leader_guard: &mut SlotGuard<'_>,
        req: &RequestCtx,
        leader_on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenStats> {
        let mut active = Vec::with_capacity(1 + peers.len());
        active.push(leader);
        active.extend(peers);
        let (active, leader_result) =
            self.prepare_prefill_work(active, leader_guard, req, leader_on_token, None)?;
        if active.is_empty() {
            return leader_result
                .ok_or_else(|| anyhow!("mixed prefill completed without a leader result"));
        }
        self.run_decode_work(active, leader_guard, req, leader_on_token, leader_result)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_prefill_work(
        &self,
        mut active: Vec<BatchWork>,
        leader_guard: &mut SlotGuard<'_>,
        req: &RequestCtx,
        leader_on_token: &mut dyn FnMut(u32) -> bool,
        mut leader_result: Option<GenStats>,
    ) -> Result<(Vec<BatchWork>, Option<GenStats>)> {
        let prompts = active
            .iter()
            .map(|work| work.prompt.clone())
            .collect::<Vec<_>>();
        let turn_checkpoints = active
            .iter()
            .map(|work| work.turn_checkpoint)
            .collect::<Vec<_>>();
        let mut primary = active[0].kv.take();
        let mut peer_kv = active
            .iter_mut()
            .skip(1)
            .map(|work| work.kv.take().expect("prefill lane owns a KV slot"))
            .collect::<Vec<_>>();
        let mut samplers = active
            .iter_mut()
            .map(|work| work.sampler.take().expect("prefill lane owns a sampler"))
            .collect::<Vec<_>>();
        let prefilled = crate::seam::generate_dense_vulkan_parallel_prefill_session(
            self.vk.as_ref(),
            self.model.gguf(),
            self.model.config(),
            self.model.engine_cfg(),
            self.model.embd(),
            self.model.per_layer_embd(),
            &prompts,
            &mut primary,
            &mut peer_kv,
            self.max_ctx,
            &turn_checkpoints,
            &mut samplers,
            Some(req),
        );

        active[0].kv = primary;
        for (work, kv) in active.iter_mut().skip(1).zip(peer_kv) {
            work.kv = Some(kv);
        }
        for (work, sampler) in active.iter_mut().zip(samplers) {
            work.sampler = Some(sampler);
        }

        let (prefill_stats, mixed_outputs) = match prefilled {
            Ok((stats, outputs))
                if stats.len() == active.len() && outputs.len() == active.len() =>
            {
                (stats, outputs)
            }
            Ok((stats, outputs)) => {
                let message = format!(
                    "parallel prefill returned {} stats and {} output lanes for {} active lanes",
                    stats.len(),
                    outputs.len(),
                    active.len()
                );
                for mut work in active {
                    work.kv
                        .as_mut()
                        .expect("invalid prefill lane owns a KV slot")
                        .reset();
                    if work.channels.is_none() {
                        leader_guard.reattach(
                            work.kv
                                .take()
                                .expect("invalid prefill leader owns a KV slot"),
                        );
                    } else {
                        self.fail_batch_work(work, &message);
                    }
                }
                return Err(anyhow!(message));
            }
            Err(error) => {
                let message = error.to_string();
                for mut work in active {
                    if work.channels.is_none() {
                        leader_guard.reattach(
                            work.kv
                                .take()
                                .expect("failed prefill leader owns a KV slot"),
                        );
                    } else {
                        self.fail_batch_work(work, &message);
                    }
                }
                return Err(error);
            }
        };

        for (work, mut stats) in active.iter_mut().zip(prefill_stats) {
            if work.prefilled {
                stats.n_prompt = 0;
                stats.n_cached = 0;
                stats.prompt_secs = 0.0;
            }
            work.stats = merge_stats(work.stats, stats);
            work.prefilled = true;
            work.turn_checkpoint = None;
        }
        let cfg = self.model.config();
        let mut next = Vec::with_capacity(active.len());
        for (mut work, output) in active.into_iter().zip(mixed_outputs) {
            let Some(token) = output else {
                next.push(work);
                continue;
            };
            let request_prompt_tokens = work.prompt.len().saturating_sub(work.generated);
            work.generated += 1;
            let progress = self.batch_decode_progress(
                request_prompt_tokens,
                work.stats.n_cached,
                work.generated,
            );
            let eos = !self.model.engine_cfg().sampling.ignore_eos
                && (cfg.eos_ids.contains(&token) || token == cfg.eos);
            let accepted = if eos {
                false
            } else if let Some(channels) = work.channels.as_mut() {
                channels
                    .events
                    .send(BatchEvent::Token {
                        id: token,
                        progress,
                    })
                    .is_ok()
                    && channels.acknowledgements.recv().unwrap_or(false)
            } else {
                req.report_progress(progress);
                leader_on_token(token)
            };
            if eos || !accepted || work.generated >= work.max_new {
                if work.channels.is_none() {
                    leader_guard.reattach(
                        work.kv
                            .take()
                            .expect("completed mixed leader owns a KV slot"),
                    );
                    leader_result = Some(work.stats);
                } else {
                    self.complete_batch_work(work);
                }
            } else {
                work.prompt.push(token);
                next.push(work);
            }
        }
        let active = next;
        if active.is_empty() {
            return Ok((active, leader_result));
        }
        if active.iter().any(|work| {
            !work
                .kv
                .as_ref()
                .is_some_and(|kv| self.batch_frontier_ready(kv, &work.prompt))
        }) {
            let message = "parallel prefill stopped before every lane reached its decode frontier";
            for mut work in active {
                work.kv
                    .as_mut()
                    .expect("incomplete prefill lane owns a KV slot")
                    .reset();
                if work.channels.is_none() {
                    leader_guard.reattach(
                        work.kv
                            .take()
                            .expect("incomplete prefill leader owns a KV slot"),
                    );
                } else {
                    self.fail_batch_work(work, message);
                }
            }
            return Err(anyhow!(message));
        }

        Ok((active, leader_result))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_decode_batch(
        &self,
        leader: BatchWork,
        peers: Vec<BatchWork>,
        leader_guard: &mut SlotGuard<'_>,
        req: &RequestCtx,
        leader_on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenStats> {
        let mut active = Vec::with_capacity(1 + peers.len());
        active.push(leader);
        active.extend(peers);
        self.run_decode_work(active, leader_guard, req, leader_on_token, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_decode_work(
        &self,
        mut active: Vec<BatchWork>,
        leader_guard: &mut SlotGuard<'_>,
        req: &RequestCtx,
        leader_on_token: &mut dyn FnMut(u32) -> bool,
        mut leader_result: Option<GenStats>,
    ) -> Result<GenStats> {
        while !active.is_empty() {
            active = self.retain_decode_mode(active);
            let pending = self.take_pending_batch_work(&active);
            if !pending.is_empty() {
                active.extend(pending);
                (active, leader_result) = self.prepare_prefill_work(
                    active,
                    leader_guard,
                    req,
                    leader_on_token,
                    leader_result,
                )?;
                if active.is_empty() {
                    return leader_result.ok_or_else(|| {
                        anyhow!("dynamic decode cohort completed without a leader result")
                    });
                }
                continue;
            }
            let steps = active
                .iter()
                .map(|work| work.max_new.saturating_sub(work.generated))
                .min()
                .unwrap_or(0);
            let steps = steps.min(self.decode_steps_before_qsa_boundary(&active));
            if steps == 0 {
                return Err(anyhow!("parallel decode cohort contains exhausted work"));
            }
            let lanes = active.len();
            let prompts = active
                .iter()
                .map(|work| work.prompt.clone())
                .collect::<Vec<_>>();
            let mut primary = active[0].kv.take();
            let mut peer_kv = active
                .iter_mut()
                .skip(1)
                .map(|work| work.kv.take().expect("active lane owns a KV slot"))
                .collect::<Vec<_>>();
            let mut samplers = active
                .iter_mut()
                .map(|work| work.sampler.take().expect("active lane owns a sampler"))
                .collect::<Vec<_>>();
            let mut lane_io = active
                .iter_mut()
                .map(|work| work.channels.take())
                .collect::<Vec<_>>();
            let progress_bases = active
                .iter()
                .map(|work| {
                    (
                        work.prompt.len().saturating_sub(work.generated),
                        work.stats.n_cached,
                        work.generated,
                    )
                })
                .collect::<Vec<_>>();
            let mut streamed = vec![0usize; lanes];
            let mut accepted = vec![true; lanes];
            let mut stream = |lane: usize, id: u32| {
                streamed[lane] += 1;
                let (prompt_tokens, cached_prompt_tokens, generated) = progress_bases[lane];
                let progress = self.batch_decode_progress(
                    prompt_tokens,
                    cached_prompt_tokens,
                    generated.saturating_add(streamed[lane]),
                );
                let keep_going = match lane_io.get_mut(lane) {
                    Some(None) => {
                        req.report_progress(progress);
                        leader_on_token(id)
                    }
                    Some(Some(channels)) => {
                        channels
                            .events
                            .send(BatchEvent::Token { id, progress })
                            .is_ok()
                            && channels.acknowledgements.recv().unwrap_or(false)
                    }
                    None => false,
                };
                if let Some(accepted) = accepted.get_mut(lane) {
                    *accepted = keep_going;
                }
                keep_going
            };
            let decoded = crate::seam::generate_dense_vulkan_parallel_sampled_session(
                self.vk.as_ref(),
                self.model.gguf(),
                self.model.config(),
                self.model.engine_cfg(),
                self.model.embd(),
                self.model.per_layer_embd(),
                &prompts,
                steps,
                &mut primary,
                &mut peer_kv,
                self.max_ctx,
                &mut samplers,
                &mut stream,
                Some(&self.batch_interrupt),
                Some(req),
            );

            active[0].kv = primary;
            for (work, kv) in active.iter_mut().skip(1).zip(peer_kv) {
                work.kv = Some(kv);
            }
            for (work, sampler) in active.iter_mut().zip(samplers) {
                work.sampler = Some(sampler);
            }
            for (work, io) in active.iter_mut().zip(lane_io) {
                if let Some(channels) = io {
                    work.channels = Some(channels);
                }
            }

            let (outputs, batch_stats) = match decoded {
                Ok(decoded) => decoded,
                Err(error) => {
                    let message = error.to_string();
                    for mut work in std::mem::take(&mut active) {
                        if work.channels.is_none() {
                            leader_guard
                                .reattach(work.kv.take().expect("failed leader owns a KV slot"));
                        } else {
                            self.fail_batch_work(work, &message);
                        }
                    }
                    if let Some(stats) = leader_result {
                        tracing::warn!(%error, "decode batch failed after the leader completed");
                        return Ok(stats);
                    }
                    return Err(error);
                }
            };
            if outputs.len() != lanes || outputs.iter().any(Vec::is_empty) {
                let message = format!(
                    "parallel decode returned {} output lanes for {lanes} active lanes",
                    outputs.len()
                );
                for mut work in std::mem::take(&mut active) {
                    if work.channels.is_none() {
                        leader_guard.reattach(
                            work.kv
                                .take()
                                .expect("invalid leader output owns a KV slot"),
                        );
                    } else {
                        self.fail_batch_work(work, &message);
                    }
                }
                return Err(anyhow!(message));
            }

            let cfg = self.model.config();
            let mut next = Vec::with_capacity(lanes);
            for (lane, (mut work, output)) in std::mem::take(&mut active)
                .into_iter()
                .zip(outputs)
                .enumerate()
            {
                work.generated += output.len();
                work.stats.n_gen += output.len();
                work.stats.decode_secs += batch_stats.decode_secs;
                work.prompt.extend_from_slice(&output);
                let eos = output.last().is_some_and(|token| {
                    !self.model.engine_cfg().sampling.ignore_eos
                        && (cfg.eos_ids.contains(token) || *token == cfg.eos)
                });
                let done = eos || !accepted[lane] || work.generated >= work.max_new;
                if done {
                    if work.channels.is_none() {
                        leader_guard
                            .reattach(work.kv.take().expect("completed leader owns a KV slot"));
                        leader_result = Some(work.stats);
                    } else {
                        self.complete_batch_work(work);
                    }
                } else {
                    next.push(work);
                }
            }
            active = next;
        }

        leader_result.ok_or_else(|| anyhow!("parallel decode completed without a leader result"))
    }

    pub fn render_chat_messages(&self, messages: &[(&str, &str)]) -> Result<String> {
        self.model.render_chat_messages(messages)
    }

    /// Generate one sequence: check out a slot, run the ordinary seam decode on it (taking turns on
    /// the GPU baton at every step), and return the slot. `&self` — N of these run concurrently.
    pub fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: &RequestCtx,
        on_piece: impl FnMut(&str),
    ) -> Result<GenStats> {
        self.generate_turn(prompt, None, max_new, constraint, req, on_piece)
    }

    /// [`generate`](Self::generate) with a stable rendered-history prefix for recurrent state
    /// checkpointing. Attention-only models ignore the resulting boundary in the runner.
    pub fn generate_turn(
        &self,
        prompt: &str,
        stable_prefix: Option<&str>,
        max_new: usize,
        constraint: Option<&mut crate::grammar::Constraint>,
        req: &RequestCtx,
        mut on_piece: impl FnMut(&str),
    ) -> Result<GenStats> {
        let prompt_tokens = self.model.encode(prompt)?;
        let turn_checkpoint = self.model.turn_checkpoint(&prompt_tokens, stable_prefix)?;
        // Cap the reply to the context actually left in THIS slot (a per-slot ctx is smaller than
        // the model's trained window under `-np N`), mirroring the sequential session path: a
        // generation ceiling is a default, not a demand. An over-long PROMPT still errors cleanly
        // in the runner.
        let max_new = max_new.min(self.max_ctx.saturating_sub(prompt_tokens.len() + 1));
        let batch_candidate = self.model.config().qwen4exp
            && constraint.is_none()
            && self.gate.is_some()
            && max_new > 0;
        // Register before checkout: a peer already at the frontier should know that another
        // eligible request is on its way through restore/prefill.
        let mut batch_prefill = batch_candidate.then(|| BatchPrefillRegistration::new(self));
        let mut guard = self.checkout(&prompt_tokens, req)?;
        let mut acc: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        // Resolve `ubatch_rows`/`kv_auto_q8` against THIS engine's pins for the decode (warm calls
        // must agree with the buffers placement sized). Concurrent requests share the one cell.
        let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
        if batch_candidate {
            let sampler = ParallelSampler::new(req, &self.model.engine_cfg().sampling);
            let mut queue = self
                .decode_batch
                .lock()
                .expect("decode batch queue poisoned");
            if queue.running {
                let (event_tx, event_rx) = mpsc::sync_channel(0);
                let (ack_tx, ack_rx) = mpsc::sync_channel(0);
                let slot = guard.idx;
                let kv = guard.detach();
                queue.prefill_waiting.push_back(BatchWork {
                    slot,
                    kv: Some(kv),
                    prompt: prompt_tokens,
                    max_new,
                    generated: 0,
                    stats: GenStats::default(),
                    prefilled: false,
                    turn_checkpoint,
                    sampler: Some(sampler),
                    channels: Some(BatchChannels {
                        events: event_tx,
                        acknowledgements: ack_rx,
                    }),
                });
                self.batch_interrupt.store(true, Ordering::Release);
                self.decode_ready.notify_all();
                drop(queue);
                drop(batch_prefill.take());
                return self.wait_for_decode_batch(
                    &mut guard,
                    event_rx,
                    ack_tx,
                    req,
                    &mut acc,
                    &mut printed,
                    &mut on_piece,
                );
            }

            queue.running = true;
            let (mut queue, _) = self
                .decode_ready
                .wait_timeout_while(queue, PREFILL_BATCH_WAIT, |queue| {
                    queue.waiting.is_empty() && queue.prefill_waiting.is_empty()
                })
                .expect("decode batch queue poisoned");
            let leader_sparse = self.qsa_sparse_at(prompt_tokens.len());
            let (peers, fallback) = {
                let mut peers = Vec::new();
                let mut fallback = Vec::new();
                let mut candidates = queue.waiting.drain(..).collect::<Vec<_>>();
                candidates.extend(queue.prefill_waiting.drain(..));
                for work in candidates {
                    if peers.len() + 1 < MAX_DECODE_BATCH
                        && (!work.prefilled
                            || self.qsa_sparse_at(work.prompt.len()) == leader_sparse)
                        && work.kv.as_ref().is_some_and(|kv| {
                            !work.prefilled || self.batch_frontier_ready(kv, &work.prompt)
                        })
                    {
                        peers.push(work);
                    } else {
                        fallback.push(work);
                    }
                }
                if peers.is_empty() || crate::sampling::abort_requested(Some(req)) {
                    queue.running = false;
                } else {
                    batch_prefill
                        .as_mut()
                        .expect("batch candidate registered before cohort collection")
                        .arrive(&mut queue);
                }
                self.batch_interrupt.store(false, Ordering::Release);
                (peers, fallback)
            };
            drop(queue);
            for work in fallback {
                self.fallback_batch_work(work);
            }

            if !peers.is_empty() && !crate::sampling::abort_requested(Some(req)) {
                drop(batch_prefill.take());
                tracing::debug!(
                    lanes = peers.len() + 1,
                    "collected layer-synchronous prefill cohort"
                );
                let leader = BatchWork {
                    slot: guard.idx,
                    kv: Some(guard.detach()),
                    prompt: prompt_tokens,
                    max_new,
                    generated: 0,
                    stats: GenStats::default(),
                    prefilled: false,
                    turn_checkpoint,
                    sampler: Some(sampler),
                    channels: None,
                };
                let mut leader_stream = |id| {
                    if crate::sampling::abort_requested(Some(req)) {
                        return false;
                    }
                    crate::stream_token(
                        self.model.tokenizer(),
                        &mut acc,
                        &mut printed,
                        id,
                        &mut on_piece,
                    );
                    !crate::sampling::abort_requested(Some(req))
                };
                let result =
                    self.run_prefill_batch(leader, peers, &mut guard, req, &mut leader_stream);
                self.close_decode_batch();
                return result;
            }
            for work in peers {
                self.fallback_batch_work(work);
            }
        }
        if !batch_candidate {
            let (_ids, stats) = crate::seam::generate_dense_vulkan_session(
                self.vk.as_ref(),
                self.model.gguf(),
                self.model.config(),
                self.model.engine_cfg(),
                self.model.embd(),
                self.model.per_layer_embd(),
                &prompt_tokens,
                max_new,
                |id| {
                    crate::stream_token(
                        self.model.tokenizer(),
                        &mut acc,
                        &mut printed,
                        id,
                        &mut on_piece,
                    )
                },
                &mut guard.kv,
                self.max_ctx,
                turn_checkpoint,
                constraint,
                Some(req),
                None,
            )?;
            return Ok(stats);
        }

        // No cohort was ready during the short admission window. Establish this prompt's frontier
        // independently, then either join the active cohort or lead a new one. An active cohort can
        // also admit this request before this point through `prefill_waiting` above.
        let (_ids, prefill_stats) = crate::seam::generate_dense_vulkan_session(
            self.vk.as_ref(),
            self.model.gguf(),
            self.model.config(),
            self.model.engine_cfg(),
            self.model.embd(),
            self.model.per_layer_embd(),
            &prompt_tokens,
            0,
            |_| {},
            &mut guard.kv,
            self.max_ctx,
            turn_checkpoint,
            None,
            Some(req),
            None,
        )?;
        if crate::sampling::abort_requested(Some(req))
            || !guard
                .kv
                .as_ref()
                .is_some_and(|kv| self.batch_frontier_ready(kv, &prompt_tokens))
        {
            drop(batch_prefill.take());
            return self.continue_after_prefill(
                &mut guard,
                &prompt_tokens,
                max_new,
                req,
                &mut acc,
                &mut printed,
                &mut on_piece,
                prefill_stats,
                false,
                None,
            );
        }

        let sampler = ParallelSampler::new(req, &self.model.engine_cfg().sampling);
        let mut queue = self
            .decode_batch
            .lock()
            .expect("decode batch queue poisoned");
        batch_prefill
            .as_mut()
            .expect("batch candidate registered before prefill")
            .arrive(&mut queue);
        drop(batch_prefill.take());
        if queue.running {
            let (event_tx, event_rx) = mpsc::sync_channel(0);
            let (ack_tx, ack_rx) = mpsc::sync_channel(0);
            let slot = guard.idx;
            let kv = guard.detach();
            queue.waiting.push_back(BatchWork {
                slot,
                kv: Some(kv),
                prompt: prompt_tokens,
                max_new,
                generated: 0,
                stats: prefill_stats,
                prefilled: true,
                turn_checkpoint: None,
                sampler: Some(sampler),
                channels: Some(BatchChannels {
                    events: event_tx,
                    acknowledgements: ack_rx,
                }),
            });
            self.batch_interrupt.store(true, Ordering::Release);
            self.decode_ready.notify_all();
            drop(queue);
            return self.wait_for_decode_batch(
                &mut guard,
                event_rx,
                ack_tx,
                req,
                &mut acc,
                &mut printed,
                &mut on_piece,
            );
        }

        // The first ready request leads this cohort. It only waits while registered work is truly
        // still reaching the frontier, so a lone request pays no unconditional batching delay.
        queue.running = true;
        let (mut queue, _) = self
            .decode_ready
            .wait_timeout_while(queue, DECODE_BATCH_WAIT, |queue| {
                should_wait_for_decode_peers(
                    queue.prefilling,
                    queue.waiting.len() + queue.prefill_waiting.len(),
                )
            })
            .expect("decode batch queue poisoned");

        let leader_kv = guard.kv.as_ref().expect("checked-out leader has a KV slot");
        let depth = leader_kv.cached_len();
        let leader_sparse = self.batch_qsa_sparse(leader_kv);
        let (peers, fallback) = {
            let mut peers = Vec::new();
            let mut fallback = Vec::new();
            let mut candidates = queue.waiting.drain(..).collect::<Vec<_>>();
            candidates.extend(queue.prefill_waiting.drain(..));
            for work in candidates {
                let compatible = peers.len() + 1 < MAX_DECODE_BATCH
                    && work.kv.as_ref().is_some_and(|kv| {
                        (!work.prefilled || self.batch_frontier_ready(kv, &work.prompt))
                            && (!work.prefilled
                                || self.qsa_sparse_at(work.prompt.len()) == leader_sparse)
                    });
                if compatible {
                    peers.push(work);
                } else {
                    fallback.push(work);
                }
            }
            self.batch_interrupt.store(false, Ordering::Release);
            (peers, fallback)
        };
        drop(queue);
        for work in fallback {
            self.fallback_batch_work(work);
        }
        let needs_prefill = peers.iter().any(|work| !work.prefilled);
        tracing::debug!(
            depth,
            lanes = peers.len() + 1,
            needs_prefill,
            "collected parallel generation cohort"
        );
        if crate::sampling::abort_requested(Some(req)) {
            self.close_decode_batch();
            for work in peers {
                self.fallback_batch_work(work);
            }
            return self.continue_after_prefill(
                &mut guard,
                &prompt_tokens,
                max_new,
                req,
                &mut acc,
                &mut printed,
                &mut on_piece,
                prefill_stats,
                false,
                None,
            );
        }

        let leader = BatchWork {
            slot: guard.idx,
            kv: Some(guard.detach()),
            prompt: prompt_tokens,
            max_new,
            generated: 0,
            stats: prefill_stats,
            prefilled: true,
            turn_checkpoint: None,
            sampler: Some(sampler),
            channels: None,
        };
        let mut leader_stream = |id| {
            if crate::sampling::abort_requested(Some(req)) {
                return false;
            }
            crate::stream_token(
                self.model.tokenizer(),
                &mut acc,
                &mut printed,
                id,
                &mut on_piece,
            );
            !crate::sampling::abort_requested(Some(req))
        };
        let result = if needs_prefill {
            self.run_prefill_batch(leader, peers, &mut guard, req, &mut leader_stream)
        } else {
            self.run_decode_batch(leader, peers, &mut guard, req, &mut leader_stream)
        };
        self.close_decode_batch();
        result
    }

    /// Generate one Qwen3.8 vision turn. Projector execution is completed by the caller before
    /// entering here, so its request-scoped weights have already returned to the unified arena.
    pub fn generate_multimodal_turn(
        &self,
        prompt: &str,
        images: Vec<MultimodalEmbedding>,
        max_new: usize,
        req: &RequestCtx,
        mut on_piece: impl FnMut(&str),
    ) -> Result<GenStats> {
        if !self.model.config().qwen4exp {
            return Err(anyhow!(
                "vision text integration currently supports qwen4exp models only"
            ));
        }
        if images.is_empty() {
            return self.generate_turn(prompt, None, max_new, None, req, on_piece);
        }
        let image_pad_id = self
            .model
            .tokenizer()
            .token_to_id("<|image_pad|>")
            .ok_or_else(|| anyhow!("model tokenizer has no <|image_pad|> token"))?;
        let base_tokens = self.model.encode(prompt)?;
        let image_count = images.len();
        let (prompt_tokens, plan) = expand_multimodal_prompt(
            &base_tokens,
            image_pad_id,
            images,
            self.model.config().n_embd,
        )?;
        if prompt_tokens.len().saturating_add(1) > self.max_ctx {
            return Err(anyhow!(
                "multimodal prompt expands to {} tokens, exceeding this slot's {}-token context",
                prompt_tokens.len(),
                self.max_ctx
            ));
        }
        let mut guard = self.checkout_fresh(req);
        if let Some(kv) = guard.kv.as_mut() {
            kv.reset();
        }
        let max_new = max_new.min(self.max_ctx.saturating_sub(prompt_tokens.len() + 1));
        let mut acc = Vec::new();
        let mut printed = 0usize;
        let _scope = crate::seam::PlacementScope::enter(self.pins.clone());
        tracing::info!(
            images = image_count,
            base_tokens = base_tokens.len(),
            expanded_tokens = prompt_tokens.len(),
            decode_base = plan.decode_base,
            "multimodal prompt prepared"
        );
        let result = crate::seam::generate_dense_vulkan_session(
            &self.vk,
            self.model.gguf(),
            self.model.config(),
            self.model.engine_cfg(),
            self.model.embd(),
            self.model.per_layer_embd(),
            &prompt_tokens,
            max_new,
            |id| {
                crate::stream_token(
                    self.model.tokenizer(),
                    &mut acc,
                    &mut printed,
                    id,
                    &mut on_piece,
                )
            },
            &mut guard.kv,
            self.max_ctx,
            None,
            None,
            Some(req),
            Some(&plan),
        );
        // Do not expose image-backed KV to ordinary token-only prefix matching. A later revision
        // can retain it once slot keys include deterministic image fingerprints.
        if let Some(kv) = guard.kv.as_mut() {
            kv.reset();
        }
        let (_, stats) = result?;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        expand_multimodal_prompt, pick_continuation, should_wait_for_decode_peers,
        MultimodalEmbedding, MAX_DECODE_BATCH,
    };

    #[test]
    fn decode_cohort_waits_only_for_registered_capacity() {
        assert!(!should_wait_for_decode_peers(0, 0));
        assert!(should_wait_for_decode_peers(1, 0));
        assert!(!should_wait_for_decode_peers(1, MAX_DECODE_BATCH - 1));
    }

    #[test]
    fn continuation_picks_longest_prefix_not_first() {
        // Three free slots, prompt_len = 100. Slots 0 and 2 both qualify (prompt extends their
        // cache: score == cached_len); slot 2 has the LONGER reusable prefix, so it must win even
        // though slot 0 appears first. Slot 1 is a different conversation (score below its cache).
        let candidates = [
            (0usize, 20usize, 20usize), // extends: score 20 == cached 20
            (1, 5, 40),                 // no: 5 != 40 and 5 != 100
            (2, 60, 60),                // extends: score 60 == cached 60 (longest)
        ];
        assert_eq!(pick_continuation(candidates, 100), Some(2));
    }

    #[test]
    fn continuation_accepts_exact_equal_prompt() {
        // score == prompt_len (the prompt EQUALS the cache) qualifies even when cached_len differs.
        let candidates = [(7usize, 30usize, 50usize)];
        assert_eq!(pick_continuation(candidates, 30), Some(7));
    }

    #[test]
    fn continuation_none_when_no_slot_qualifies() {
        // A partial-but-diverged prefix (score < cached_len and < prompt_len) does not continue.
        let candidates = [(0usize, 10usize, 40usize), (1, 0, 0)];
        assert_eq!(pick_continuation(candidates, 100), None);
        // Empty candidate set.
        assert_eq!(pick_continuation(std::iter::empty(), 100), None);
    }

    #[test]
    fn multimodal_expansion_preserves_order_and_grid_positions() {
        let images = vec![
            MultimodalEmbedding {
                values: vec![1.0; 2 * 2 * 3],
                grid_nx: 2,
                grid_ny: 2,
            },
            MultimodalEmbedding {
                values: vec![2.0; 3 * 3],
                grid_nx: 3,
                grid_ny: 1,
            },
        ];
        let (tokens, plan) = expand_multimodal_prompt(&[10, 99, 11, 99, 12], 99, images, 3)
            .expect("valid synthetic multimodal prompt");
        assert_eq!(tokens, [10, 99, 99, 99, 99, 11, 99, 99, 99, 12]);
        assert_eq!(plan.spans[0].start, 1);
        assert_eq!(plan.spans[0].n_tokens, 4);
        assert_eq!(plan.spans[1].start, 6);
        assert_eq!(plan.spans[1].n_tokens, 3);
        assert_eq!(
            &plan.prompt_pos4[4..20],
            &[1, 1, 1, 0, 1, 1, 2, 0, 1, 2, 1, 0, 1, 2, 2, 0]
        );
        assert_eq!(plan.decode_base, 8);
    }

    #[test]
    fn multimodal_expansion_rejects_marker_count_mismatch() {
        let image = MultimodalEmbedding {
            values: vec![0.0; 4],
            grid_nx: 1,
            grid_ny: 1,
        };
        assert!(expand_multimodal_prompt(&[1, 2], 99, vec![image], 4).is_err());
    }
}
