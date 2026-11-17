"""Voice-aware scheduler for vLLM (benchmark prototype, written against v0.32.0).

Load it without forking vLLM:

    PYTHONPATH=. vllm serve <model> --scheduling-policy priority \
        --scheduler-cls voice_sched.VoiceScheduler

Clients talk to it through the OpenAI `priority` field:

    priority = 0                 batch / best effort, no deadline
    priority = N   (0 < N < 1e6) voice turn, TTFT SLO of N ms
    priority = 1e6 + N           speculative prefill on a partial ASR
                                 transcript (send max_tokens=1), soft deadline N ms

What it does:
  1. Deadline scheduling: priority is rewritten to the absolute TTFT deadline, so
     vLLM's priority queue becomes earliest-deadline-first, and its KV-pressure
     preemption evicts batch work before voice work.
  2. KV kept on barge-in: when a voice request ends (aborted by barge-in,
     finished, or a speculative prefill), its KV blocks are pinned for
     VOICE_PIN_TTL_S so the session's next turn gets a prefix-cache hit even if
     batch load would otherwise have evicted them.
  3. Prefill on partial ASR: a speculative request computes and pins the KV for
     history + partial transcript, so the final request only prefills the tail.
  4. SLO tracking: server-side TTFT-deadline attainment, logged every 10 s.
"""

import logging
import os
import time
from collections import deque

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

log = logging.getLogger("vllm.voice_sched")

PREFILL_FLAG = 1_000_000
BATCH_PRIORITY = 1 << 62
PIN_TTL_S = float(os.environ.get("VOICE_PIN_TTL_S", "15"))  # 0 disables pinning
PIN_MAX_FRAC = float(os.environ.get("VOICE_PIN_MAX_FRAC", "0.3"))  # of KV pool
LOG_EVERY_S = 10.0


class VoiceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pins = deque()  # (expiry_time, [KVCacheBlock]), oldest first
        self._num_pinned = 0
        self._awaiting_first_token = {}  # request_id -> Request (voice turns)
        self._slo_met = self._slo_total = 0
        self._last_log = time.time()

    # --- 1. deadline (EDF) scheduling -------------------------------------
    def add_request(self, request):
        if not hasattr(request, "voice_kind"):
            p = request.priority
            request.voice_kind = (
                None if p <= 0 else "prefill" if p >= PREFILL_FLAG else "turn"
            )
            if request.voice_kind is None:
                request.priority = BATCH_PRIORITY
            else:
                request.voice_deadline = request.arrival_time + (p % PREFILL_FLAG) / 1e3
                request.priority = int(request.voice_deadline * 1e3)
                if request.voice_kind == "turn":
                    self._awaiting_first_token[request.request_id] = request
        super().add_request(request)

    # --- 2 + 3. keep KV after barge-in / turn end / speculative prefill ----
    def _free_request(self, request, *args, **kwargs):
        if self._awaiting_first_token.pop(request.request_id, None):
            self._score(request, time.time())
        if getattr(request, "voice_kind", None) and PIN_TTL_S > 0:
            self._pin(request)
        return super()._free_request(request, *args, **kwargs)

    def _pin(self, request):
        """Take an extra reference on the request's blocks so that the free that
        follows leaves them allocated (and still in the prefix-cache table)."""
        pool = self.kv_cache_manager.block_pool
        groups = self.kv_cache_manager.get_blocks(request.request_id).blocks
        blocks = [b for g in groups for b in g if not b.is_null]
        cap = int(PIN_MAX_FRAC * pool.num_gpu_blocks)
        if not blocks or len(blocks) > cap:
            return
        for b in blocks:
            b.ref_cnt += 1
        self._pins.append((time.time() + PIN_TTL_S, blocks))
        self._num_pinned += len(blocks)
        while self._num_pinned > cap:
            self._unpin_oldest()

    def _unpin_oldest(self):
        _, blocks = self._pins.popleft()
        self._num_pinned -= len(blocks)
        # Tail blocks first, so they are evicted before the shared prefix.
        self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    def schedule(self, *args, **kwargs):
        now = time.time()
        while self._pins and self._pins[0][0] <= now:
            self._unpin_oldest()
        return super().schedule(*args, **kwargs)

    # --- 4. SLO attainment --------------------------------------------------
    def _score(self, request, now):
        if request.num_output_tokens > 0:
            self._slo_total += 1
            self._slo_met += now <= request.voice_deadline

    def update_from_output(self, *args, **kwargs):
        out = super().update_from_output(*args, **kwargs)
        now = time.time()
        for rid, req in list(self._awaiting_first_token.items()):
            if req.num_output_tokens > 0:
                del self._awaiting_first_token[rid]
                self._score(req, now)
        if now - self._last_log >= LOG_EVERY_S and self._slo_total:
            self._last_log = now
            log.info(
                "voice SLO attainment %d/%d (%.1f%%), pinned KV blocks %d",
                self._slo_met,
                self._slo_total,
                100 * self._slo_met / self._slo_total,
                self._num_pinned,
            )
        return out


class VoiceScheduler(VoiceMixin, AsyncScheduler):
    """Default: vLLM runs async scheduling unless told otherwise."""


class SyncVoiceScheduler(VoiceMixin, Scheduler):
    """Use with --no-async-scheduling."""
