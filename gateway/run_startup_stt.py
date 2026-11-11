"""The local-STT CUDA check the gateway starts at boot (fire-and-forget).

Split out of ``gateway/run_startup.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from gateway.run_shutdown import _log_suppressed

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayStartupSttMixin:
    _startup_stt_task: Optional[asyncio.Task] = None

    def _start_startup_stt_check(self) -> None:
        """Kick off the local-STT CUDA check (+ warm) as a FIRE-AND-FORGET task.

        Deliberately not part of ``_warm_turn_prerequisites``: that warm-up is awaited with the
        inbound gate shut for up to ``HERMES_STARTUP_WARMUP_TIMEOUT`` (20 s), and importing
        ctranslate2 plus loading medium must never delay gateway start (#stt-cuda-survives-update
        step 4). Nothing awaits this task; it is only kept reachable and reported on failure —
        and a failure is a WARNING, because "no CUDA warning at start" (e4's acceptance line) must
        not be indistinguishable from "the check died"."""
        self._startup_stt_task = asyncio.ensure_future(self._check_startup_stt())
        self._startup_stt_task.add_done_callback(
            self._late_failure_callback("startup local-STT CUDA check failed"))

    async def _check_startup_stt(self) -> None:
        """Run the check on a daemon thread inside the launch profile's scope, so ``stt.*``
        resolves for the profile that launched the gateway (multiplex fails closed unscoped)."""
        from tools.stt_startup import check_local_stt_at_startup
        with _log_suppressed(logging.WARNING, "Startup local-STT CUDA check failed", exc_info=True):
            await self._run_boot_probe_in_launch_scope(check_local_stt_at_startup, daemon=True)
