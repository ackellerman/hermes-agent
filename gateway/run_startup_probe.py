"""Off-loop boot probes for GatewayRunner: run a boot-time probe under the launch scope.

Split out of ``gateway/run_startup.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextvars import copy_context

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayStartupProbeMixin:
    async def _run_boot_probe_in_launch_scope(self, fn, *, daemon: bool = False):
        """Run a boot-time probe off the event loop under the LAUNCH profile's scope when
        multiplexing (the ``_discover_gateway_mcp_tools`` shape). Boot probes (``check_fn``s) read profile-scoped secrets; an unscoped read under multiplex fails
        closed, so routing overrides resolve as absent and default routing applies. ``copy_context``
        carries the contextvars across the hop; single-profile keeps environ semantics.

        "Off the event loop" is about the loop's THREAD, not the GIL: the probe's own work can still
        hold the interpreter lock (``transcription_local.preload_cuda_libraries`` and its
        ``ctypes.CDLL`` calls do), which freezes the loop for that call even from this thread.

        ``daemon=True`` runs the probe on a daemon ``threading.Thread`` instead of the loop's
        DEFAULT executor (review r1 F2). ``asyncio.run`` joins that executor at close
        (``shutdown_default_executor`` with ``THREAD_JOIN_TIMEOUT`` = 300 s) and cancelling a
        ``run_in_executor`` future does not stop its thread, so a cold first-time model download
        (~1.5 GB) behind the STT check would hold ``hermes gateway stop`` and the ``hermes update``
        restart for minutes. A daemon thread is never joined, dies with the interpreter, and still
        carries the copied context — the probe is awaited for its result, just not on a thread the
        loop owns."""
        if daemon:
            async def _run_probe():
                return await self._run_probe_on_daemon_thread(fn)
        else:
            loop = asyncio.get_running_loop()

            async def _run_probe():
                return await loop.run_in_executor(None, copy_context().run, fn)
        if getattr(self.config, "multiplex_profiles", False):
            from gateway.run import _async_profile_runtime_scope
            from hermes_constants import get_hermes_home
            try:
                # ONLY the scope entry is guarded: an ``async with`` around the probe would let a
                # probe body's exception land in ``except Exception`` and re-run the probe at the
                # bottom, outside the launch profile — two CUDA warnings, two model loads, the
                # second one reading ``stt.*`` from the wrong profile (review r2 F2).
                scope = _async_profile_runtime_scope(get_hermes_home())
                await scope.__aenter__()
            except Exception:
                # Same fallback as load_gateway_config_for_runner: a scope that cannot be built must not
                # abort startup; the probe runs unscoped as it did before.
                logger.debug("multiplex launch-scope entry failed for boot probe %s; running unscoped",
                             getattr(fn, "__name__", fn), exc_info=True)
            else:
                try:
                    return await _run_probe()
                finally:
                    await scope.__aexit__(None, None, None)
        return await _run_probe()

    @staticmethod
    async def _run_probe_on_daemon_thread(fn):
        """Run *fn* on a daemon thread carrying the calling thread's contextvars; await the result.

        Not joined at loop close, so process exit never waits for the probe (see
        ``_run_boot_probe_in_launch_scope``). Cancellation of the awaiting task detaches it: the
        thread finishes on its own and the result is dropped.
        """
        loop = asyncio.get_running_loop()
        context = copy_context()  # copied HERE: the probe must run in the launch profile's scope
        done: asyncio.Future = loop.create_future()

        def _resolve(error, value):
            if done.done():  # cancelled by shutdown — nothing is listening any more
                return
            if error is not None:
                done.set_exception(error)
            else:
                done.set_result(value)

        def _target():
            error = None
            value = None
            try:
                value = context.run(fn)
            except BaseException as exc:  # health: allow BLE001 -- never swallowed: the exception is re-raised on the awaiting future, and BaseException is deliberate so a SystemExit in the probe still reaches its awaiter
                error = exc
            try:
                loop.call_soon_threadsafe(_resolve, error, value)
            except RuntimeError:
                # Loop already closed: a shutdown raced the probe. Nothing to report to.
                logger.debug("Boot probe %s finished after the event loop closed",
                             getattr(fn, "__name__", fn))

        threading.Thread(target=_target, name="startup-stt-check", daemon=True).start()
        return await done
