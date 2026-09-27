"""State-driven GPU scheduling without an independent in-memory queue."""
from __future__ import annotations

import datetime as dt
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from codeaction.batch.state import BatchStateError, BatchStateStore, CellUnavailableError


@dataclass(frozen=True)
class SchedulerDecision:
    status: str
    cell_id: str | None = None
    lease: Mapping[str, Any] | None = None
    reason: str | None = None


def _parse_time(value: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise BatchStateError(f"invalid retry timestamp {value!r}")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BatchStateError(f"invalid retry timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise BatchStateError("retry timestamp must include a timezone")
    return parsed


class BatchScheduler:
    """Choose queued cells while deriving every concurrency/pause fact from batch state."""

    def __init__(
        self,
        store: BatchStateStore,
        *,
        model_credentials: Mapping[str, str | None],
        credential_limits: Mapping[str, int],
    ) -> None:
        models = tuple(store.specification["models"])
        if set(model_credentials) != set(models):
            raise BatchStateError("credential mapping must exactly cover the frozen model set")
        selected_credentials = {
            credential for credential in model_credentials.values() if credential is not None}
        if set(credential_limits) != selected_credentials:
            raise BatchStateError(
                "credential limits must exactly cover selected non-null credential aliases")
        if any(not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
               for limit in credential_limits.values()):
            raise BatchStateError("credential limits must be positive integers")
        self.store = store
        self.model_credentials = dict(model_credentials)
        self.credential_limits = dict(credential_limits)
        # How many episodes of ONE model may be in flight at once. Read from the FROZEN batch
        # specification, not passed in, so the scheduler and the durable state store cannot
        # disagree about it -- the store enforces the same number when it hands out a lease and
        # re-checks it on every reopen. The default of 1 is the behaviour every batch had before
        # lanes were expressible; raising it is what lets one agent occupy several GPUs, which is
        # the whole point for a subscription driver with more than one lane on one account.
        self.model_limits = {model: store.model_lanes(model) for model in models}
        self._model_order = {model: index for index, model in enumerate(models)}
        self._dispatch_lock = threading.RLock()

    @staticmethod
    def _open_pause_scopes(state: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
        paused_cells: set[str] = set()
        paused_models: set[str] = set()
        paused_credentials: set[str] = set()
        for attention in state.get("attentions", {}).values():
            if attention.get("status") != "open":
                continue
            scope = attention.get("scope") or {}
            kind, identifier = scope.get("kind"), scope.get("id")
            if kind == "cell":
                paused_cells.add(identifier)
            elif kind == "model":
                paused_models.add(identifier)
            elif kind == "credential":
                paused_credentials.add(identifier)
        return paused_cells, paused_models, paused_credentials

    def requeue_due_retries(self, now: dt.datetime) -> tuple[str, ...]:
        with self._dispatch_lock:
            if now.tzinfo is None:
                raise BatchStateError("scheduler time must include a timezone")
            requeued = []
            snapshot = self.store.snapshot()
            for identifier in snapshot["requested_cells"]:
                cell = snapshot["cells"][identifier]
                if cell["status"] != "retry_wait":
                    continue
                retry_at = cell.get("next_retry_at")
                if retry_at is None or _parse_time(retry_at) <= now:
                    self.store.requeue_cell(identifier, note="retry cooldown elapsed")
                    requeued.append(identifier)
            return tuple(requeued)

    def try_acquire(
        self,
        *,
        worker_id: str,
        gpu: int,
        now: dt.datetime,
        deadline_at: str | None,
    ) -> SchedulerDecision:
        with self._dispatch_lock:
            return self._try_acquire_locked(
                worker_id=worker_id, gpu=gpu, now=now, deadline_at=deadline_at)

    def _try_acquire_locked(
        self,
        *,
        worker_id: str,
        gpu: int,
        now: dt.datetime,
        deadline_at: str | None,
    ) -> SchedulerDecision:
        self.requeue_due_retries(now)
        # A window pause is released by time, so the clock is read on the same tick that asks for
        # work rather than by a separate timer nobody restarts after a crash.
        self.store.clear_due_credential_pauses(now.timestamp())
        state = self.store.snapshot()
        counts = self.store.stage_counts()
        if counts["accepted"] + counts["abandoned"] == counts["total"]:
            return SchedulerDecision(
                "stage_complete", reason="every requested cell is accepted or abandoned")
        pause = state.get("queue_pause")
        if pause:
            # Running leases are deliberately untouched: the pause stops NEW spend, it does not
            # discard an episode already in flight.
            return SchedulerDecision(
                "paused", reason=f"queue paused: {pause.get('note')}")

        active_models: Counter[str] = Counter()
        active_credentials: Counter[str] = Counter()
        for lease in state["leases"].values():
            cell = state["cells"][lease["cell_id"]]
            model = cell["model"]
            active_models[model] += 1
            credential = self.model_credentials[model]
            if credential is not None:
                active_credentials[credential] += 1
        paused_cells, paused_models, paused_credentials = self._open_pause_scopes(state)
        # Window pauses join the credential scope the scheduler already honours, so a spent
        # subscription blocks exactly what an operator-opened credential attention blocks.
        paused_credentials |= set((state.get("credential_pauses") or {}))

        dispatch = state.get("dispatch_queue")
        if not isinstance(dispatch, list):
            # Backward-compatible reconstruction for batches created before the durable queue.
            models = list(self._model_order)
            model_order = {model: index for index, model in enumerate(models)}
            task_order: dict[str, int] = {}
            for identifier in state["requested_cells"]:
                task_order.setdefault(state["cells"][identifier]["task"], len(task_order))
            dispatch = sorted(
                state["requested_cells"],
                key=lambda value: (
                    task_order[state["cells"][value]["task"]],
                    model_order[state["cells"][value]["model"]],
                ),
            )
        order = {identifier: index for index, identifier in enumerate(dispatch)}
        eligible = []
        for identifier in state["requested_cells"]:
            cell = state["cells"][identifier]
            if cell["status"] != "queued":
                continue
            model = cell["model"]
            credential = self.model_credentials[model]
            if identifier in paused_cells or model in paused_models \
                    or credential in paused_credentials:
                continue
            if active_models[model] >= self.model_limits[model]:
                continue
            if credential is not None \
                    and active_credentials[credential] >= self.credential_limits[credential]:
                continue
            eligible.append(identifier)

        if not eligible:
            attention_blocks = counts["needs_attention"] > 0 or bool(
                paused_cells or paused_models or paused_credentials)
            return SchedulerDecision(
                "needs_attention" if attention_blocks and not counts["running"] else "waiting",
                reason=("no eligible cell until attention is resolved" if attention_blocks
                        else "cells are running or waiting for retry cooldown"),
            )

        identifier = min(eligible, key=order.__getitem__)
        try:
            lease = self.store.acquire_cell(
                identifier, worker_id=worker_id, gpu=gpu, deadline_at=deadline_at)
        except CellUnavailableError as exc:
            # Another worker may have acquired the candidate between snapshot and transition.
            # Report a retryable scheduling wait; never manufacture a second in-memory queue.
            return SchedulerDecision("waiting", reason=str(exc))
        return SchedulerDecision("acquired", cell_id=identifier, lease=lease)
