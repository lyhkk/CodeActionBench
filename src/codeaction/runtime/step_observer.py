"""Host-side per-physics-step observer seam. ONE seam, multiple consumers:
the poll_and_latch monitor (codeaction.verification.verifiers.LatchMonitor) and the episode recorder register
callbacks that run on the sim-owning thread immediately after every physics step, in registration
order. `sapien.Scene` is a Python wrapper class, so the wrap is a plain INSTANCE-level override of
`env.scene.step` (verified monkeypatchable on the pinned sim, 2026-07-18) — zero upstream edits,
and it covers every step site in envs/_base_task.py (take_action / take_dense_action / delay /
setup settling) as well as the primitives-based motion the codeaction tools use.

GT-free by design: this module never reads scene content. Callbacks close over whatever state they
need — GT-reading latch predicates live in codeaction.verification.verifiers (the sanctioned out-of-band file);
the recorder reads cameras only. Callback failures are disabled and kept in `errors`; observers
report facts and never alter robot execution. The task-level physical-time budget is different:
reaching it is latched here, while the current
model-visible D0 tool is allowed to finish.  The common tool-registry boundary then raises the
terminal before another tool can start.  This keeps inherited primitive internals atomic and avoids
depending on every broad ``except Exception`` in upstream motion code to preserve a control-flow
exception.

Never agent-visible: not a tool, not in any registry; attached only by episode hosts
(reference/probe scripts, mcp_episode_server). Re-attach after any env re-boot — setup_demo
builds a fresh scene object, and the wrap lives on the scene instance."""
import math

from codeaction.contracts.failures import PhysicalTimeBudgetExhausted


class StepObserver:
    def __init__(self, env, *, physical_time_budget_s=None, expert_sim_duration_s=None):
        self._scene = env.scene
        self._orig_step = None
        self._callbacks = []          # [name, fn, enabled]
        self.step_count = 0
        self.budget_reached_step = None
        self.errors = {}
        self.physical_time_budget_s = (
            None if physical_time_budget_s is None else float(physical_time_budget_s))
        self.expert_sim_duration_s = (
            None if expert_sim_duration_s is None else float(expert_sim_duration_s))
        if self.expert_sim_duration_s is not None and (
                not math.isfinite(self.expert_sim_duration_s)
                or self.expert_sim_duration_s <= 0):
            raise ValueError("expert_sim_duration_s must be a positive finite number")
        get_timestep = getattr(self._scene, "get_timestep", None)
        self.sim_dt = (
            float(get_timestep()) if callable(get_timestep) else None)
        if self.sim_dt is not None \
                and (not math.isfinite(self.sim_dt) or self.sim_dt <= 0):
            raise ValueError("scene timestep must be a positive finite number")
        if self.physical_time_budget_s is None:
            self.threshold_physics_steps = None
        else:
            if not math.isfinite(self.physical_time_budget_s) \
                    or self.physical_time_budget_s <= 0:
                raise ValueError("physical_time_budget_s must be a positive finite number")
            if self.sim_dt is None:
                raise ValueError(
                    "scene.get_timestep() is required when a physical time budget is enabled")
            # SAPIEN may return its timestep through a lower-precision C float. Requiring the
            # quotient to be an exact integer rejected the ordinary 1/250 s scene timestep. The
            # configured seconds define the threshold, so derive the largest whole step count that
            # cannot cross it. Any later overshoot belongs only to the completing atomic D0 tool.
            self.threshold_physics_steps = math.floor(
                self.physical_time_budget_s / self.sim_dt)
            if self.threshold_physics_steps < 1:
                raise ValueError(
                    "physical_time_budget_s must permit at least one scene timestep")
        self.effective_physical_time_threshold_s = (
            None if self.threshold_physics_steps is None
            else self.threshold_physics_steps * self.sim_dt)

    def _physical_budget_error(self):
        return PhysicalTimeBudgetExhausted(
            physics_step=self.step_count,
            threshold_physics_steps=self.threshold_physics_steps,
            budget_s=self.physical_time_budget_s,
            sim_dt=self.sim_dt,
        )

    def raise_if_exhausted(self):
        """End the episode at a D0 tool boundary after the budget threshold was reached."""
        if self.threshold_physics_steps is not None \
                and self.step_count >= self.threshold_physics_steps:
            raise self._physical_budget_error()

    def add(self, name, fn):
        """Register fn(step_index) — called after each physics step while enabled."""
        self._callbacks.append([str(name), fn, True])
        return self

    def attach(self):
        if self._orig_step is not None:
            raise RuntimeError("StepObserver already attached")
        orig = self._scene.step

        def _stepped():
            orig()
            self.step_count += 1
            if self.threshold_physics_steps is not None \
                    and self.step_count >= self.threshold_physics_steps \
                    and self.budget_reached_step is None:
                self.budget_reached_step = self.step_count
            for rec in self._callbacks:
                if not rec[2]:
                    continue
                try:
                    rec[1](self.step_count)
                except Exception as e:      # noqa: BLE001 — fail-safe by contract
                    rec[2] = False
                    self.errors[rec[0]] = f"{type(e).__name__}: {e}"

        self._scene.step = _stepped
        self._orig_step = orig
        return self

    def detach(self):
        """Restore the scene's own step (instance attr removed -> class method resolves again)."""
        if self._orig_step is None:
            return
        try:
            del self._scene.step
        except AttributeError:
            self._scene.step = self._orig_step
        self._orig_step = None

    def state(self):
        physical_time_s = (
            None if self.sim_dt is None
            else round(self.step_count * self.sim_dt, 6))
        overshoot_steps = (
            0 if self.threshold_physics_steps is None
            else max(0, self.step_count - self.threshold_physics_steps))
        return {"steps": self.step_count,
                "sim_dt": self.sim_dt,
                "physical_time_s": physical_time_s,
                "physical_time_budget_s": self.physical_time_budget_s,
                "threshold_physics_steps": self.threshold_physics_steps,
                "effective_physical_time_threshold_s": self.effective_physical_time_threshold_s,
                "budget_reached_step": self.budget_reached_step,
                "tool_boundary_overshoot_steps": overshoot_steps,
                "tool_boundary_overshoot_s": (
                    None if self.sim_dt is None
                    else round(overshoot_steps * self.sim_dt, 6)),
                "expert_sim_duration_s": self.expert_sim_duration_s,
                "physical_time_to_expert_ratio": (
                    None if physical_time_s is None or self.expert_sim_duration_s is None
                    else round(physical_time_s / self.expert_sim_duration_s, 6)),
                "callbacks": {rec[0]: ("enabled" if rec[2] else "disabled")
                              for rec in self._callbacks},
                "errors": dict(self.errors)}
