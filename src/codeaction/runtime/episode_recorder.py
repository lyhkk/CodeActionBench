"""Episode video recorder — a consumer of the per-physics-step
observer seam (codeaction.runtime.step_observer). Every `every` physics steps it refreshes the render, grabs
the head-camera RGB, and pipes the raw frame into an ffmpeg child — the upstream pattern from
script/eval_policy.py, including the load-bearing LD_LIBRARY_PATH scrub (the robotwin conda libffi
otherwise crashes the pipe).

Scope and storage contract: this is an evaluator-view recording of what physically executes in
the environment, not a screen recording or a complete record of agent execution. It intentionally
does not capture model text generation, API/CLI latency, tool serialization, or other wall-clock
activity. The transcript is the source for those events.

Timeline contract: SIM TIME at fixed cadence. Video fps = 1 / (sim_dt * every); agent thinking
gaps hold zero physics steps and therefore zero frames. This avoids both misleading frozen frames
and storage growth proportional to agent wall time: storage is bounded by simulated physics time
and the configured sampling cadence. finalize() closes full.mp4, derives review.mp4 (uniform
setpts speedup only when the full cut exceeds `review_max_s`; otherwise a stream copy), and writes
video_meta.json with the speedup so a reviewer always knows the time compression.

GT-free: reads cameras only (the same surface the agent sees), never scene state. Never
agent-visible; attached only by episode hosts. A broken pipe raises out of on_step and the
StepObserver disables the callback — recording dies, physics survives."""
import functools
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

_DEFAULT_DT = 1.0 / 250.0        # envs/_base_task.py setup default when the scene has no getter


def _scrubbed_env():
    return {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Resolve the ffmpeg BINARY: PATH first, else the static build imageio-ffmpeg ships with.

    The sim image installs `ffmpeg==1.4` (a thin PYTHON wrapper) and `imageio-ffmpeg` (which
    bundles a real binary but does not put it on PATH), and no apt ffmpeg. Spawning the bare name
    therefore raised FileNotFoundError, the recorder disabled itself, and every container episode
    silently produced `recording.enabled=false` with no video — measured on the 2026-08-05
    grab_roller batch. Resolving the bundled binary fixes this with a dependency that is already
    in docker/requirements.lock, so no image grows and no lock digest moves.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    from imageio_ffmpeg import get_ffmpeg_exe   # already pinned in docker/requirements.lock
    return str(get_ffmpeg_exe())


def _spawn_ffmpeg(path, width, height, fps):
    return subprocess.Popen(
        [ffmpeg_exe(), "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{width}x{height}", "-framerate", f"{fps:.6f}",
         "-i", "-",
         "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
         str(path)],
        stdin=subprocess.PIPE, env=_scrubbed_env())


def derive_review(full_path, review_path, duration_s, max_s=150.0, min_s=90.0):
    """review.mp4 = the full episode, continuously (never a keyframe slideshow): stream-copied
    when it already fits under max_s, else uniformly sped up toward the 90-150 s review window.
    Returns {"speedup", "duration_s"}. Speedup is never < 1 (a short episode is not stretched)."""
    duration_s = float(duration_s)
    if duration_s <= float(max_s):
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(full_path),
               "-c", "copy", str(review_path)]
        speedup, review_s = 1.0, duration_s
    else:
        target = max(float(min_s), min(float(max_s), (float(min_s) + float(max_s)) / 2.0))
        speedup = duration_s / target
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(full_path),
               "-filter:v", f"setpts=PTS/{speedup:.6f}", "-an", str(review_path)]
        review_s = target
    subprocess.run(cmd, check=True, env=_scrubbed_env())
    return {"speedup": round(speedup, 4), "duration_s": round(review_s, 2)}


class EpisodeRecorder:
    def __init__(self, env, out_dir, every=10, camera="head_camera",
                 review_max_s=150.0, review_min_s=90.0):
        self.env = env
        self.out = Path(out_dir)
        self.every = max(1, int(every))
        self.camera = camera
        self.review_max_s = float(review_max_s)
        self.review_min_s = float(review_min_s)
        self.frames = 0
        self.event_frames = []
        self._ffmpeg = None
        self._size = None
        self._cam = None
        try:
            self.sim_dt = float(env.scene.get_timestep())
        except Exception:
            self.sim_dt = _DEFAULT_DT
        self.fps = 1.0 / (self.sim_dt * self.every)

    def _resolve_camera(self):
        """Find the named camera object for the single-camera fast path (rendering ALL mounted
        cameras per frame is ~4x the cost and dominated the first smoke); None -> fallback to the
        upstream all-camera path."""
        for cam in getattr(self.env.cameras, "static_camera_list", []) or []:
            name = cam.get_name() if callable(getattr(cam, "get_name", None)) \
                else getattr(cam, "name", None)
            if name == self.camera:
                return cam
        return None

    def _grab(self):
        self.env._update_render()
        if self._cam is not None:
            self._cam.take_picture()
            rgba = self._cam.get_picture("Color")
            frame = (np.asarray(rgba) * 255).clip(0, 255).astype("uint8")[:, :, :3]
        else:
            self.env.cameras.update_picture()
            frame = self.env.cameras.get_rgb()[self.camera]["rgb"]
        return np.ascontiguousarray(frame)

    def start(self):
        """Probe one frame for the size, spawn the full.mp4 writer. Call after scene boot."""
        self.out.mkdir(parents=True, exist_ok=True)
        self._cam = self._resolve_camera()
        frame = self._grab()
        h, w = frame.shape[:2]
        self._size = (w, h)
        self._ffmpeg = _spawn_ffmpeg(self.out / "full.mp4", w, h, self.fps)
        try:
            self._ffmpeg.stdin.write(frame.tobytes())
        except Exception:
            proc, self._ffmpeg = self._ffmpeg, None
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait()
            raise
        self.frames = 1
        return self

    def on_step(self, step_index):
        """StepObserver callback: one frame every `every` physics steps."""
        if self._ffmpeg is None or step_index % self.every:
            return
        self._ffmpeg.stdin.write(self._grab().tobytes())
        self.frames += 1

    def finalize(self):
        """Close full.mp4, derive review.mp4, write video_meta.json. Returns the meta dict."""
        if self._ffmpeg is None:
            return None
        proc, self._ffmpeg = self._ffmpeg, None
        close_error = None
        try:
            proc.stdin.close()
        except Exception as e:
            close_error = e
        returncode = proc.wait()
        if returncode not in (None, 0):
            raise RuntimeError(f"ffmpeg writer exited with code {returncode}")
        if close_error is not None:
            raise close_error
        sim_duration = self.frames / self.fps
        meta = {"camera": self.camera, "width": self._size[0], "height": self._size[1],
                "sim_dt": round(self.sim_dt, 6), "every_steps": self.every,
                "video_fps": round(self.fps, 4), "frames": self.frames,
                "sim_duration_s": round(sim_duration, 2), "timeline": "sim_time",
                "codec": "h264", "pixel_format": "yuv420p", "crf": 23}
        if self.event_frames:
            meta["event_frames"] = list(self.event_frames)
        try:
            meta["review"] = derive_review(self.out / "full.mp4", self.out / "review.mp4",
                                           sim_duration, max_s=self.review_max_s,
                                           min_s=self.review_min_s)
        except Exception as e:      # noqa: BLE001 — full.mp4 must survive a review derivation bug
            meta["review"] = {"error": f"{type(e).__name__}: {e}"}
        (self.out / "video_meta.json").write_text(json.dumps(meta, indent=1))
        return meta
