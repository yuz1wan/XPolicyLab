"""Experiment tracking with a preferred backend and an explicit fallback.

Weights & Biases stays the default. When this host cannot reach the configured W&B
server, training falls back to SwanLab instead of silently losing the run. The choice
is made once, printed, and recorded next to the checkpoint so a resumed run rejoins
the same backend and run id.

Configuration, in priority order:

* ``OPENPI_TRACKER``    - force ``wandb``, ``swanlab``, or ``disabled``; no probing.
* ``WANDB_API_KEY``     - credentials for W&B. ``WANDB_BASE_URL`` selects a private server.
* ``SWANLAB_API_KEY``   - credentials for SwanLab. ``SWANLAB_HOST`` selects a private server.
* ``WANDB_MODE`` / ``SWANLAB_MODE`` - ``offline`` records locally for a later ``sync``.

An offline W&B run needs no network, so ``WANDB_MODE=offline`` keeps the W&B backend.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os

WANDB = "wandb"
SWANLAB = "swanlab"
DISABLED = "disabled"
BACKENDS = (WANDB, SWANLAB, DISABLED)
_STATE_FILE = "tracker_run.json"
_LEGACY_FILE = "wandb_id.txt"


def _importable(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _wandb_usable() -> tuple[bool, str]:
    """A plain HTTP probe cannot tell a working server from a blocking proxy, so ask
    the W&B client to actually authenticate before committing a run to this backend."""
    if not _importable(WANDB):
        return False, "wandb is not installed"
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode in ("offline", "dryrun"):
        return True, f"WANDB_MODE={mode} records locally without network access"
    if not os.environ.get("WANDB_API_KEY"):
        return False, "WANDB_API_KEY is not set"
    host = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/")
    timeout = float(os.environ.get("OPENPI_TRACKER_PROBE_SECONDS", "15"))
    import wandb

    try:
        if wandb.login(key=os.environ["WANDB_API_KEY"], host=host, timeout=timeout, verify=True):
            return True, f"authenticated against {host}"
        return False, f"wandb.login was refused by {host}"
    except Exception as error:  # noqa: BLE001 - any failure means "do not use wandb here"
        return False, f"cannot authenticate against {host}: {type(error).__name__}: {error}"


def _swanlab_usable() -> tuple[bool, str]:
    if not _importable(SWANLAB):
        return False, "swanlab is not installed"
    mode = os.environ.get("SWANLAB_MODE", "").lower()
    if mode in ("offline", "local", "disabled"):
        return True, f"SWANLAB_MODE={mode} records locally"
    if os.environ.get("SWANLAB_API_KEY"):
        return True, "SWANLAB_API_KEY is set"
    return False, "SWANLAB_API_KEY is not set and SWANLAB_MODE is not offline"


def select_backend() -> tuple[str, str]:
    """Return the backend to use and the one-line reason for the choice."""
    forced = os.environ.get("OPENPI_TRACKER", "").strip().lower()
    if forced:
        if forced not in BACKENDS:
            raise ValueError(f"OPENPI_TRACKER must be one of {BACKENDS}, got {forced!r}")
        return forced, "forced by OPENPI_TRACKER"
    ok, why = _wandb_usable()
    if ok:
        return WANDB, why
    wandb_why = why
    ok, why = _swanlab_usable()
    if ok:
        return SWANLAB, f"wandb unavailable ({wandb_why}); swanlab {why}"
    raise RuntimeError(
        "No experiment tracker is usable.\n"
        f"  wandb:   {wandb_why}\n"
        f"  swanlab: {why}\n"
        "Set WANDB_API_KEY (and WANDB_BASE_URL for a private server), or SWANLAB_API_KEY,\n"
        "or run offline with WANDB_MODE=offline / SWANLAB_MODE=offline,\n"
        "or disable tracking with OPENPI_TRACKER=disabled."
    )


class Tracker:
    """Minimal scalar/image logger over whichever backend was selected."""

    def __init__(self, backend: str, module, run_id: str | None):
        self.backend = backend
        self._module = module
        self.run_id = run_id

    @property
    def enabled(self) -> bool:
        return self.backend != DISABLED

    def image(self, array, *, value_range: tuple[float, float] = (-1.0, 1.0)):
        """Log an image, converting a float array from `value_range` to bytes.

        Callers log what the model actually sees, and `Observation` carries images
        normalized to [-1, 1]. Both backends read a float array as [0, 1] and clip,
        which turns roughly half of every pixel black. The range is a parameter rather
        than something inferred from the data: an all-zero or uniformly bright frame
        gives no evidence of which convention produced it, and guessing wrong on a
        sanity-check image is exactly the failure this is meant to prevent.
        """
        if self.backend == DISABLED:
            return None
        import numpy as np

        array = np.asarray(array)
        if np.issubdtype(array.dtype, np.floating):
            low, high = value_range
            if not high > low:
                raise ValueError(f"value_range must be increasing, got {value_range}")
            array = np.clip((array - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
        return self._module.Image(array)

    def log(self, data: dict, step: int) -> None:
        if self.backend == DISABLED:
            return
        self._module.log(data, step=step)

    def finish(self) -> None:
        if self.backend == DISABLED:
            return
        try:
            self._module.finish()
        except Exception:  # noqa: BLE001 - never fail a finished run on teardown
            logging.exception("Tracker finish failed")


def _read_state(ckpt_dir) -> dict:
    state_path = ckpt_dir / _STATE_FILE
    if state_path.exists():
        return json.loads(state_path.read_text())
    legacy = ckpt_dir / _LEGACY_FILE
    if legacy.exists():
        return {"backend": WANDB, "run_id": legacy.read_text().strip()}
    raise FileNotFoundError(
        f"Cannot resume tracking: neither {state_path} nor {legacy} exists. "
        "Set OPENPI_TRACKER=disabled to resume training without a tracker."
    )


def init(config, *, resuming: bool, enabled: bool = True, log_code: bool = False) -> Tracker:
    """Start or rejoin a run for this training config."""
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if not enabled:
        backend, reason = DISABLED, "tracking disabled by the training config"
    elif resuming:
        state = _read_state(ckpt_dir)
        backend, reason = state["backend"], f"resuming the run recorded in {ckpt_dir.name}"
    else:
        backend, reason = select_backend()
    logging.info(f"Experiment tracker: {backend} ({reason})")

    if backend == DISABLED:
        import wandb  # wandb's disabled mode keeps the call sites uniform.

        wandb.init(mode="disabled")
        return Tracker(DISABLED, wandb, None)

    module = __import__(backend)
    settings = {"project": config.project_name}
    if backend == SWANLAB:
        # Without an explicit directory SwanLab writes ./swanlog under the working
        # directory, which for this repository is the source tree. Resumed runs need
        # the same destination, so this applies to both branches below.
        settings["log_dir"] = os.environ.get("SWANLAB_LOG_DIR") or str(ckpt_dir / "swanlog")
        for key, name in (("SWANLAB_WORKSPACE", "workspace"), ("SWANLAB_MODE", "mode")):
            if os.environ.get(key):
                settings[name] = os.environ[key]

    if resuming:
        run_id = _read_state(ckpt_dir)["run_id"]
        module.init(id=run_id, resume="must", **settings)
        return Tracker(backend, module, run_id)

    module.init(name=config.exp_name, config=dataclasses.asdict(config), **settings)
    run = getattr(module, "run", None)
    run_id = str(getattr(run, "id", "") or getattr(run, "public", None) and run.public.run_id or "")
    (ckpt_dir / _STATE_FILE).write_text(
        json.dumps({"backend": backend, "run_id": run_id, "project": config.project_name,
                    "exp_name": config.exp_name, "reason": reason}, indent=2) + "\n"
    )
    if backend == WANDB:
        # Preserved for tooling that still reads the historical file name.
        (ckpt_dir / _LEGACY_FILE).write_text(run_id)
        if log_code:
            module.run.log_code(".")
    return Tracker(backend, module, run_id)
