"""Video-free, vectorized normalization for the explicitly supported YAM contract.

This module intentionally imports neither the model nor its tokenizer/data loader.
Statistics use the same 50-step, episode-clamped action population as LeRobot, with
12 joint deltas relative to the anchor state and absolute grippers. Exact linear
quantiles replace RunningStats' order-dependent, rebinned histogram approximation.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from openpi.training.yam_tasks import TASKS
from openpi.training.yam_tasks import YamTask


def load_numeric(root: Path, task: YamTask):
    """Read and validate scalar Parquet columns only, including episode boundaries."""
    info = json.loads((root / "meta/info.json").read_text())
    manifest = json.loads((root / "rhospolicy_conversion.json").read_text())
    if (
        info["codebase_version"] != "v3.0"
        or info["fps"] != 30
        or not manifest.get("completed")
        or manifest["task"] != task.prompt
        or manifest["action_semantics"] != "absolute_accepted_joint_target_at_grid_time"
        or manifest["arm_order"] != ["left", "right"]
        or manifest["timebase"] != "uniform_30hz_causal_host_source_time_allow_repeated_images"
    ):
        raise ValueError("Unsupported dataset contract; use the reference normalization path")
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pydict()
    if tasks != {"task_index": [0], "__index_level_0__": [task.prompt]}:
        raise ValueError("Dataset task text does not match the training config")
    paths = sorted((root / "data").rglob("*.parquet"))
    columns = [task.state_key, task.action_key, "index", "episode_index", "frame_index", "timestamp", "task_index"]
    table = pq.read_table(paths, columns=columns).combine_chunks()
    order = np.argsort(table["index"].to_numpy())
    table = table.take(pa.array(order))
    n = len(table)
    if n != info["total_frames"] or n != manifest["total_frames"]:
        raise ValueError("Frame count mismatch")
    if not np.array_equal(table["index"].to_numpy(), np.arange(n)):
        raise ValueError("Non-contiguous global indices")
    if np.any(table["task_index"].to_numpy() != 0):
        raise ValueError("Unexpected task index")
    dim = len(task.delta_mask)
    vectors = []
    for key in [task.state_key, task.action_key]:
        col = table[key].combine_chunks()
        if not np.all(pc.list_value_length(col).to_numpy() == dim):
            raise ValueError(f"Unexpected vector dimension: {key}")
        a = col.values.to_numpy().reshape(n, dim)
        if a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f"Expected finite float32 data: {key}")
        vectors.append(a)
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    episodes = pa.concat_tables(
        [
            pq.read_table(p, columns=["episode_index", "dataset_from_index", "dataset_to_index", "length"])
            for p in episode_files
        ]
    ).to_pylist()
    episodes.sort(key=lambda e: e["episode_index"])
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode count mismatch")
    ends = np.empty(n, dtype=np.int64)
    previous = 0
    ep_col, frame_col = table["episode_index"].to_numpy(), table["frame_index"].to_numpy()
    timestamps = table["timestamp"].to_numpy()
    for i, e in enumerate(episodes):
        lo, hi = e["dataset_from_index"], e["dataset_to_index"]
        if e["episode_index"] != i or lo != previous or hi <= lo or hi - lo != e["length"]:
            raise ValueError("Invalid episode boundaries")
        if not np.all(ep_col[lo:hi] == i) or not np.array_equal(frame_col[lo:hi], np.arange(hi - lo)):
            raise ValueError("Episode/frame index mismatch")
        if not np.allclose(timestamps[lo:hi], np.arange(hi - lo) / info["fps"], atol=3e-6, rtol=0):
            raise ValueError("Non-uniform timeline")
        ends[lo:hi] = hi - 1
        previous = hi
    if previous != n:
        raise ValueError("Incomplete episode coverage")
    fingerprint = hashlib.sha256()
    for path in [root / "meta/info.json", root / "rhospolicy_conversion.json", *paths, *episode_files]:
        fingerprint.update(str(path.relative_to(root)).encode())
        fingerprint.update(path.read_bytes())
    return (
        *vectors,
        ends,
        {
            "frames": n,
            "episodes": len(episodes),
            "scalar_bytes": sum(p.stat().st_size for p in paths),
            "source_sha256": fingerprint.hexdigest(),
        },
    )


def action_indices(ends, horizon, start=0, stop=None):
    stop = len(ends) if stop is None else stop
    if horizon < 1 or not 0 <= start < stop <= len(ends):
        raise ValueError("Invalid horizon or frame interval")
    return np.minimum(np.arange(start, stop)[:, None] + np.arange(horizon), ends[start:stop, None])


def action_chunks(states, actions, ends, horizon, mask, start=0, stop=None):
    """Reference-compatible float32 delta arithmetic; terminal actions repeat."""
    stop = len(ends) if stop is None else stop
    chunks = actions[action_indices(ends, horizon, start, stop)].copy()
    chunks -= np.where(np.asarray(mask), states[start:stop], np.float32(0))[:, None, :]
    return chunks


def vector_statistics(values):
    values = np.asarray(values).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Statistics require at least two finite values")
    q01, q99 = np.quantile(values, [0.01, 0.99], method="linear")
    return [float(np.mean(values, dtype=np.float64)), float(np.std(values, dtype=np.float64)), float(q01), float(q99)]


def compute_statistics(states, actions, ends, task, workers=None, *, include_tail=True):
    n = len(states) if include_tail else len(states) // task.batch_size * task.batch_size
    workers = min(states.shape[1], os.cpu_count() or 1) if workers is None else workers
    if n < 2 or workers < 1:
        raise ValueError("Insufficient frames or invalid worker count")
    indices = action_indices(ends, task.action_horizon, stop=n)

    def dimension(j):
        # Work by column: no full [N, horizon, 14] float tensor is needed.
        values = actions[:, j][indices]
        if task.delta_mask[j]:
            values -= states[:n, j, None]
        return vector_statistics(states[:n, j]), vector_statistics(values)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        columns = list(pool.map(dimension, range(states.shape[1])))
    fields = ["mean", "std", "q01", "q99"]
    return {
        key: {field: [col[k][i] for col in columns] for i, field in enumerate(fields)}
        for k, key in enumerate(["state", "actions"])
    }, n


def run(config_name, *, dataset_root=None, output_dir=None, workers=None, include_tail=True):
    task = TASKS[config_name]  # Explicit allowlist: no generic transform inference.
    workers = min(len(task.delta_mask), os.cpu_count() or 1) if workers is None else workers
    repo_id = task.resolved_repo_id()
    root = (
        Path(dataset_root)
        if dataset_root
        else Path(os.environ.get("HF_LEROBOT_HOME", str(Path.home() / ".cache/huggingface/lerobot"))) / repo_id
    )
    output = Path(output_dir) if output_dir else task.assets_dir() / repo_id
    start = time.perf_counter()
    states, actions, ends, provenance = load_numeric(root, task)
    read_s = time.perf_counter() - start
    stats, anchors = compute_statistics(states, actions, ends, task, workers, include_tail=include_tail)
    elapsed = time.perf_counter() - start
    payload = {"norm_stats": stats}
    report = {
        **provenance,
        "config_name": task.name,
        "repo_id": repo_id,
        "task": task.prompt,
        "dataset_root": str(root.resolve()),
        "action_horizon": task.action_horizon,
        "delta_mask": list(task.delta_mask),
        "state_vectors": anchors,
        "action_vectors": anchors * task.action_horizon,
        "include_tail": include_tail,
        "omitted_tail_frames": len(states) - anchors,
        "workers": workers,
        "quantiles": "exact_numpy_linear_q01_q99",
        "moments": "float64_population_mean_std",
        "video_frames_decoded": 0,
        "read_seconds": read_s,
        "compute_seconds": elapsed - read_s,
        "total_seconds": elapsed,
        "output_dir": str(output.resolve()),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [("norm_stats.json", payload), ("norm_stats_provenance.json", report)]:
        tmp = output / (name + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        tmp.replace(output / name)
    print(json.dumps(report, indent=2), flush=True)
    return report
