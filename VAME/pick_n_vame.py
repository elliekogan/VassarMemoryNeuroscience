#!/usr/bin/env python3
"""
Pick a VAME cluster count before training.

Usual order:
1) run csv_to_numpy / create_trainset
2) run this script
3) train with the chosen k

Quick examples:
  python pick_n_vame.py --project_dir /path/to/vame_project
  python pick_n_vame.py --project_dir /path/to/vame_project --write_config --verbose
  python pick_n_vame.py --project_dir /path/to/vame_project --number_only
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler


# Keep BLAS thread usage tame.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# Logging
logger = logging.getLogger("pick_n_vame")


# YAML helpers
def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict[str, Any], path: Path) -> None:
    """Save YAML config without re-sorting keys."""
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_n_clusters(cfg: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Set model.n_clusters, creating model dict if missing."""
    if "model" not in cfg or not isinstance(cfg["model"], dict):
        cfg["model"] = {}
    cfg["model"]["n_clusters"] = int(n)
    return cfg


# Find/load pose arrays
def find_pose_npys(
    project_dir: Path,
    search_roots: Tuple[str, ...] = ("data/train",),
    glob_pattern: str = "*.npy",
) -> List[Path]:
    """Find pose npy files under the given project subfolders."""
    skip_tokens = ("seq_mean", "seq_std", "scaler", "latent", "transition", "label_")

    files: List[Path] = []
    for rel in search_roots:
        root = (project_dir / rel).resolve()
        if not root.exists():
            logger.debug("Search root missing: %s", root)
            continue

        for p in glob.glob(str(root / "**" / glob_pattern), recursive=True):
            name = os.path.basename(p).lower()
            if any(tok in name for tok in skip_tokens):
                continue
            files.append(Path(p))

    if not files:
        raise FileNotFoundError(
            f"No pose .npy files found under {search_roots}. "
            "Ensure you've run vame.csv_to_numpy() / vame.create_trainset()."
        )

    return sorted(files)


def _maybe_fix_transpose(arr: np.ndarray) -> np.ndarray:
    """Flip likely transposed arrays from (D, T) to (T, D)."""
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1] and arr.shape[0] <= 64:
        return arr.T
    return arr


def load_pose_arrays(npy_paths: List[Path]) -> Tuple[List[np.ndarray], List[Path]]:
    """Load pose arrays, coerce shape where needed, and skip invalid files."""
    arrays: List[np.ndarray] = []
    kept: List[Path] = []
    skipped: List[Tuple[Path, Any]] = []

    for p in npy_paths:
        try:
            arr = np.load(p, allow_pickle=False)

            # Flatten extra feature axes while keeping frame axis first.
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)

            arr = _maybe_fix_transpose(arr)

            if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
                skipped.append((p, tuple(arr.shape)))
                continue

            arrays.append(arr.astype(np.float32))
            kept.append(p)

        except Exception as e:
            skipped.append((p, f"error: {e}"))

    if not arrays:
        raise RuntimeError(
            "No usable .npy arrays could be loaded (all were empty, wrong-shape, or errored)."
        )

    logger.info("Loaded %d arrays.", len(arrays))
    logger.debug("Loaded arrays (T, D):")
    for i, (p, a) in enumerate(zip(kept, arrays)):
        T, D = a.shape
        logger.debug("  file#%d: %s  shape=(T=%d, D=%d)", i, p, T, D)

    if skipped:
        logger.warning("Skipped %d non-usable .npy files:", len(skipped))
        for p, why in skipped:
            logger.warning("  - %s  (%s)", p, why)

    return arrays, kept


# Sliding-window features
def build_windows(X: np.ndarray, win: int, step: int) -> np.ndarray:
    """Convert a (T, D) sequence into flattened windows."""
    T, D = X.shape
    if T < win:
        return np.empty((0, win * D), dtype=np.float32)

    if step < 1:
        raise ValueError("step must be >= 1")

    # Number of windows at this stride.
    n = 1 + (T - win) // step

    # View is (n, win, D) without copying data.
    s0, s1 = X.strides
    view = np.lib.stride_tricks.as_strided(
        X,
        shape=(n, win, D),
        strides=(step * s0, s0, s1),
        writeable=False,
    )

    return view.reshape(n, win * D).astype(np.float32, copy=False)


def make_feature_matrix(
    arrs: List[np.ndarray],
    paths: List[Path],
    win: int,
    step: int,
) -> np.ndarray:
    """Build one big feature matrix; fail if per-file widths do not match."""
    mats: List[np.ndarray] = []
    widths: List[int] = []
    used_paths: List[Path] = []

    for a, p in zip(arrs, paths):
        M = build_windows(a, win, step)
        if M.size == 0:
            logger.warning("No windows produced for %s (T=%d < win=%d). Skipping.", p, a.shape[0], win)
            continue
        mats.append(M)
        widths.append(M.shape[1])
        used_paths.append(p)

    if not mats:
        raise RuntimeError("No windows produced; check your window size vs. sequence lengths.")

    if len(set(widths)) != 1:
        logger.error("Inconsistent window widths — cannot stack.")
        for p, w in zip(used_paths, widths):
            implied_D = w // max(win, 1)
            logger.error("  width=%d -> implied D_eff=%d  file=%s", w, implied_D, p)
        raise ValueError(
            "Window widths differ. This indicates per-file feature differences "
            "or per-file filtering in preprocessing. Ensure all arrays use the same "
            "column schema and that any feature masks are computed globally."
        )

    Z = np.vstack(mats)
    logger.info("Feature matrix built: Z.shape = %s", Z.shape)
    return Z


# Metrics
def centroid_corr_mean(centroids: np.ndarray) -> float:
    """Average pairwise centroid similarity."""
    if centroids.shape[0] < 2:
        return np.nan
    D = centroids.shape[1]
    C = (centroids @ centroids.T) / max(D - 1, 1)
    iu = np.triu_indices(C.shape[0], k=1)
    return float(np.nanmean(C[iu]))


def intra_corr_mean(Zs: np.ndarray, labels: np.ndarray) -> float:
    """Average within-cluster similarity."""
    rng = np.random.default_rng(0)
    vals: List[float] = []
    D = Zs.shape[1]

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = idx.size
        if n < 2:
            continue

        if n > 4000:
            # Sampling is much faster than full O(n^2) here.
            m = 200_000
            i = rng.integers(0, n, size=m)
            j = rng.integers(0, n, size=m)
            mask = i < j
            i, j = idx[i[mask]], idx[j[mask]]
            dots = np.sum(Zs[i] * Zs[j], axis=1) / max(D - 1, 1)
            vals.append(float(np.nanmean(dots)))
        else:
            Xc = Zs[idx]
            C = (Xc @ Xc.T) / max(D - 1, 1)
            iu = np.triu_indices(n, k=1)
            vals.append(float(np.nanmean(C[iu])))

    return float(np.nanmean(vals)) if vals else np.nan


def dwell_stats(labels: np.ndarray) -> Tuple[float, float]:
    """Return median and mean run length for cluster labels."""
    if labels.size == 0:
        return (np.nan, np.nan)

    runs: List[int] = []
    prev = labels[0]
    run_len = 1

    for t in range(1, labels.shape[0]):
        if labels[t] == prev:
            run_len += 1
        else:
            runs.append(run_len)
            prev = labels[t]
            run_len = 1

    runs.append(run_len)
    r = np.asarray(runs, dtype=float)
    return (float(np.median(r)), float(np.mean(r)))


def occupancy_and_entropy(labels: np.ndarray, k: int) -> Tuple[float, float]:
    """Return min occupancy and normalized label entropy."""
    counts = np.bincount(labels, minlength=k).astype(float)
    p = counts / max(counts.sum(), 1.0)

    min_occ = float(p.min()) if p.size else np.nan

    pp = p[p > 0]
    ent = -np.sum(pp * np.log(pp)) if pp.size else 0.0
    entropy_norm = float(ent / np.log(k)) if k > 1 and ent > 0 else 0.0

    return min_occ, entropy_norm


# Evaluate each k across seeds
def evaluate_k_grid(
    Z: np.ndarray,
    ks: List[int],
    seeds: Tuple[int, ...],
) -> List[Dict[str, float]]:
    """Compute clustering metrics for each k."""
    Z = np.asarray(Z, dtype=np.float32)
    Zs = StandardScaler().fit_transform(Z)

    results: List[Dict[str, float]] = []

    for k in ks:
        labels_by_seed: List[np.ndarray] = []
        cents_by_seed: List[np.ndarray] = []

        for s in seeds:
            km = KMeans(n_clusters=k, n_init=10, random_state=s)
            labels = km.fit_predict(Zs)
            labels_by_seed.append(labels)
            cents_by_seed.append(km.cluster_centers_)

        # Stability across seeds.
        aris: List[float] = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                aris.append(adjusted_rand_score(labels_by_seed[i], labels_by_seed[j]))
        ari = float(np.mean(aris)) if aris else np.nan

        # Keep other metrics tied to one seed for consistency/speed.
        labels0 = labels_by_seed[0]
        cents0 = cents_by_seed[0]

        intra = intra_corr_mean(Zs, labels0)
        inter = centroid_corr_mean(cents0)
        delta = (intra - inter) if np.isfinite(intra) and np.isfinite(inter) else np.nan

        # Standard clustering indices.
        try:
            sil = float(silhouette_score(Zs, labels0, metric="cosine"))
        except Exception:
            sil = np.nan

        try:
            db = float(davies_bouldin_score(Zs, labels0))
        except Exception:
            db = np.nan

        try:
            ch = float(calinski_harabasz_score(Zs, labels0))
        except Exception:
            ch = np.nan

        median_dwell, mean_dwell = dwell_stats(labels0)
        min_occ, entropy_norm = occupancy_and_entropy(labels0, k)

        results.append(
            {
                "k": float(k),
                "delta_corr": float(delta) if np.isfinite(delta) else np.nan,
                "intra_corr": float(intra) if np.isfinite(intra) else np.nan,
                "inter_corr": float(inter) if np.isfinite(inter) else np.nan,
                "silhouette_cos": float(sil) if np.isfinite(sil) else np.nan,
                "davies_bouldin": float(db) if np.isfinite(db) else np.nan,
                "calinski_harabasz": float(ch) if np.isfinite(ch) else np.nan,
                "ari_stability": float(ari) if np.isfinite(ari) else np.nan,
                "median_dwell": float(median_dwell),
                "mean_dwell": float(mean_dwell),
                "min_occ": float(min_occ),
                "entropy_norm": float(entropy_norm),
            }
        )

    return results


# Selection policy
def pick_best_k(
    results: List[Dict[str, float]],
    win: int,
    prefer_negative_inter: bool = True,
    dwell_floor: Optional[int] = None,
    min_occ_floor: float = 0.005,
    entropy_floor: float = 0.70,
) -> int:
    """Choose k using hard floors first, then metric ranking."""
    if not results:
        raise RuntimeError("No results to pick from.")

    dwell_req = dwell_floor if dwell_floor is not None else max(10, win // 2)

    def finite_or(v: float, fallback: float) -> float:
        return v if np.isfinite(v) else fallback

    def sort_key(r: Dict[str, float]) -> Tuple[float, float, float, float]:
        return (
            -finite_or(r["delta_corr"], -np.inf),
            -finite_or(r["ari_stability"], -np.inf),
            -finite_or(r["silhouette_cos"], -np.inf),
            finite_or(r["davies_bouldin"], np.inf),
        )

    def apply_filters(
        rs: Iterable[Dict[str, float]],
        require_inter: bool = True,
        require_entropy: bool = True,
        require_minocc: bool = True,
        require_dwell: bool = True,
    ) -> List[Dict[str, float]]:
        out = list(rs)

        if require_dwell:
            out = [r for r in out if r["median_dwell"] >= dwell_req]

        if require_minocc:
            out = [r for r in out if np.isfinite(r["min_occ"]) and r["min_occ"] >= min_occ_floor]

        if require_entropy:
            out = [r for r in out if np.isfinite(r["entropy_norm"]) and r["entropy_norm"] >= entropy_floor]

        if require_inter and prefer_negative_inter:
            out = [r for r in out if np.isfinite(r["inter_corr"]) and r["inter_corr"] <= 0]

        return out

    strategies = [
        dict(require_inter=True,  require_entropy=True,  require_minocc=True,  require_dwell=True),
        dict(require_inter=True,  require_entropy=False, require_minocc=True,  require_dwell=True),
        dict(require_inter=False, require_entropy=True,  require_minocc=True,  require_dwell=True),
        dict(require_inter=False, require_entropy=False, require_minocc=True,  require_dwell=True),
        dict(require_inter=False, require_entropy=False, require_minocc=False, require_dwell=True),
        dict(require_inter=False, require_entropy=False, require_minocc=False, require_dwell=False),
    ]

    for st in strategies:
        cands = apply_filters(results, **st)
        if cands:
            best = sorted(cands, key=sort_key)[0]
            return int(best["k"])

    # Last-resort fallback.
    best = sorted(results, key=sort_key)[0]
    return int(best["k"])


# CLI
def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick n_clusters for VAME before training.")
    ap.add_argument("--project_dir", required=True, help="Path to VAME project directory.")
    ap.add_argument("--config", help="Path to config.yaml (default: <project_dir>/config.yaml)")

    ap.add_argument("--ks", default="25,30,35,40,45,50,55",
                    help="Comma-separated integers for k grid. Example: 12,13,14,15,16")
    ap.add_argument("--window", type=int,
                    help="Window size; default uses config['model']['time_window'] else 30.")
    ap.add_argument("--step", type=int, default=1, help="Window stride (default 1).")
    ap.add_argument("--seeds", default="0,1,2,3", help="Comma-separated seeds.")

    ap.add_argument("--write_config", action="store_true",
                    help="Write chosen n into config.yaml (model.n_clusters).")
    ap.add_argument("--number_only", action="store_true",
                    help="Print only the chosen integer k.")

    # Selection thresholds.
    ap.add_argument("--dwell_floor", type=int, default=None,
                    help="Minimum median dwell in frames/windows (default=max(10, window//2)).")
    ap.add_argument("--min_occ", type=float, default=0.005,
                    help="Minimum per-cluster occupancy fraction (default 0.005 = 0.5%).")
    ap.add_argument("--entropy_floor", type=float, default=0.70,
                    help="Minimum normalized usage entropy in [0,1] (default 0.70).")

    # Search location for input arrays.
    ap.add_argument("--search_roots", default="data/train",
                    help="Comma-separated subdirs (relative to project_dir) to search for .npy files. "
                         "Default: data/train")
    ap.add_argument("--glob", default="*.npy",
                    help="Filename glob to include (default: *.npy).")

    ap.add_argument("--verbose", action="store_true",
                    help="Enable debug logging.")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    project_dir = Path(args.project_dir).resolve()
    cfg_path = Path(args.config).resolve() if args.config else (project_dir / "config.yaml")

    if not cfg_path.exists():
        logger.error("Config not found at %s", cfg_path)
        sys.exit(1)

    cfg = load_config(cfg_path)

    # CLI window overrides config; default to 30 if missing.
    win = args.window or cfg.get("model", {}).get("time_window") or cfg.get("model", {}).get("window") or 30
    step = int(args.step)
    ks = parse_int_list(args.ks)
    seeds = tuple(parse_int_list(args.seeds))

    roots = tuple(s.strip() for s in args.search_roots.split(",") if s.strip())

    npy_files = find_pose_npys(project_dir, search_roots=roots, glob_pattern=args.glob)
    arrays, paths = load_pose_arrays(npy_files)

    # Guard: per-frame feature dimension must match across files.
    Ds = [a.shape[1] for a in arrays]
    if len(set(Ds)) != 1:
        logger.error("Inconsistent per-frame feature dimension D across files:")
        for p, D in zip(paths, Ds):
            logger.error("  %s  D=%d", p, D)
        raise ValueError(
            "Per-file D differs. Ensure all arrays have the same column schema "
            "(same markers/features, same preprocessing)."
        )
    logger.info("All files share D=%d", Ds[0])

    # Build feature matrix.
    Z = make_feature_matrix(arrays, paths, win=win, step=step)

    # Evaluate candidates and select best k.
    results = evaluate_k_grid(Z, ks=ks, seeds=seeds)
    best_k = pick_best_k(
        results,
        win=win,
        prefer_negative_inter=True,
        dwell_floor=args.dwell_floor,
        min_occ_floor=args.min_occ,
        entropy_floor=args.entropy_floor,
    )

    if args.write_config:
        cfg = set_n_clusters(cfg, best_k)
        save_config(cfg, cfg_path)

    if args.number_only:
        print(best_k)
        return

    # Results table.
    header = (
        "k", "delta_corr", "intra_corr", "inter_corr", "silhouette_cos",
        "davies_bouldin", "calinski_harabasz", "ari_stability",
        "median_dwell", "mean_dwell", "min_occ", "entropy_norm"
    )

    print(f"\nProject: {project_dir}")
    print(f"Window={win}  Step={step}")
    print("Results:")
    print("\t".join(header))

    for r in sorted(results, key=lambda x: x["k"]):
        def fmt(v: float) -> str:
            return f"{v:.12g}" if isinstance(v, float) else str(v)
        print("\t".join(fmt(float(r[h])) for h in header))

    print(f"\nChosen n_clusters = {best_k}")
    if args.write_config:
        print(f"Updated {cfg_path} with model.n_clusters = {best_k}")


if __name__ == "__main__":
    main()
