"""
Tutorial: Summarize per-period velocity from DeepLabCut coordinate CSVs

What this script does
1) Reads DLC coordinate CSVs (one file per video)
2) Converts frame-level coordinates into per-second averaged positions
3) Computes per-second speed (cm/s) using nose and tailbase motion
4) Computes mean velocity within named time windows ("periods")
5) Saves:
   - per-file per-second velocity traces
   - a wide summary table (one row per animal/file, one column per period)
   - a long summary table (one row per animal/file/period)

What you need to customize
- INPUT_DIRS: folders where your DLC CSVs live, grouped by video type
- PERIODS: the time windows (in seconds) for each video type
- FPS: frames per second for your videos
- PIXEL_TO_CM: camera calibration (cm per pixel)
- (Optional) SUBJECT_INFO: metadata keyed by subject ID parsed from filenames
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# 1) CONFIG YOU EDIT
# =========================

# A) Where your DLC CSVs are located (group by "video type" or condition)
INPUT_DIRS = {
    "ETEST":  "path/to/ETEST/csvs",
    "ETRAIN": "path/to/ETRAIN/csvs",
    "FCD1":   "path/to/FCD1/csvs",
    "FMT":    "path/to/FMT/csvs",
}

# B) Where outputs will be written
OUTPUT_DIR = "path/to/output_summaries"

# C) Video sampling + calibration
FPS = 8                 # frames per second
PIXEL_TO_CM = 0.1       # cm per pixel (set from your calibration)

# D) Period dictionaries (seconds): {period_name: (start_sec_inclusive, end_sec_exclusive)}
PERIODS = {
    "ETEST": {
        "Habituation": (0, 169),
        "CS1": (169, 188),
        "ITI1": (188, 262),
        "CS2": (262, 281),
        "ITI2": (281, 300),
        "CS3": (300, 319),
        "Post CS": (319, 375),
    },
    "ETRAIN": {
        "Habituation": (0, 169),
        "CS1": (169, 188),
        "ITI1": (188, 192),
        "CS2": (192, 211),
        "ITI2": (211, 215),
        "CS3": (215, 234),
        "ITI3": (234, 239),
        "CS4": (239, 258),
        "ITI4": (258, 262),
        "CS5": (262, 281),
        "ITI5": (281, 286),
        "CS6": (286, 305),
        "ITI6": (305, 309),
        "CS7": (309, 328),
        "ITI7": (328, 333),
        "CS8": (333, 352),
        "ITI8": (352, 356),
        "CS9": (356, 375),
        "ITI9": (375, 379),
        "CS10": (379, 398),
        "ITI10": (398, 403),
        "CS11": (403, 422),
        "ITI11": (422, 426),
        "CS12": (426, 445),
        "ITI12": (445, 450),
        "CS13": (450, 469),
        "ITI13": (469, 473),
        "CS14": (473, 492),
        "ITI14": (492, 497),
        "CS15": (497, 516),
        "ITI15": (516, 520),
        "CS16": (520, 539),
        "ITI16": (539, 543),
        "CS17": (543, 562),
        "ITI17": (562, 567),
        "CS18": (567, 586),
        "ITI18": (586, 590),
        "CS19": (590, 609),
        "ITI19": (609, 614),
        "CS20": (614, 633),
        "ITI20": (633, 637),
        "CS21": (637, 656),
        "ITI21": (656, 661),
        "CS22": (661, 680),
        "ITI22": (680, 684),
        "CS23": (684, 703),
        "ITI23": (703, 708),
        "CS24": (708, 727),
        "ITI24": (727, 731),
        "CS25": (731, 750),
        "Post CS": (750, 806),
    },
    "FCD1": {
        "Habituation": (0, 169),
        "CS1": (169, 188),
        "ITI1": (188, 206),
        "CS2": (206, 225),
        "ITI2": (225, 300),
        "CS3": (300, 319),
        "Post CS": (319, 375),
    },
    "FMT": {
        "Habituation": (0, 169),
        "CS1": (169, 188),
        "Post CS": (188, 243),
    },
}

# E) Optional subject metadata (keyed by subject ID parsed from filename)
# If you don't have metadata, leave this empty and the script will fill NaN.
SUBJECT_INFO = {
    # "subject_id": {"Sex": 1, "DOB": "YYYY-MM-DD", "Cohort": 8, "Ext_kHz": 16},
}

# F) Column names in your DLC CSV for nose and tailbase x/y.
# DLC exports vary; adjust these to match your header columns.
DLC_COLUMNS = {
    "nose_x": "nose",
    "nose_y": "nose.1",
    "tail_x": "tailbase",
    "tail_y": "tailbase.1",
}


# =========================
# 2) HELPERS
# =========================

def list_csvs(folder: str) -> list[str]:
    """Return sorted list of .csv files in a folder."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return sorted(str(p) for p in folder_path.iterdir() if p.suffix.lower() == ".csv")


def is_numeric_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and pd.api.types.is_numeric_dtype(df[col])


def parse_subject_id(filename: str) -> str:
    """
    Extract a subject ID from the filename.

    Default behavior:
    - Takes the part before the first underscore.
    - Strips .csv extension.

    Example:
      '91.4_something.csv' -> '91.4'

    Adjust this if your filenames follow a different convention.
    """
    stem = Path(filename).name
    return stem.split("_")[0].replace(".csv", "").replace(".CSV", "")


def trim_per_second(dlc: pd.DataFrame, fps: int) -> pd.DataFrame:
    """
    Convert frame-level DLC coordinates to per-second averaged positions.

    Output columns:
    - Second (int)
    - NoseX, NoseY, TailX, TailY (float)
    """
    df = dlc.copy().apply(pd.to_numeric, errors="coerce")

    missing = []
    for key, col in DLC_COLUMNS.items():
        if not is_numeric_col(df, col):
            missing.append((key, col))
    if missing:
        preview = list(df.columns)[:20]
        raise ValueError(
            f"Missing expected coordinate columns: {missing}. "
            f"First columns seen: {preview}"
        )

    df["Frame"] = df.index.astype("Int64")
    df["Second"] = (df["Frame"] // fps).astype("Int64")

    per_sec = (
        df.groupby("Second")
          .agg({
              DLC_COLUMNS["nose_x"]: "mean",
              DLC_COLUMNS["nose_y"]: "mean",
              DLC_COLUMNS["tail_x"]: "mean",
              DLC_COLUMNS["tail_y"]: "mean",
          })
          .reset_index()
          .rename(columns={
              DLC_COLUMNS["nose_x"]: "NoseX",
              DLC_COLUMNS["nose_y"]: "NoseY",
              DLC_COLUMNS["tail_x"]: "TailX",
              DLC_COLUMNS["tail_y"]: "TailY",
          })
    )
    return per_sec


def add_velocities(per_sec: pd.DataFrame, fps: int, pixel_to_cm: float) -> pd.DataFrame:
    """
    Add per-second speed estimates (cm/s) using successive per-second positions.

    Speed is computed from Euclidean distance between consecutive seconds' mean positions.
    """
    out = per_sec.copy()
    out["Nose_Speed"] = 0.0
    out["Tailbase_Speed"] = 0.0

    for i in range(1, len(out)):
        prev = out.iloc[i - 1]
        cur = out.iloc[i]

        nose_dist_px = np.hypot(cur["NoseX"] - prev["NoseX"], cur["NoseY"] - prev["NoseY"])
        tail_dist_px = np.hypot(cur["TailX"] - prev["TailX"], cur["TailY"] - prev["TailY"])

        # Convert to cm and then to cm/s (note: positions are per-second means;
        # multiplying by fps approximates speed relative to frame-level sampling)
        out.at[i, "Nose_Speed"] = (nose_dist_px * pixel_to_cm) * fps
        out.at[i, "Tailbase_Speed"] = (tail_dist_px * pixel_to_cm) * fps

    return out


def mean_velocity(window_df: pd.DataFrame) -> float:
    """Mean of the average of nose and tailbase speeds within a window."""
    if len(window_df) == 0:
        return np.nan
    return float(((window_df["Nose_Speed"] + window_df["Tailbase_Speed"]) / 2).mean())


def summarize_one_csv(csv_path: str, periods: dict, velocity_out_dir: str) -> dict:
    """
    For one DLC CSV:
    - Load DLC data
    - Compute per-second positions and speeds
    - Save per-second velocity trace
    - Return per-period mean velocity
    """
    dlc_raw = pd.read_csv(csv_path, skiprows=1)  # common DLC format: skip scorer row
    per_sec = trim_per_second(dlc_raw, fps=FPS)
    per_sec = add_velocities(per_sec, fps=FPS, pixel_to_cm=PIXEL_TO_CM)

    Path(velocity_out_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem
    vel_out_path = Path(velocity_out_dir, f"{stem}-velocity_data.csv")
    per_sec.to_csv(vel_out_path, index=False)

    out = {}
    for period_name, (start, end) in periods.items():
        sl = per_sec[(per_sec["Second"] >= start) & (per_sec["Second"] < end)]
        out[period_name] = mean_velocity(sl)

    return {"file": Path(csv_path).name, "velocity": out}


def to_wide_df(results: list[dict], period_names: list[str]) -> pd.DataFrame:
    """Wide table: one row per subject, one column per period."""
    rows = []
    for r in results:
        subject_id = parse_subject_id(r["file"])
        meta = SUBJECT_INFO.get(subject_id, {"Sex": np.nan, "DOB": np.nan, "Cohort": np.nan, "Ext_kHz": np.nan})
        row = {
            "Subject": subject_id,
            "Sex": meta["Sex"],
            "DOB": meta["DOB"],
            "Cohort": meta["Cohort"],
            "Ext_kHz": meta["Ext_kHz"],
        }
        for p in period_names:
            row[p] = r["velocity"].get(p, np.nan)
        rows.append(row)

    return pd.DataFrame(rows, columns=["Subject", "Sex", "DOB", "Cohort", "Ext_kHz"] + period_names)


def to_long_df(video_type: str, results: list[dict], period_names: list[str]) -> pd.DataFrame:
    """Long table: one row per subject × period."""
    rows = []
    for r in results:
        subject_id = parse_subject_id(r["file"])
        meta = SUBJECT_INFO.get(subject_id, {"Sex": np.nan, "DOB": np.nan, "Cohort": np.nan, "Ext_kHz": np.nan})
        for p in period_names:
            rows.append({
                "VideoType": video_type,
                "Subject": subject_id,
                "Sex": meta["Sex"],
                "DOB": meta["DOB"],
                "Cohort": meta["Cohort"],
                "Ext_kHz": meta["Ext_kHz"],
                "Period": p,
                "MeanVelocity_cm_s": r["velocity"].get(p, np.nan),
            })
    return pd.DataFrame(rows)


def process_video_type(video_type: str, csv_files: list[str], out_dir: str) -> None:
    """Process all CSVs for one video type and write outputs."""
    periods_dict = PERIODS[video_type]
    period_names = list(periods_dict.keys())

    results = []
    vel_trace_dir = Path(out_dir, f"{video_type}_velocity_traces")

    for csv_path in csv_files:
        try:
            results.append(summarize_one_csv(csv_path, periods_dict, velocity_out_dir=str(vel_trace_dir)))
        except Exception as e:
            print(f"[{video_type}] Skipping {csv_path} (error: {e})")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    wide_df = to_wide_df(results, period_names)
    wide_path = Path(out_dir, f"{video_type}_velocity_cmps.csv")
    wide_df.to_csv(wide_path, index=False)

    long_df = to_long_df(video_type, results, period_names)
    long_path = Path(out_dir, f"{video_type}_velocity_cmps_long.csv")
    long_df.to_csv(long_path, index=False)

    print(f"[{video_type}] Saved:")
    print(f"  - {wide_path}")
    print(f"  - {long_path}")
    print(f"  - Per-file traces in {vel_trace_dir}")


# =========================
# 3) RUN
# =========================

def main():
    print(f"Using calibration: {PIXEL_TO_CM} cm/pixel; FPS={FPS}")
    out_dir = OUTPUT_DIR

    for video_type, folder in INPUT_DIRS.items():
        csv_files = list_csvs(folder)
        if video_type not in PERIODS:
            raise KeyError(f"Missing PERIODS definition for video type: {video_type}")
        process_video_type(video_type, csv_files, out_dir)


if __name__ == "__main__":
    main()
