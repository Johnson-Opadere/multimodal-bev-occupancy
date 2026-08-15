"""
report_utils.py
----------------
Reusable experiment reporting and visualization utilities
for Phases 1–7 (RGB, LiDAR, Fusion, Hybrid).

Features:
---------
UTF-8 safe Markdown + CSV exports
Training start/end/duration tracking
Automatic leaderboard generation
Automatic leaderboard visualization
Numeric coercion for safety (fixes TypeError)
Legacy + modern compatibility

Usage:
-------
from src.report_utils import save_report, merge_reports, plot_leaderboard
"""

import os
import io
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import datetime as dt


# ============================================================
#  Base Report Writer (Markdown + CSV, UTF-8 Safe)
# ============================================================
def save_report(
    summary_dict,
    robustness_dict=None,
    model_name="Model",
    best_iou=None,
    start_time=None,
    end_time=None,
    auto_update=False,
):
    """
    Unified safe report writer (legacy + modern compatible).
    Supports:
      - robustness_dict as dict  → new-style breakdown
      - robustness_dict as str   → legacy output path
    """

    # ---  Fix: handle legacy string argument BEFORE .items() call ---
    if isinstance(robustness_dict, str):
        out_path = robustness_dict        
        robustness_dict = None
    else:
        out_path = None

    # --- Duration tracking ---
    duration_min = None
    if start_time and end_time:
        duration_min = round((end_time - start_time) / 60, 2)

    # --- Ensure numeric types (avoid string issues) ---
    clean_summary = {}
    for k, v in summary_dict.items():
        try:
            clean_summary[k] = float(v)
        except Exception:
            clean_summary[k] = v

    # --- Summary table ---
    summary_df = pd.DataFrame({
        "Metric": list(clean_summary.keys()),
        "Value":  list(clean_summary.values())
    })

    # --- Robustness table (safe) ---
    if isinstance(robustness_dict, dict) and robustness_dict:
        rob_df = pd.DataFrame(list(robustness_dict.items()), columns=["Condition", "Mean IoU"])
    else:
        rob_df = pd.DataFrame(columns=["Condition", "Mean IoU"])

    # ------------------------------------------------------------
    # Output filenames
    #
    # Portfolio version:
    # Reports are overwritten on each run to keep only the latest
    # evaluation artifacts.
    # ------------------------------------------------------------
    os.makedirs("reports", exist_ok=True)

    safe_name = (
        model_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "_")
        .replace("/", "_")
        .lower()
    )

    md_path = f"reports/{safe_name}_summary.md"
    csv_path = f"reports/{safe_name}_summary.csv"

    # --- Markdown export ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    with io.open(md_path, "w", encoding="utf-8") as f:
        f.write(f"#  {model_name} Evaluation Report\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        if start_time and end_time:
            f.write(f"**Elapsed Duration:** {duration_min:.2f} min\n\n")
        f.write("## Quantitative Metrics\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Robustness Results\n")
        f.write(rob_df.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write(f" **Model:** {model_name}\n")
        if best_iou is not None:
            f.write(f" **Best Val IoU:** {best_iou:.4f}\n")

    summary_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f" Markdown report saved → {md_path}")
    print(f" CSV report saved → {csv_path}")

    # --- Generate project leaderboard ---
    if auto_update:
        leaderboard = merge_reports()
        if leaderboard is not None:
            plot_leaderboard(leaderboard)
            

    # --- Optional JSON export for legacy calls ---
    if out_path:
        with open(out_path, "w") as jf:
            json.dump(clean_summary, jf, indent=4)
        print(f" JSON report saved → {out_path}")

    return md_path, csv_path


# ============================================================
#  Report Merger: Aggregate All Reports into One Leaderboard
# ============================================================
def merge_reports(report_dir="reports", out_prefix="leaderboard"):
    """Merge all CSV reports into a single leaderboard summary."""
    os.makedirs(report_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(report_dir, "*_summary.csv")))
    if not csv_files:
        print("No report CSV files found to merge.")
        return None

    all_rows = []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)

            # --- Coerce Value column to numeric ---
            if "Value" in df.columns:
                df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

            model_name = os.path.basename(fpath).split("_summary_")[0]
            
            mean_iou = df.loc[df["Metric"].str.contains("IoU", case=False, na=False), "Value"].mean()

            # Try to extract duration (from Markdown if available)
            md_path = fpath.replace(".csv", ".md")
            model_name = None

            if os.path.exists(md_path):
                with io.open(md_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith(" **Model:**"):
                            model_name = line.replace("**Model:**", "").strip()
                            break

            if model_name is None:
                model_name = (
                    os.path.basename(fpath)
                    .replace("_summary.csv", "")
                    .replace("_", " ")
                    .title()
                )
            duration_min = None
            if os.path.exists(md_path):
                with io.open(md_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "Elapsed Duration" in line or "Training Duration" in line:
                            try:
                                duration_min = float(line.strip().split()[-2])
                                break
                            except:
                                continue
            
            all_rows.append({
                "Model": model_name,
                "Mean IoU": round(float(mean_iou), 4) if pd.notna(mean_iou) else None,
                "Duration (min)": duration_min
            })
        except Exception as e:
            print(f"[WARN] Skipping {fpath}: {e}")

    if not all_rows:
        print("No valid reports to aggregate.")
        return None

    leaderboard_df = pd.DataFrame(all_rows).dropna(subset=["Mean IoU"])
    leaderboard_df = leaderboard_df.sort_values(by="Mean IoU", ascending=False)

    # ------------------------------------------------------------
    # Save merged leaderboard
    #
    # Portfolio version:
    # The leaderboard always reflects the latest experiment set.
    # ------------------------------------------------------------
    csv_out = os.path.join(report_dir, f"{out_prefix}.csv")
    md_out = os.path.join(report_dir, f"{out_prefix}.md")    

    leaderboard_df.to_csv(csv_out, index=False, encoding="utf-8")
    with io.open(md_out, "w", encoding="utf-8") as f:
        f.write("#  Model Leaderboard Summary\n\n")
        f.write(leaderboard_df.to_markdown(index=False))
        f.write("\n\n_Auto-generated by `merge_reports()`_\n")

    print(f" Leaderboard saved → {csv_out}")
    print(f"️ Markdown summary → {md_out}")
    return leaderboard_df


# ============================================================
#  Visualization: Leaderboard Bar Plot
# ============================================================
def plot_leaderboard(leaderboard_df, report_dir="reports"):
    """Create a horizontal bar plot for the leaderboard summary."""
    if leaderboard_df is None or leaderboard_df.empty:
        print("No leaderboard data available to plot.")
        return

    os.makedirs(report_dir, exist_ok=True)
    leaderboard_df = leaderboard_df.sort_values(by="Mean IoU", ascending=True)

    plt.figure(figsize=(8, 4))
    bars = plt.barh(
        leaderboard_df["Model"],
        leaderboard_df["Mean IoU"],
        color=[
            "#0099CC" if "fusion" in m.lower() else "#66BB6A" if "lidar" in m.lower() else "#FFA726"
            for m in leaderboard_df["Model"]
        ],
    )

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{width:.3f}", va="center", fontsize=9, color="black")

    plt.title("Model Leaderboard — Mean IoU", fontsize=14, pad=15)
    plt.xlabel("Mean IoU", fontsize=12)
    plt.ylabel("Model", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(report_dir, "leaderboard.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" Leaderboard plot saved → {out_path}")


