"""
Standalone re-analysis driver for L-IPS v2.6 1CKK long-convergence trajectories.
Does NOT run any MD. Reuses analyze_1ckk() from test_lips_vs_pmeV2.6.py against
the ALREADY-EXISTING LONG5ns_*_seed*_1ckk.{dcd,csv} files.

Covers README_v2.6_1CKK_CONCLUSION.md "下一步" items 1-2:
  1. Re-run full-trajectory metrics with the fixed compute_water_q_profile()
     (it now force-keeps every water within NEAR_CA_CUTOFF_NM of Ca before
     linspace-subsampling bulk water), so ca_near_water_delta_QO_mean no longer
     silently drops to NaN (this previously happened for CWLD_aq0p5_ca_w2 seed0).
  2. Split each 5 ns trajectory into 5 x 1 ns blocks, and separately recompute
     using only the last 4 ns (i.e. dropping block 0), to check whether the
     PME / aq0-w1 / aq0-w2 / aq0.5-w2 ranking is stable over time or drifts.

Does NOT touch the original 1ckk_v26_long_convergence_5ns_{per_seed_metrics,
summary}.csv. Writes new files instead (see bottom of main() for the list) so
you can diff before deciding whether to replace the originals.

Inputs (LONG5ns_*.dcd / .csv) are read from L_IPS_DATA_DIR, which defaults to this
script's own directory. Since the 2026-08-31 move the code lives in L-IPS/2.5 while
the old trajectories stayed in L-IPS/2.4, so re-analysing them needs the env var.
Outputs are always written next to this script, never into the data dir.

No GPU needed -- this only re-derives structural/dynamical metrics from saved
coordinates, it never steps a Simulation:

    python block_time_check_v26.py                                   # data in 2.5
    L_IPS_DATA_DIR=/home/ruigengji/L-IPS/2.4 python block_time_check_v26.py
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import mdtraj as md

from lips import paths
from lips.paths import PROJECT_ROOT, DATA_DIR as _DATA_DIR
SCRIPT_DIR = str(PROJECT_ROOT)   # 兼容旧写法；新代码请直接用 lips.paths


# 输入轨迹目录：默认脚本同目录；2026-08-31 之后代码在 L-IPS/2.5、旧 dcd 在 L-IPS/2.4，
# 所以复分析旧轨迹要显式指到 2.4。只影响读，输出一律写 SCRIPT_DIR，不污染数据目录。
DATA_DIR = str(_DATA_DIR)
if not os.path.isdir(DATA_DIR):
    raise SystemExit(f"L_IPS_DATA_DIR 指向的目录不存在: {DATA_DIR}")

from lips import systems as ls

#: 本模块用哪个参考体系。1CKK 已非当前体系，默认跟随 lips.systems。
SYSTEM_NAME = ls.DEFAULT_SYSTEM
from lips.engine import v26 as lips  # MD/main() 在 `if __name__ == "__main__"` 后面，import 安全

N_BLOCKS = 5
SEEDS = lips.REPLICATE_SEEDS

CONFIGS = {
    "PME": dict(
        label="LONG5ns_PME", a_q2=None, water_response=False,
        solute_polarization=lips.ENABLE_SOLUTE_POLARIZATION,
        ca_source_weight=lips.CA_SOURCE_WEIGHT,
    ),
    "CWLD_aq0_ca_w1": dict(
        label="LONG5ns_CWLD_aq0_ca_w1", a_q2=lips.PURE_DENSITY_A_Q2, water_response=True,
        solute_polarization=lips.ENABLE_SOLUTE_POLARIZATION, ca_source_weight=1.0,
    ),
    "CWLD_aq0_ca_w2": dict(
        label="LONG5ns_CWLD_aq0_ca_w2", a_q2=lips.PURE_DENSITY_A_Q2, water_response=True,
        solute_polarization=lips.ENABLE_SOLUTE_POLARIZATION, ca_source_weight=2.0,
    ),
    "CWLD_aq0p5_ca_w2": dict(
        label="LONG5ns_CWLD_aq0p5_ca_w2", a_q2=lips.DEFAULT_A_Q2, water_response=True,
        solute_polarization=lips.ENABLE_SOLUTE_POLARIZATION, ca_source_weight=2.0,
    ),
}


def build_system_for_config(cfg, sys_pme, topology):
    if cfg["a_q2"] is None:
        return sys_pme
    # engine="customgb": this compares against the pre-2026-09-05 block-time
    # matrix, and the two engines are not bitwise identical. Switch to
    # engine="plugin" only when the whole matrix is regenerated.
    return lips.setup_cwld_lips_system(
        sys_pme, topology, engine="customgb",
        a_q2=cfg["a_q2"],
        enable_water_response=cfg["water_response"],
        enable_solute_polarization=cfg["solute_polarization"],
        ca_source_weight=cfg["ca_source_weight"],
    )


def clean_state_columns(df):
    df = df.copy()
    df.columns = [c.strip().strip('"').lstrip("#").strip() for c in df.columns]
    return df


def slice_to_temp_files(traj, state_df, frame_slice, tmp_dir, tag):
    sub_traj = traj[frame_slice]
    dcd_path = os.path.join(tmp_dir, f"{tag}_1ckk.dcd")
    csv_path = os.path.join(tmp_dir, f"{tag}_1ckk.csv")
    sub_traj.save_dcd(dcd_path)
    state_df.iloc[frame_slice].to_csv(csv_path, index=False)
    return dcd_path, csv_path


def run_one(run_label, dcd_path, csv_path, topology, system, cfg):
    analysis = lips.analyze_1ckk(
        run_label, dcd_path, csv_path, topology, system,
        a_q2=cfg["a_q2"],
        enable_water_response=cfg["water_response"],
        enable_solute_polarization=cfg["solute_polarization"],
        ca_source_weight=cfg["ca_source_weight"],
        q_profile_mode="exact",
    )
    return dict(analysis["metrics"])


def summarize(df, group_cols):
    if df.empty:
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("seed", errors="ignore")
    summary = df.groupby(group_cols)[numeric_cols].agg(["count", "mean", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary


def main():
    # 体系是参数：1CKK 已非当前体系，且这条路径原本没有离子名规范化。
    topology, sys_pme = ls.load_reference_system(name=SYSTEM_NAME)

    full_rows, block_rows, last4ns_rows = [], [], []

    tmp_dir = tempfile.mkdtemp(prefix="block_time_check_", dir=SCRIPT_DIR)
    print(f"[info] input data dir : {DATA_DIR}" + ("  (= script dir)" if DATA_DIR == SCRIPT_DIR else "  (via L_IPS_DATA_DIR)"))
    print(f"[info] output dir     : {SCRIPT_DIR}")
    print(f"[info] temp slice dir: {tmp_dir}")

    try:
        for config_name, cfg in CONFIGS.items():
            system = build_system_for_config(cfg, sys_pme, topology)
            for seed in SEEDS:
                run_label_base = f"{cfg['label']}_seed{seed}"
                dcd_file = os.path.join(DATA_DIR, f"{run_label_base}_1ckk.dcd")
                csv_file = os.path.join(DATA_DIR, f"{run_label_base}_1ckk.csv")
                if not (os.path.exists(dcd_file) and os.path.exists(csv_file)):
                    print(f"[skip] missing {dcd_file} or {csv_file}")
                    continue

                print(f"[load] {run_label_base}")
                md_top = md.Topology.from_openmm(topology)
                traj_full = md.load(dcd_file, top=md_top).image_molecules(inplace=False)
                state_df_full = clean_state_columns(pd.read_csv(csv_file))
                n_frames = traj_full.n_frames
                if len(state_df_full) < n_frames:
                    print(f"[warn] {run_label_base}: csv rows ({len(state_df_full)}) < dcd frames "
                          f"({n_frames}); truncating trajectory to match csv.")
                    n_frames = len(state_df_full)
                    traj_full = traj_full[:n_frames]

                # --- (1) full trajectory, recomputed with fixed q-profile sampling ---
                dcd_path, csv_path = slice_to_temp_files(
                    traj_full, state_df_full, slice(0, n_frames), tmp_dir, f"{run_label_base}_full")
                metrics = run_one(f"{run_label_base}_full_qfix", dcd_path, csv_path, topology, system, cfg)
                metrics.update(config=config_name, seed=seed, scope="full_5ns_qfix",
                                protocol_version="v2.6-block-time-check-v1")
                full_rows.append(metrics)

                # --- (2) 5 x 1ns blocks ---
                frames_per_block = n_frames // N_BLOCKS
                if frames_per_block == 0:
                    print(f"[warn] {run_label_base}: only {n_frames} frames, cannot split into "
                          f"{N_BLOCKS} blocks; skipping block/last-4ns analysis for this trajectory.")
                    continue

                for block_idx in range(N_BLOCKS):
                    start = block_idx * frames_per_block
                    end = n_frames if block_idx == N_BLOCKS - 1 else (block_idx + 1) * frames_per_block
                    dcd_path, csv_path = slice_to_temp_files(
                        traj_full, state_df_full, slice(start, end), tmp_dir,
                        f"{run_label_base}_block{block_idx}")
                    metrics = run_one(f"{run_label_base}_block{block_idx}", dcd_path, csv_path, topology, system, cfg)
                    metrics.update(config=config_name, seed=seed, scope=f"block{block_idx}",
                                    block_index=block_idx, frame_start=start, frame_end=end,
                                    protocol_version="v2.6-block-time-check-v1")
                    block_rows.append(metrics)

                # --- (3) last-4ns-only (everything after block 0) ---
                dcd_path, csv_path = slice_to_temp_files(
                    traj_full, state_df_full, slice(frames_per_block, n_frames), tmp_dir,
                    f"{run_label_base}_last4ns")
                metrics = run_one(f"{run_label_base}_last4ns", dcd_path, csv_path, topology, system, cfg)
                metrics.update(config=config_name, seed=seed, scope="last_4ns",
                                frame_start=frames_per_block, frame_end=n_frames,
                                protocol_version="v2.6-block-time-check-v1")
                last4ns_rows.append(metrics)

        full_df = pd.DataFrame(full_rows)
        block_df = pd.DataFrame(block_rows)
        last4ns_df = pd.DataFrame(last4ns_rows)

        full_df.to_csv(paths.result("1ckk_v26_long_convergence_5ns_per_seed_metrics_QFIX.csv"), index=False)
        block_df.to_csv(paths.result("1ckk_v26_block_time_check_per_block_metrics.csv"), index=False)
        last4ns_df.to_csv(paths.result("1ckk_v26_block_time_check_last4ns_metrics.csv"), index=False)

        summarize(full_df, ["config"]).to_csv(
            paths.result("1ckk_v26_long_convergence_5ns_summary_QFIX.csv"))
        summarize(block_df, ["config", "scope"]).to_csv(
            paths.result("1ckk_v26_block_time_check_block_summary.csv"))
        summarize(last4ns_df, ["config"]).to_csv(
            paths.result("1ckk_v26_block_time_check_last4ns_summary.csv"))

        print("\n[done] wrote:")
        print("  1ckk_v26_long_convergence_5ns_per_seed_metrics_QFIX.csv  (full 5ns, fixed q-profile)")
        print("  1ckk_v26_long_convergence_5ns_summary_QFIX.csv")
        print("  1ckk_v26_block_time_check_per_block_metrics.csv          (5 x 1ns blocks)")
        print("  1ckk_v26_block_time_check_block_summary.csv")
        print("  1ckk_v26_block_time_check_last4ns_metrics.csv            (last 4ns only)")
        print("  1ckk_v26_block_time_check_last4ns_summary.csv")
        print("\nNext: diff the _QFIX files against the original")
        print("1ckk_v26_long_convergence_5ns_{per_seed_metrics,summary}.csv (should mainly just fill in")
        print("previously-missing ca_near_water_* values), then compare last4ns_summary against the")
        print("original 5ns summary and the per-block summary to check whether the PME/aq0-w1/aq0-w2/")
        print("aq0.5-w2 ranking on ca_water_peak_g / coord / residence / salt-bridge drifts over time")
        print("(README_v2.6_1CKK_CONCLUSION.md, 下一步 items 1-3).")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
