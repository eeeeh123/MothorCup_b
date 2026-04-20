from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulators.bo3_simulator import simulate_bo3, simulate_many_bo3
from simulators.round_simulator import PolicyFn, get_policy_name
from optimizers.policy_q3 import greedy_q3_policy
from optimizers.resource_policy_q4 import (
    protective_q3_policy,
    momentum_q3_policy,
    risk_aware_q3_policy,
)
from simulators.round_simulator import simple_rule_policy


# -----------------------------
# 路径配置
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Matplotlib 中文显示设置
# -----------------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------
# 实验配置
# -----------------------------
# 由于现在改为 5 种策略 round-robin 对比，默认样本数建议适度下调
DEFAULT_N_RUNS = 50

# 是否包含自对弈（如 greedy vs greedy）
INCLUDE_SELF_PLAY = True

# 是否生成镜像实验（A_vs_B 与 B_vs_A 都跑）
# 默认 False，减少运行时间；若要排除先后手影响，可设 True
INCLUDE_MIRROR_MATCH = False

# 直接参与对比的 Q3 策略集合
POLICY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "greedy_q3_policy": {
        "policy": greedy_q3_policy,
        "short_name": "greedy",
        "description": "一步前瞻的主策略基线",
    },
    "simple_rule_policy": {
        "policy": simple_rule_policy,
        "short_name": "simple_rule",
        "description": "轻量规则策略基线",
    },
    "protective_q3_policy": {
        "policy": protective_q3_policy,
        "short_name": "protective",
        "description": "保护型策略：优先防守/恢复，仅在明显优势时追击",
    },
    "momentum_q3_policy": {
        "policy": momentum_q3_policy,
        "short_name": "momentum",
        "description": "动量型策略：主动权/对手失衡时加强压制，否则退回 simple_rule",
    },
    "risk_aware_q3_policy": {
        "policy": risk_aware_q3_policy,
        "short_name": "risk_aware",
        "description": "风险感知策略：greedy 为主，但在高风险动作上做状态过滤",
    },
}

COMPARE_POLICY_ORDER = [
    "greedy_q3_policy",
    "simple_rule_policy",
    "protective_q3_policy",
    "momentum_q3_policy",
    "risk_aware_q3_policy",
]


def get_experiment_configs() -> List[Dict[str, Any]]:
    """
    生成 greedy / simple_rule / protective / momentum / risk_aware 的直接对比实验组。

    默认行为：
    - 跑上三角 round-robin（避免重复）
    - 包含自对弈
    - 不跑镜像实验
    """
    experiments: List[Dict[str, Any]] = []

    for i, my_name in enumerate(COMPARE_POLICY_ORDER):
        start_j = i if INCLUDE_SELF_PLAY else i + 1
        for j in range(start_j, len(COMPARE_POLICY_ORDER)):
            opp_name = COMPARE_POLICY_ORDER[j]

            my_item = POLICY_REGISTRY[my_name]
            opp_item = POLICY_REGISTRY[opp_name]

            exp_name = f"q4_bo3_{my_item['short_name']}_vs_{opp_item['short_name']}"
            description = f"BO3: {my_item['short_name']} 对 {opp_item['short_name']}"

            experiments.append(
                {
                    "exp_name": exp_name,
                    "my_policy": my_item["policy"],
                    "opp_policy": opp_item["policy"],
                    "description": description,
                    "my_policy_name": my_name,
                    "opp_policy_name": opp_name,
                }
            )

            if INCLUDE_MIRROR_MATCH and my_name != opp_name:
                mirror_exp_name = f"q4_bo3_{opp_item['short_name']}_vs_{my_item['short_name']}"
                mirror_description = f"BO3: {opp_item['short_name']} 对 {my_item['short_name']}"
                experiments.append(
                    {
                        "exp_name": mirror_exp_name,
                        "my_policy": opp_item["policy"],
                        "opp_policy": my_item["policy"],
                        "description": mirror_description,
                        "my_policy_name": opp_name,
                        "opp_policy_name": my_name,
                    }
                )

    return experiments


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, encoding="utf_8_sig")


def build_bo3_round_summary_dataframe(bo3_result) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for summary in bo3_result.round_summaries:
        rows.append(summary.to_dict())
    return pd.DataFrame(rows)


def build_resource_usage_dataframe(bo3_result) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in bo3_result.all_resource_usage:
        rows.append(item.to_dict())
    return pd.DataFrame(rows)


def build_resource_usage_by_round_dataframe(bo3_result) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for summary in bo3_result.round_summaries:
        base = {
            "round_index": summary.round_index,
            "my_policy": summary.my_policy,
            "opp_policy": summary.opp_policy,
            "winner": summary.winner,
            "win_reason": summary.win_reason,
            "scoreline_after_round": summary.scoreline_after_round,
        }

        count_reset_my = sum(
            1 for x in summary.resource_usage if x.side == "my" and x.resource_type == "reset"
        )
        count_timeout_my = sum(
            1 for x in summary.resource_usage if x.side == "my" and x.resource_type == "timeout"
        )
        count_repair_my = sum(
            1 for x in summary.resource_usage if x.side == "my" and x.resource_type == "repair"
        )

        count_reset_opp = sum(
            1 for x in summary.resource_usage if x.side == "opp" and x.resource_type == "reset"
        )
        count_timeout_opp = sum(
            1 for x in summary.resource_usage if x.side == "opp" and x.resource_type == "timeout"
        )
        count_repair_opp = sum(
            1 for x in summary.resource_usage if x.side == "opp" and x.resource_type == "repair"
        )

        row = dict(base)
        row.update(
            {
                "my_reset_count": count_reset_my,
                "my_timeout_count": count_timeout_my,
                "my_repair_count": count_repair_my,
                "opp_reset_count": count_reset_opp,
                "opp_timeout_count": count_timeout_opp,
                "opp_repair_count": count_repair_opp,
                "resource_usage_count": len(summary.resource_usage),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_series_comparison_row(
    exp_name: str,
    description: str,
    my_policy: PolicyFn,
    opp_policy: PolicyFn,
    single_result,
    many_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "experiment": exp_name,
        "description": description,
        "my_policy": get_policy_name(my_policy),
        "opp_policy": get_policy_name(opp_policy),
        "single_winner": single_result.winner,
        "single_reason": single_result.win_reason,
        "single_rounds_played": single_result.rounds_played,
        "single_final_scoreline": single_result.final_bo3_state.scoreline(),
        "single_my_round_wins": single_result.final_bo3_state.my.round_wins,
        "single_opp_round_wins": single_result.final_bo3_state.opp.round_wins,
        "single_my_fault_level": single_result.final_bo3_state.my.fault_level.value,
        "single_opp_fault_level": single_result.final_bo3_state.opp.fault_level.value,
        "single_resource_usage_count": len(single_result.all_resource_usage),
        "n_runs": many_result["n_runs"],
        "my_series_win_rate": many_result["my_series_win_rate"],
        "opp_series_win_rate": many_result["opp_series_win_rate"],
        "draw_series_rate": many_result["draw_series_rate"],
        "avg_rounds_played": many_result["avg_rounds_played"],
        "resource_usage_reset_total": many_result["resource_usage_totals"].get("reset", 0),
        "resource_usage_timeout_total": many_result["resource_usage_totals"].get("timeout", 0),
        "resource_usage_repair_total": many_result["resource_usage_totals"].get("repair", 0),
    }


def build_series_reason_rows(exp_name: str, many_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    reasons = many_result.get("series_win_reason_distribution", {})
    n_runs = max(1, int(many_result.get("n_runs", 1)))

    for reason, count in reasons.items():
        rows.append(
            {
                "experiment": exp_name,
                "series_win_reason": reason,
                "count": count,
                "ratio": count / n_runs,
            }
        )
    return rows


def build_scoreline_rows(exp_name: str, many_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scoreline_dist = many_result.get("final_scoreline_distribution", {})
    n_runs = max(1, int(many_result.get("n_runs", 1)))

    for scoreline, count in scoreline_dist.items():
        rows.append(
            {
                "experiment": exp_name,
                "scoreline": scoreline,
                "count": count,
                "ratio": count / n_runs,
            }
        )
    return rows


def build_resource_total_rows(exp_name: str, many_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    totals = many_result.get("resource_usage_totals", {})
    n_runs = max(1, int(many_result.get("n_runs", 1)))

    for resource_type, count in totals.items():
        rows.append(
            {
                "experiment": exp_name,
                "resource_type": resource_type,
                "count": count,
                "avg_per_series": count / n_runs,
            }
        )
    return rows


def plot_series_win_rate_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return

    x = np.arange(len(comparison_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(comparison_df) * 0.8), 7))
    ax.bar(x - width, comparison_df["my_series_win_rate"], width, label="my_series_win_rate", edgecolor="black")
    ax.bar(x, comparison_df["opp_series_win_rate"], width, label="opp_series_win_rate", edgecolor="black")
    ax.bar(x + width, comparison_df["draw_series_rate"], width, label="draw_series_rate", edgecolor="black")

    ax.set_title("Q4 不同 BO3 策略组合胜率对比", fontsize=16)
    ax.set_xlabel("实验组", fontsize=12)
    ax.set_ylabel("比例", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["experiment"], rotation=30, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_avg_rounds_played(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(comparison_df) * 0.75), 6))
    bars = ax.bar(comparison_df["experiment"], comparison_df["avg_rounds_played"], edgecolor="black")

    ax.set_title("Q4 不同 BO3 策略组合平均局数", fontsize=16)
    ax.set_xlabel("实验组", fontsize=12)
    ax.set_ylabel("avg_rounds_played", fontsize=12)
    plt.xticks(rotation=30, ha="right")

    for bar, value in zip(bars, comparison_df["avg_rounds_played"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_resource_usage_comparison(resource_totals_df: pd.DataFrame, output_path: Path) -> None:
    if resource_totals_df.empty:
        return

    pivot_df = (
        resource_totals_df.pivot(index="experiment", columns="resource_type", values="avg_per_series")
        .fillna(0.0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(max(12, len(pivot_df) * 0.8), 7))
    bottom = np.zeros(len(pivot_df))

    for col in pivot_df.columns:
        values = pivot_df[col].to_numpy(dtype=float)
        ax.bar(pivot_df.index, values, bottom=bottom, label=col, edgecolor="black")
        bottom += values

    ax.set_title("Q4 BO3 资源使用强度对比", fontsize=16)
    ax.set_xlabel("实验组", fontsize=12)
    ax.set_ylabel("avg_per_series", fontsize=12)
    ax.legend()

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scoreline_distribution(scoreline_df: pd.DataFrame, output_path: Path) -> None:
    if scoreline_df.empty:
        return

    pivot_df = (
        scoreline_df.pivot(index="experiment", columns="scoreline", values="ratio")
        .fillna(0.0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(max(12, len(pivot_df) * 0.8), 7))
    bottom = np.zeros(len(pivot_df))

    for col in pivot_df.columns:
        values = pivot_df[col].to_numpy(dtype=float)
        ax.bar(pivot_df.index, values, bottom=bottom, label=col, edgecolor="black")
        bottom += values

    ax.set_title("Q4 BO3 最终比分分布", fontsize=16)
    ax.set_xlabel("实验组", fontsize=12)
    ax.set_ylabel("比例", fontsize=12)
    ax.legend()

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_series_summary(
    exp_name: str,
    description: str,
    my_policy: PolicyFn,
    opp_policy: PolicyFn,
    single_result,
    many_result: Dict[str, Any],
) -> None:
    print(f"\n================= {exp_name} =================")
    print("description:", description)
    print("my base policy:", get_policy_name(my_policy))
    print("opp base policy:", get_policy_name(opp_policy))

    print("\n[单次 BO3]")
    print("winner:", single_result.winner)
    print("reason:", single_result.win_reason)
    print("rounds played:", single_result.rounds_played)
    print("final scoreline:", single_result.final_bo3_state.scoreline())
    print("my fault:", single_result.final_bo3_state.my.fault_level.value)
    print("opp fault:", single_result.final_bo3_state.opp.fault_level.value)
    print("resource usage count:", len(single_result.all_resource_usage))

    print("\n[多次 BO3]")
    print("my series win rate:", round(many_result["my_series_win_rate"], 3))
    print("opp series win rate:", round(many_result["opp_series_win_rate"], 3))
    print("draw series rate:", round(many_result["draw_series_rate"], 3))
    print("avg rounds played:", round(many_result["avg_rounds_played"], 3))
    print("series reasons:", many_result["series_win_reason_distribution"])
    print("resource usage totals:", many_result["resource_usage_totals"])
    print("scoreline dist:", many_result["final_scoreline_distribution"])
    print("================================================")


def run_one_experiment(
    exp_name: str,
    description: str,
    my_policy: PolicyFn,
    opp_policy: PolicyFn,
    n_runs: int = DEFAULT_N_RUNS,
    single_seed: int = 42,
    many_seed: int = 123,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    single_result = simulate_bo3(
        my_base_policy=my_policy,
        opp_base_policy=opp_policy,
        mode="sample",
        seed=single_seed,
    )

    many_result = simulate_many_bo3(
        n_runs=n_runs,
        my_base_policy=my_policy,
        opp_base_policy=opp_policy,
        mode="sample",
        seed=many_seed,
    )

    comparison_row = build_series_comparison_row(
        exp_name=exp_name,
        description=description,
        my_policy=my_policy,
        opp_policy=opp_policy,
        single_result=single_result,
        many_result=many_result,
    )
    reason_rows = build_series_reason_rows(exp_name, many_result)
    scoreline_rows = build_scoreline_rows(exp_name, many_result)
    resource_total_rows = build_resource_total_rows(exp_name, many_result)

    round_summary_df = build_bo3_round_summary_dataframe(single_result)
    resource_usage_df = build_resource_usage_dataframe(single_result)
    resource_by_round_df = build_resource_usage_by_round_dataframe(single_result)

    round_summary_path = TABLE_DIR / f"{exp_name}_single_bo3_round_summary.csv"
    resource_usage_path = TABLE_DIR / f"{exp_name}_single_bo3_resource_usage.csv"
    resource_by_round_path = TABLE_DIR / f"{exp_name}_single_bo3_resource_usage_by_round.csv"

    export_csv(round_summary_df, round_summary_path)
    export_csv(resource_usage_df, resource_usage_path)
    export_csv(resource_by_round_df, resource_by_round_path)

    print_series_summary(
        exp_name=exp_name,
        description=description,
        my_policy=my_policy,
        opp_policy=opp_policy,
        single_result=single_result,
        many_result=many_result,
    )
    print("单次 BO3 文件已导出：")
    print("-", round_summary_path)
    print("-", resource_usage_path)
    print("-", resource_by_round_path)

    return comparison_row, reason_rows, scoreline_rows, resource_total_rows


def main() -> None:
    print("run_q4.py 开始执行（多策略 round-robin 版）...")

    experiments = get_experiment_configs()
    print(f"实验组数量: {len(experiments)}")
    print(f"每组多次 BO3 样本数: {DEFAULT_N_RUNS}")
    print("参与策略:", ", ".join(COMPARE_POLICY_ORDER))

    comparison_rows: List[Dict[str, Any]] = []
    reason_rows: List[Dict[str, Any]] = []
    scoreline_rows: List[Dict[str, Any]] = []
    resource_total_rows: List[Dict[str, Any]] = []

    total_exps = len(experiments)
    for idx, exp in enumerate(experiments, start=1):
        print(f"\n>>> [{idx}/{total_exps}] 开始实验: {exp['exp_name']}")
        row, rows_reason, rows_scoreline, rows_resource = run_one_experiment(
            exp_name=exp["exp_name"],
            description=exp["description"],
            my_policy=exp["my_policy"],
            opp_policy=exp["opp_policy"],
            n_runs=DEFAULT_N_RUNS,
        )
        comparison_rows.append(row)
        reason_rows.extend(rows_reason)
        scoreline_rows.extend(rows_scoreline)
        resource_total_rows.extend(rows_resource)
        print(f">>> [{idx}/{total_exps}] 完成实验: {exp['exp_name']}")

    comparison_df = pd.DataFrame(comparison_rows)
    reason_df = pd.DataFrame(reason_rows)
    scoreline_df = pd.DataFrame(scoreline_rows)
    resource_totals_df = pd.DataFrame(resource_total_rows)

    comparison_csv_path = TABLE_DIR / "q4_bo3_strategy_comparison.csv"
    reason_csv_path = TABLE_DIR / "q4_bo3_series_win_reason_distribution.csv"
    scoreline_csv_path = TABLE_DIR / "q4_bo3_scoreline_distribution.csv"
    resource_csv_path = TABLE_DIR / "q4_bo3_resource_usage_totals.csv"

    export_csv(comparison_df, comparison_csv_path)
    export_csv(reason_df, reason_csv_path)
    export_csv(scoreline_df, scoreline_csv_path)
    export_csv(resource_totals_df, resource_csv_path)

    win_rate_fig_path = FIGURE_DIR / "q4_bo3_series_win_rate_comparison.png"
    avg_rounds_fig_path = FIGURE_DIR / "q4_bo3_avg_rounds_played.png"
    resource_fig_path = FIGURE_DIR / "q4_bo3_resource_usage_comparison.png"
    scoreline_fig_path = FIGURE_DIR / "q4_bo3_scoreline_distribution.png"

    plot_series_win_rate_comparison(comparison_df, win_rate_fig_path)
    plot_avg_rounds_played(comparison_df, avg_rounds_fig_path)
    plot_resource_usage_comparison(resource_totals_df, resource_fig_path)
    plot_scoreline_distribution(scoreline_df, scoreline_fig_path)

    print("\n================= Q4 汇总结果 =================")
    if not comparison_df.empty:
        shown_cols = [
            "experiment",
            "my_policy",
            "opp_policy",
            "my_series_win_rate",
            "opp_series_win_rate",
            "draw_series_rate",
            "avg_rounds_played",
            "resource_usage_reset_total",
            "resource_usage_timeout_total",
            "resource_usage_repair_total",
        ]
        print(comparison_df[shown_cols].round(4).to_string(index=False))
    print("==============================================")

    print("\n文件已导出：")
    print(f"- {comparison_csv_path}")
    print(f"- {reason_csv_path}")
    print(f"- {scoreline_csv_path}")
    print(f"- {resource_csv_path}")
    print(f"- {win_rate_fig_path}")
    print(f"- {avg_rounds_fig_path}")
    print(f"- {resource_fig_path}")
    print(f"- {scoreline_fig_path}")
    print("\nrun_q4.py 执行完成。")


if __name__ == "__main__":
    main()
