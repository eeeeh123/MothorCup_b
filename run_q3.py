from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from simulators.round_simulator import (
    simulate_round,
    simulate_many_rounds,
    simple_rule_policy,
    random_policy,
    get_policy_name,
    PolicyFn,
)
from optimizers.policy_q3 import greedy_q3_policy, make_rollout_policy


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
DEFAULT_N_RUNS = 200

# 是否额外启用 rollout 策略对照
ENABLE_ROLLOUT_EXPERIMENT = False

ROLLOUT_POLICY = make_rollout_policy(
    rollout_steps=3,
    n_rollouts_per_action=8,
    gamma=0.92,
)


def get_experiment_configs() -> List[Dict[str, Any]]:
    experiments: List[Dict[str, Any]] = [
        {
            "exp_name": "q3_greedy_vs_simple_rule",
            "my_policy": greedy_q3_policy,
            "opp_policy": simple_rule_policy,
            "description": "greedy 对 simple_rule",
        },
        {
            "exp_name": "q3_simple_rule_vs_simple_rule",
            "my_policy": simple_rule_policy,
            "opp_policy": simple_rule_policy,
            "description": "simple_rule 对 simple_rule",
        },
        {
            "exp_name": "q3_greedy_vs_greedy",
            "my_policy": greedy_q3_policy,
            "opp_policy": greedy_q3_policy,
            "description": "greedy 对 greedy",
        },
        {
            "exp_name": "q3_greedy_vs_random",
            "my_policy": greedy_q3_policy,
            "opp_policy": random_policy,
            "description": "greedy 对 random",
        },
    ]

    if ENABLE_ROLLOUT_EXPERIMENT:
        experiments.extend(
            [
                {
                    "exp_name": "q3_rollout_vs_simple_rule",
                    "my_policy": ROLLOUT_POLICY,
                    "opp_policy": simple_rule_policy,
                    "description": "rollout 对 simple_rule",
                },
                {
                    "exp_name": "q3_rollout_vs_greedy",
                    "my_policy": ROLLOUT_POLICY,
                    "opp_policy": greedy_q3_policy,
                    "description": "rollout 对 greedy",
                },
            ]
        )

    return experiments


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, encoding="utf_8_sig")


def build_single_round_trace_dataframe(single_result) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [record.to_dict() for record in single_result.step_records]
    return pd.DataFrame(rows)


def build_comparison_row(
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
        "single_steps": single_result.total_steps,
        "single_reward_my": single_result.total_reward_my,
        "single_reward_opp": single_result.total_reward_opp,
        "single_final_my_hp": single_result.final_state.my.hp_proxy,
        "single_final_my_stability": single_result.final_state.my.stability,
        "single_final_my_energy": single_result.final_state.my.energy,
        "single_final_opp_hp": single_result.final_state.opp.hp_proxy,
        "single_final_opp_stability": single_result.final_state.opp.stability,
        "single_final_opp_energy": single_result.final_state.opp.energy,
        "n_runs": many_result["n_runs"],
        "my_win_rate": many_result["my_win_rate"],
        "opp_win_rate": many_result["opp_win_rate"],
        "draw_rate": many_result["draw_rate"],
        "avg_reward_my": many_result["avg_reward_my"],
        "avg_reward_opp": many_result["avg_reward_opp"],
        "avg_step_count": many_result["avg_step_count"],
    }


def build_win_reason_rows(exp_name: str, many_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    reasons = many_result.get("win_reason_distribution", {})
    n_runs = max(1, int(many_result.get("n_runs", 1)))

    for reason, count in reasons.items():
        rows.append(
            {
                "experiment": exp_name,
                "win_reason": reason,
                "count": count,
                "ratio": count / n_runs,
            }
        )
    return rows


def plot_win_rate_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return

    x = np.arange(len(comparison_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, comparison_df["my_win_rate"], width, label="my_win_rate", edgecolor="black")
    ax.bar(x, comparison_df["opp_win_rate"], width, label="opp_win_rate", edgecolor="black")
    ax.bar(x + width, comparison_df["draw_rate"], width, label="draw_rate", edgecolor="black")

    ax.set_title("Q3 不同策略组合胜率对比", fontsize=16)
    ax.set_xlabel("实验组", fontsize=12)
    ax.set_ylabel("比例", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["experiment"], rotation=20, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_avg_step_count(comparison_df: pd.DataFrame, output_path: Path) -> None:
    if comparison_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(comparison_df["experiment"], comparison_df["avg_step_count"], edgecolor="black")

    ax.set_title("Q3 不同策略组合平均步数", fontsize=16)
    ax.set_xlabel("实验组", fontsize=12)
    ax.set_ylabel("avg_step_count", fontsize=12)
    ax.set_xticklabels(comparison_df["experiment"], rotation=20, ha="right")

    for bar, value in zip(bars, comparison_df["avg_step_count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_win_reason_stacked(reason_df: pd.DataFrame, output_path: Path) -> None:
    if reason_df.empty:
        return

    pivot_df = (
        reason_df.pivot(index="experiment", columns="win_reason", values="ratio")
        .fillna(0.0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(pivot_df))

    for col in pivot_df.columns:
        values = pivot_df[col].to_numpy(dtype=float)
        ax.bar(pivot_df.index, values, bottom=bottom, label=col, edgecolor="black")
        bottom += values

    ax.set_title("Q3 不同策略组合终局原因分布", fontsize=16)
    ax.set_xlabel("实验组", fontsize=12)
    ax.set_ylabel("比例", fontsize=12)
    ax.legend()

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_experiment_summary(
    exp_name: str,
    description: str,
    my_policy: PolicyFn,
    opp_policy: PolicyFn,
    single_result,
    many_result: Dict[str, Any],
) -> None:
    print(f"\n================= {exp_name} =================")
    print("description:", description)
    print("my policy:", get_policy_name(my_policy))
    print("opp policy:", get_policy_name(opp_policy))

    print("\n[单局 expected 模式]")
    print("winner:", single_result.winner)
    print("reason:", single_result.win_reason)
    print("steps:", single_result.total_steps)
    print("my total reward:", round(single_result.total_reward_my, 3))
    print("opp total reward:", round(single_result.total_reward_opp, 3))
    print(
        "final my hp/stability/energy:",
        round(single_result.final_state.my.hp_proxy, 3),
        round(single_result.final_state.my.stability, 3),
        round(single_result.final_state.my.energy, 3),
    )
    print(
        "final opp hp/stability/energy:",
        round(single_result.final_state.opp.hp_proxy, 3),
        round(single_result.final_state.opp.stability, 3),
        round(single_result.final_state.opp.energy, 3),
    )

    print("\n[多局 sample 模式]")
    print("my win rate:", round(many_result["my_win_rate"], 3))
    print("opp win rate:", round(many_result["opp_win_rate"], 3))
    print("draw rate:", round(many_result["draw_rate"], 3))
    print("avg reward my:", round(many_result["avg_reward_my"], 3))
    print("avg reward opp:", round(many_result["avg_reward_opp"], 3))
    print("avg step count:", round(many_result["avg_step_count"], 3))
    print("win reasons:", many_result["win_reason_distribution"])
    print("================================================")


def run_one_experiment(
    exp_name: str,
    description: str,
    my_policy: PolicyFn,
    opp_policy: PolicyFn,
    n_runs: int = DEFAULT_N_RUNS,
    single_seed: int = 42,
    many_seed: int = 123,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    single_result = simulate_round(
        my_policy=my_policy,
        opp_policy=opp_policy,
        mode="expected",
        seed=single_seed,
    )

    many_result = simulate_many_rounds(
        n_runs=n_runs,
        my_policy=my_policy,
        opp_policy=opp_policy,
        mode="sample",
        seed=many_seed,
    )

    comparison_row = build_comparison_row(
        exp_name=exp_name,
        description=description,
        my_policy=my_policy,
        opp_policy=opp_policy,
        single_result=single_result,
        many_result=many_result,
    )

    reason_rows = build_win_reason_rows(exp_name, many_result)
    trace_df = build_single_round_trace_dataframe(single_result)

    # 导出单局轨迹
    trace_path = TABLE_DIR / f"{exp_name}_single_round_trace.csv"
    export_csv(trace_df, trace_path)

    print_experiment_summary(
        exp_name=exp_name,
        description=description,
        my_policy=my_policy,
        opp_policy=opp_policy,
        single_result=single_result,
        many_result=many_result,
    )

    print("单局轨迹文件已导出：", trace_path)

    return comparison_row, reason_rows, trace_df


def main() -> None:
    print("run_q3.py 开始执行...")

    experiments = get_experiment_configs()
    print(f"实验组数量: {len(experiments)}")

    comparison_rows: List[Dict[str, Any]] = []
    reason_rows: List[Dict[str, Any]] = []

    for exp in experiments:
        row, rows_reason, _ = run_one_experiment(
            exp_name=exp["exp_name"],
            description=exp["description"],
            my_policy=exp["my_policy"],
            opp_policy=exp["opp_policy"],
            n_runs=DEFAULT_N_RUNS,
        )
        comparison_rows.append(row)
        reason_rows.extend(rows_reason)

    comparison_df = pd.DataFrame(comparison_rows)
    reason_df = pd.DataFrame(reason_rows)

    comparison_csv_path = TABLE_DIR / "q3_strategy_comparison.csv"
    reason_csv_path = TABLE_DIR / "q3_win_reason_distribution.csv"

    export_csv(comparison_df, comparison_csv_path)
    export_csv(reason_df, reason_csv_path)

    win_rate_fig_path = FIGURE_DIR / "q3_win_rate_comparison.png"
    avg_step_fig_path = FIGURE_DIR / "q3_avg_step_count.png"
    win_reason_fig_path = FIGURE_DIR / "q3_win_reason_stacked_bar.png"

    plot_win_rate_comparison(comparison_df, win_rate_fig_path)
    plot_avg_step_count(comparison_df, avg_step_fig_path)
    plot_win_reason_stacked(reason_df, win_reason_fig_path)

    print("\n================= Q3 汇总结果 =================")
    if not comparison_df.empty:
        shown_cols = [
            "experiment",
            "my_policy",
            "opp_policy",
            "my_win_rate",
            "opp_win_rate",
            "draw_rate",
            "avg_reward_my",
            "avg_reward_opp",
            "avg_step_count",
        ]
        print(comparison_df[shown_cols].round(4).to_string(index=False))
    print("==============================================")

    print("\n文件已导出：")
    print(f"- {comparison_csv_path}")
    print(f"- {reason_csv_path}")
    print(f"- {win_rate_fig_path}")
    print(f"- {avg_step_fig_path}")
    print(f"- {win_reason_fig_path}")
    print("\nrun_q3.py 执行完成。")


if __name__ == "__main__":
    main()
