from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

from config.action_library import get_action_list
from models.attack_dynamics import batch_calculate_dynamics
from models.attack_scoring import score_actions


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


def build_feature_dataframe(
    action_priors: List[Dict[str, Any]],
    dynamics_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    构造动作特征表：
    - 原始动作先验
    - 动力学指标
    """
    prior_df = pd.DataFrame(action_priors)
    dyn_df = pd.DataFrame(dynamics_results)

    # 避免重复列冲突：保留 dynamics 为主，priors 中已存在的基础列可不重复拼接
    merged_df = pd.merge(
        prior_df,
        dyn_df,
        on=["code", "name", "category", "part"],
        how="left",
        suffixes=("_prior", "")
    )

    return merged_df


def build_ranking_dataframe(
    scored_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    构造排名表，并挑出最适合论文展示的列。
    """
    df = pd.DataFrame(scored_results)

    preferred_columns = [
        "rank",
        "code",
        "name",
        "category",
        "part",
        "attack_gain_score",
        "stability_penalty_score",
        "energy_penalty_score",
        "stability_benefit_score",
        "energy_efficiency_score",
        "total_score",
        "terminal_speed",
        "momentum",
        "kinetic_energy",
        "avg_impact_force",
        "support_loss_proxy",
        "rotation_risk_proxy",
        "recovery_burden_proxy",
        "exposure_proxy",
        "impact_level",
        "accuracy_level",
        "balance_break_level",
        "combo_potential",
        "pressure_potential",
        "balance_risk",
        "time_cost",
        "energy_cost",
        "counter_risk",
    ]

    existing_columns = [c for c in preferred_columns if c in df.columns]
    df = df[existing_columns]

    return df


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    导出 CSV，使用 utf_8_sig 便于 Excel 直接打开中文不乱码。
    """
    df.to_csv(output_path, index=False, encoding="utf_8_sig")


def plot_action_ranking(ranking_df: pd.DataFrame, output_path: Path) -> None:
    """
    绘制综合得分柱状图。
    """
    plt.figure(figsize=(12, 7))

    names = ranking_df["name"]
    scores = ranking_df["total_score"]

    bars = plt.bar(names, scores, edgecolor="black")

    plt.title("Q1 攻击动作综合评分排序", fontsize=16)
    plt.xlabel("攻击动作", fontsize=12)
    plt.ylabel("综合得分", fontsize=12)
    plt.xticks(rotation=35, ha="right")

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_summary(ranking_df: pd.DataFrame, top_n: int = 5) -> None:
    """
    在终端打印 Top N 结果，方便快速查看。
    """
    print("\n================= Q1 结果摘要 =================")
    print(f"动作总数: {len(ranking_df)}")
    print(f"Top {top_n} 动作：\n")

    for _, row in ranking_df.head(top_n).iterrows():
        print(
            f"Rank {int(row['rank'])}: {row['name']}"
            f" | total={row['total_score']:.2f}"
            f" | attack={row['attack_gain_score']:.2f}"
            f" | stability_penalty={row['stability_penalty_score']:.2f}"
            f" | energy_penalty={row['energy_penalty_score']:.2f}"
        )

    print("================================================\n")


def main() -> None:
    print("run_q1.py 开始执行...")

    # 1. 读取动作库
    action_priors = get_action_list()
    print(f"已读取动作数量: {len(action_priors)}")

    # 2. 计算动力学与稳定性代理量
    dynamics_results = batch_calculate_dynamics(action_priors)
    print("已完成动力学指标计算")

    # 3. 综合评分与排序
    scored_results = score_actions(dynamics_results, action_priors=action_priors)
    print("已完成综合评分与排序")

    # 4. 构造 DataFrame
    features_df = build_feature_dataframe(action_priors, dynamics_results)
    ranking_df = build_ranking_dataframe(scored_results)

    # 5. 导出结果
    features_csv_path = TABLE_DIR / "action_features.csv"
    ranking_csv_path = TABLE_DIR / "action_ranking.csv"
    figure_path = FIGURE_DIR / "q1_action_ranking.png"

    export_csv(features_df, features_csv_path)
    export_csv(ranking_df, ranking_csv_path)
    plot_action_ranking(ranking_df, figure_path)

    # 6. 终端摘要
    print_summary(ranking_df, top_n=5)

    print("文件已导出：")
    print(f"- {features_csv_path}")
    print(f"- {ranking_csv_path}")
    print(f"- {figure_path}")
    print("\nrun_q1.py 执行完成。")


if __name__ == "__main__":
    main()