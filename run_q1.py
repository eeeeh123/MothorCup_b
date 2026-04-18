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
    构造动作全量特征表：
    - 最终表对齐后的动作底层连续变量
    - 动力学/稳定性代理量
    """
    prior_df = pd.DataFrame(action_priors)
    dyn_df = pd.DataFrame(dynamics_results)

    merged_df = pd.merge(
        prior_df,
        dyn_df,
        on=["code", "name", "category", "part"],
        how="left",
        suffixes=("_prior", "_dyn"),
    )

    preferred_columns = [
        "code", "name", "category", "part", "attack_range", "target_zones",
        "eta", "omega", "dt",
        "Ek", "F", "Tpre", "Lreach", "theta_sweep", "Kbreak", "Ccombo", "Vvul", "Pfall", "Trec", "Cstam",
        "impact_level", "accuracy_level", "balance_break_level", "combo_potential", "time_cost", "balance_risk", "energy_cost", "counter_risk", "pressure_potential",
        "tactical_total_score", "confidence", "data_source", "notes",
        "limb_mass", "limb_length", "equivalent_mass", "terminal_speed", "momentum", "kinetic_energy", "avg_impact_force",
        "support_loss_proxy", "rotation_risk_proxy", "recovery_burden_proxy", "exposure_proxy",
    ]
    existing_columns = [c for c in preferred_columns if c in merged_df.columns]
    return merged_df[existing_columns]


def build_ranking_dataframe(scored_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    构造 Q1 排名表，保留最适合论文与结果展示的字段。
    """
    df = pd.DataFrame(scored_results)

    preferred_columns = [
        "rank", "code", "name", "category", "part", "attack_range",
        "total_score", "attack_gain_score", "stability_penalty_score", "energy_penalty_score",
        "stability_benefit_score", "energy_efficiency_score",
        "F", "Ek", "momentum", "Kbreak", "Ccombo", "Lreach", "Tpre", "Trec", "Cstam", "Pfall", "Vvul",
        "terminal_speed", "avg_impact_force",
        "impact_level", "accuracy_level", "balance_break_level", "combo_potential", "pressure_potential",
        "support_loss_proxy", "rotation_risk_proxy", "recovery_burden_proxy", "exposure_proxy",
        "tactical_total_score", "confidence", "data_source", "notes",
    ]
    existing_columns = [c for c in preferred_columns if c in df.columns]
    return df[existing_columns]


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, encoding="utf_8_sig")


def plot_action_ranking(ranking_df: pd.DataFrame, output_path: Path) -> None:
    if ranking_df.empty:
        return

    plt.figure(figsize=(12, 7))

    names = ranking_df["name"]
    scores = ranking_df["total_score"]

    bars = plt.bar(names, scores, edgecolor="black")
    plt.title("Q1 攻击动作综合评分排序（最终表对齐版）", fontsize=16)
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


def plot_core_metrics(ranking_df: pd.DataFrame, output_path: Path) -> None:
    if ranking_df.empty:
        return

    core_cols = [c for c in ["F", "Ek", "Pfall", "Cstam"] if c in ranking_df.columns]
    if len(core_cols) < 2:
        return

    plot_df = ranking_df.set_index("name")[core_cols]
    ax = plot_df.plot(kind="bar", figsize=(14, 7), edgecolor="black")
    ax.set_title("Q1 核心底层指标对比", fontsize=16)
    ax.set_xlabel("攻击动作", fontsize=12)
    ax.set_ylabel("指标值", fontsize=12)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_summary(ranking_df: pd.DataFrame, top_n: int = 5) -> None:
    print("\n================= Q1 结果摘要 =================")
    print(f"动作总数: {len(ranking_df)}")
    print(f"Top {top_n} 动作：\n")

    for _, row in ranking_df.head(top_n).iterrows():
        extra_parts = []
        for field in ["F", "Ek", "Pfall", "Cstam"]:
            if field in row and pd.notna(row[field]):
                val = row[field]
                if field in {"F", "Ek", "Cstam"}:
                    extra_parts.append(f"{field}={val:.1f}")
                else:
                    extra_parts.append(f"{field}={val:.3f}")

        extras = " | " + " | ".join(extra_parts) if extra_parts else ""

        print(
            f"Rank {int(row['rank'])}: {row['name']}"
            f" | total={row['total_score']:.2f}"
            f" | attack={row['attack_gain_score']:.2f}"
            f" | stability_penalty={row['stability_penalty_score']:.2f}"
            f" | energy_penalty={row['energy_penalty_score']:.2f}"
            f"{extras}"
        )

    print("================================================\n")


def main() -> None:
    print("run_q1.py 开始执行（最终表对齐版）...")

    # 1. 读取最终表对齐后的动作库
    action_priors = get_action_list()
    print(f"已读取动作数量: {len(action_priors)}")

    # 2. 计算动力学与稳定性代理量（表优先，公式兜底）
    dynamics_results = batch_calculate_dynamics(action_priors)
    print("已完成动力学指标计算")

    # 3. 综合评分与排序（连续变量为主，映射字段为辅）
    scored_results = score_actions(dynamics_results, action_priors=action_priors)
    print("已完成综合评分与排序")

    # 4. 构造 DataFrame
    features_df = build_feature_dataframe(action_priors, dynamics_results)
    ranking_df = build_ranking_dataframe(scored_results)

    # 5. 导出结果
    features_csv_path = TABLE_DIR / "q1_action_features_final_aligned.csv"
    ranking_csv_path = TABLE_DIR / "q1_action_ranking_final_aligned.csv"
    ranking_fig_path = FIGURE_DIR / "q1_action_ranking_final_aligned.png"
    metrics_fig_path = FIGURE_DIR / "q1_action_core_metrics_final_aligned.png"

    export_csv(features_df, features_csv_path)
    export_csv(ranking_df, ranking_csv_path)
    plot_action_ranking(ranking_df, ranking_fig_path)
    plot_core_metrics(ranking_df, metrics_fig_path)

    # 6. 终端摘要
    print_summary(ranking_df, top_n=5)

    print("文件已导出：")
    print(f"- {features_csv_path}")
    print(f"- {ranking_csv_path}")
    print(f"- {ranking_fig_path}")
    print(f"- {metrics_fig_path}")
    print("\nrun_q1.py 执行完成。")


if __name__ == "__main__":
    main()
