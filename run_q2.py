from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from models.matchup_matrix import (
    prepare_attack_records,
    prepare_defense_records,
    build_matchup_matrix,
    get_top_defenses_for_attack,
    summarize_top_defenses,
)


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


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    """导出 CSV，使用 utf_8_sig 便于 Excel 直接打开中文不乱码。"""
    df.to_csv(output_path, index=False, encoding="utf_8_sig")


def build_wide_matchup_matrix(matchup_long_df: pd.DataFrame) -> pd.DataFrame:
    """
    将长表转换为 13 × 22 的宽表矩阵，值为 matchup_score。
    第一列保留 attack_name，方便直接导出与绘图。
    """
    if matchup_long_df.empty:
        return pd.DataFrame(columns=["attack_name"])

    wide = matchup_long_df.pivot_table(
        index="attack_name",
        columns="defense_name",
        values="matchup_score",
        aggfunc="first",
    ).reset_index()

    defense_cols = [c for c in wide.columns if c != "attack_name"]
    defense_cols_sorted = sorted(defense_cols)
    wide = wide[["attack_name"] + defense_cols_sorted]
    return wide


def extract_top_defenses(matchup_long_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    从长表中提取每个攻击动作的 Top-N 防守动作。
    排序优先级与 matchup_matrix 模块保持一致：
    1. matchup_score 降序
    2. explicit_match 降序
    3. counter_gain_0_10 降序
    4. fail_loss_0_10 升序
    """
    if matchup_long_df.empty:
        return pd.DataFrame()

    df = matchup_long_df.copy()
    df["counter_gain_0_10"] = df["counter_gain_0_10"].fillna(0.0)
    df["fail_loss_0_10"] = df["fail_loss_0_10"].fillna(10.0)

    df = df.sort_values(
        by=["attack_name", "matchup_score", "explicit_match", "counter_gain_0_10", "fail_loss_0_10"],
        ascending=[True, False, False, False, True],
    )

    top_df = df.groupby("attack_name", as_index=False, group_keys=False).head(top_n).copy()
    top_df["rank"] = top_df.groupby("attack_name").cumcount() + 1

    preferred_columns = [
        "attack_code",
        "attack_name",
        "attack_category",
        "rank",
        "defense_code",
        "defense_name",
        "defense_category",
        "matchup_score",
        "explicit_match",
        "reference_score_0_5",
        "final_score_0_10",
        "counter_gain_0_10",
        "fail_loss_0_10",
        "reason",
        "source",
        "confidence",
        "remark",
    ]
    cols = [c for c in preferred_columns if c in top_df.columns]
    return top_df[cols].reset_index(drop=True)


def build_attack_summary_table(
    matchup_long_df: pd.DataFrame,
    attack_records: List[Dict[str, Any]],
    top_n: int = 3,
) -> pd.DataFrame:
    """
    构造更适合论文文字摘要的表：每个攻击动作一行，列出 Top1~TopN 防守动作及原因。
    """
    rows: List[Dict[str, Any]] = []
    attack_name_order = [r["name"] for r in attack_records]

    for attack_name in attack_name_order:
        sub = extract_top_defenses(matchup_long_df[matchup_long_df["attack_name"] == attack_name], top_n=top_n)
        row: Dict[str, Any] = {"attack_name": attack_name}
        for i in range(top_n):
            if i < len(sub):
                row[f"top{i+1}_defense"] = sub.iloc[i]["defense_name"]
                row[f"top{i+1}_score"] = sub.iloc[i]["matchup_score"]
                row[f"top{i+1}_reason"] = sub.iloc[i]["reason"]
            else:
                row[f"top{i+1}_defense"] = ""
                row[f"top{i+1}_score"] = ""
                row[f"top{i+1}_reason"] = ""
        rows.append(row)

    return pd.DataFrame(rows)


def plot_matchup_heatmap(wide_df: pd.DataFrame, output_path: Path) -> None:
    """
    绘制攻击-防守匹配热力图。
    不使用 seaborn，保持依赖简单。
    """
    if wide_df.empty:
        return

    plot_df = wide_df.copy()
    attack_names = plot_df["attack_name"].tolist()
    defense_names = [c for c in plot_df.columns if c != "attack_name"]

    values = plot_df[defense_names].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(values, aspect="auto")

    ax.set_title("Q2 攻击-防守匹配热力图", fontsize=16)
    ax.set_xlabel("防守动作", fontsize=12)
    ax.set_ylabel("攻击动作", fontsize=12)

    ax.set_xticks(np.arange(len(defense_names)))
    ax.set_xticklabels(defense_names, rotation=60, ha="right", fontsize=9)

    ax.set_yticks(np.arange(len(attack_names)))
    ax.set_yticklabels(attack_names, fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("matchup_score", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_q2_summary(top3_df: pd.DataFrame, attack_limit: int = 5) -> None:
    """
    在终端打印前若干个攻击动作的 Top 3 防守动作摘要。
    """
    print("\n================= Q2 结果摘要 =================")
    attack_names = top3_df["attack_name"].drop_duplicates().tolist()

    shown = 0
    for attack_name in attack_names:
        if shown >= attack_limit:
            break

        print(f"\n【{attack_name}】")
        sub = (
            top3_df[top3_df["attack_name"] == attack_name]
            .sort_values(by="rank")
            .reset_index(drop=True)
        )

        for _, row in sub.iterrows():
            print(
                f"  Rank {int(row['rank'])}: {row['defense_name']}"
                f" | score={row['matchup_score']:.2f}"
                f" | explicit={int(row['explicit_match'])}"
                f" | counter={row['counter_gain_0_10']:.1f}"
                f" | fail_loss={row['fail_loss_0_10']:.1f}"
                f" | reason={row['reason']}"
            )

        shown += 1

    print("\n================================================\n")


def main() -> None:
    print("run_q2.py 开始执行...")

    # 1. 准备攻击动作记录（来自最终版 action_library）
    attack_records = prepare_attack_records()
    print(f"已准备攻击动作数量: {len(attack_records)}")

    # 2. 准备防守动作记录（来自最终版 defense_library）
    defense_records = prepare_defense_records()
    print(f"已准备防守动作数量: {len(defense_records)}")

    # 3. 构造长表矩阵（来自最终版 matchup_matrix）
    matchup_long_df = build_matchup_matrix(attack_records, defense_records)
    print(f"已构造攻防长表矩阵，行数: {len(matchup_long_df)}")

    # 4. 宽表矩阵（13 × 22）
    matchup_wide_df = build_wide_matchup_matrix(matchup_long_df)
    print("已构造攻防宽表矩阵")

    # 5. 提取 Top 3 防守动作
    top3_df = extract_top_defenses(matchup_long_df, top_n=3)
    print("已提取每个攻击动作的 Top 3 最优防守动作")

    # 6. 构造论文摘要表（每个攻击动作一行）
    attack_summary_df = build_attack_summary_table(matchup_long_df, attack_records, top_n=3)
    print("已构造攻击动作防守摘要表")

    # 7. 导出文件
    long_csv_path = TABLE_DIR / "attack_defense_matchup_long.csv"
    wide_csv_path = TABLE_DIR / "attack_defense_matchup_wide.csv"
    top3_csv_path = TABLE_DIR / "attack_top3_defenses.csv"
    summary_csv_path = TABLE_DIR / "attack_defense_summary.csv"
    heatmap_path = FIGURE_DIR / "q2_attack_defense_heatmap.png"

    export_csv(matchup_long_df, long_csv_path)
    export_csv(matchup_wide_df, wide_csv_path)
    export_csv(top3_df, top3_csv_path)
    export_csv(attack_summary_df, summary_csv_path)
    plot_matchup_heatmap(matchup_wide_df, heatmap_path)

    # 8. 终端摘要
    print_q2_summary(top3_df, attack_limit=5)

    print("文件已导出：")
    print(f"- {long_csv_path}")
    print(f"- {wide_csv_path}")
    print(f"- {top3_csv_path}")
    print(f"- {summary_csv_path}")
    print(f"- {heatmap_path}")
    print("\nrun_q2.py 执行完成。")


if __name__ == "__main__":
    main()
