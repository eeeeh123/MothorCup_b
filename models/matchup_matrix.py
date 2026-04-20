from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
from config.action_library import iter_attack_actions
from config.defense_library import iter_defense_actions

def prepare_attack_records() -> List[Dict]:
    return [dict(item) for item in iter_attack_actions()]

def prepare_defense_records() -> List[Dict]:
    return [dict(item) for item in iter_defense_actions()]

def _range_fit_score(attack_range: str, effective_ranges: List[str]) -> Tuple[float, str]:
    if attack_range in effective_ranges:
        return 20.0, "范围直接匹配"
    soft_pairs = {("near-mid", "near"), ("near-mid", "mid"), ("mid-far", "mid"), ("mid-far", "far"), ("near", "clinch"), ("clinch", "near")}
    for er in effective_ranges:
        if (attack_range, er) in soft_pairs:
            return 12.0, "范围近似匹配"
    return 2.0, "范围不占优"

def _distance_bias_interaction(attack_bias: float, defense_bias: float) -> Tuple[float, str]:
    if attack_bias <= -0.5 and defense_bias <= -0.5:
        return 8.0, "拉开型应对良好"
    if attack_bias >= 0.5 and defense_bias >= 0.0:
        return 7.0, "压近型应对良好"
    if attack_bias >= 0.5 and defense_bias <= -0.8:
        return 4.0, "后撤避近有一定效果"
    if attack_bias <= -0.5 and defense_bias >= 0.5:
        return 1.0, "距离倾向相冲突"
    return 4.0, "距离倾向中性"

def _special_adjustments(a: Dict, d: Dict) -> Tuple[float, List[str]]:
    bonus = 0.0
    reasons: List[str] = []
    name_a = a["name"]
    code_d = d["code"]
    if name_a in {"前踢", "侧踢", "回旋踢"} and code_d in {"D08", "D09", "D10", "D18", "D22"}:
        bonus += 8.0; reasons.append("腿法对应撤防/绕防")
    if name_a == "低扫腿" and code_d in {"D12", "D14", "D15"}:
        bonus += 9.0; reasons.append("下盘攻击对应重心稳定防守")
    if name_a in {"直拳", "勾拳", "组合拳", "摆拳"} and code_d in {"D01", "D03", "D04", "D20"}:
        bonus += 7.0; reasons.append("近身拳法对应上肢拦截")
    if name_a in {"膝撞", "冲撞"} and code_d in {"D12", "D14", "D16", "D19"}:
        bonus += 9.0; reasons.append("近压类动作对应稳姿/缠斗防守")
    if a["accuracy_level"] >= 4.0 and d["timing_difficulty"] >= 3.4:
        bonus -= 3.0; reasons.append("高时机难度对高精度攻击不利")
    return bonus, reasons

def compute_matchup_score(attack: Dict, defense: Dict) -> Tuple[float, str]:
    score = 45.0
    reasons: List[str] = []
    s1, r1 = _range_fit_score(attack["attack_range"], defense["effective_ranges"])
    score += s1; reasons.append(r1)
    s2, r2 = _distance_bias_interaction(float(attack.get("distance_bias", 0.0)), float(defense.get("distance_bias", 0.0)))
    score += s2; reasons.append(r2)
    score += 4.0 * float(defense["direct_defense_level"])
    score += 2.5 * float(defense["stability_recovery"])
    score += 1.8 * float(defense["counter_attack_potential"])
    score -= 2.0 * float(defense["timing_difficulty"])
    score -= 1.5 * float(defense["risk_if_failed"])
    score -= 1.0 * float(defense["energy_cost"])
    score -= 0.8 * float(defense["mobility_loss"])
    score += 5.0 * float(attack.get("Vvul", attack.get("counter_risk", 0.0)))
    score += 4.0 * float(attack.get("Pfall", attack.get("balance_risk", 0.0)))
    bonus, rs = _special_adjustments(attack, defense)
    score += bonus; reasons.extend(rs)
    score = max(0.0, min(100.0, score))
    reason = "；".join(reasons[:3]) if reasons else "综合适配"
    return score, reason

def build_matchup_matrix(attack_records: List[Dict] | None = None, defense_records: List[Dict] | None = None) -> pd.DataFrame:
    attack_records = attack_records or prepare_attack_records()
    defense_records = defense_records or prepare_defense_records()
    rows: List[Dict] = []
    for a in attack_records:
        for d in defense_records:
            score, reason = compute_matchup_score(a, d)
            rows.append({"attack_code": a["code"], "attack_name": a["name"], "attack_range": a["attack_range"], "defense_code": d["code"], "defense_name": d["name"], "matchup_score": round(score, 3), "reason": reason})
    df = pd.DataFrame(rows)
    return df.sort_values(["attack_code", "matchup_score"], ascending=[True, False]).reset_index(drop=True)

def build_wide_matchup_matrix(matchup_df: pd.DataFrame) -> pd.DataFrame:
    return matchup_df.pivot(index="attack_name", columns="defense_name", values="matchup_score").sort_index(axis=0).sort_index(axis=1)

def extract_top_defenses(matchup_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    rows: List[Dict] = []
    for attack_name, sub in matchup_df.groupby("attack_name"):
        sub_sorted = sub.sort_values("matchup_score", ascending=False).head(top_n)
        for rank, (_, row) in enumerate(sub_sorted.iterrows(), start=1):
            rows.append({"attack_name": attack_name, "rank": rank, "defense_code": row["defense_code"], "defense_name": row["defense_name"], "matchup_score": row["matchup_score"], "reason": row["reason"]})
    return pd.DataFrame(rows)
