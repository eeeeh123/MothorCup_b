from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Iterable, Any, Optional


@dataclass
class AttackScoreWeights:
    """
    Q1 综合评分权重配置（最终表对齐版）。

    设计口径：
    1. 以最终表中的连续变量为主，1-5 映射字段为辅；
    2. 总分 = 攻击收益 + 稳定性收益 + 能耗收益；
    3. 稳定性收益 = 100 - 稳定性惩罚，能耗收益 = 100 - 能耗惩罚。
    """

    # 三大项总权重（建议和为 1）
    attack_gain_weight: float = 0.60
    stability_weight: float = 0.25
    energy_efficiency_weight: float = 0.15

    # 攻击收益内部权重（建议和为 1）
    force_w: float = 0.22
    kinetic_energy_w: float = 0.18
    momentum_w: float = 0.10
    break_w: float = 0.18
    combo_w: float = 0.10
    accuracy_w: float = 0.10
    pressure_w: float = 0.07
    reach_w: float = 0.05

    # 稳定性惩罚内部权重（建议和为 1）
    pfall_w: float = 0.30
    vuln_w: float = 0.20
    trec_w: float = 0.14
    support_loss_w: float = 0.14
    rotation_risk_w: float = 0.10
    recovery_burden_w: float = 0.07
    exposure_w: float = 0.05

    # 能耗惩罚内部权重（建议和为 1）
    cstam_w: float = 0.70
    time_cost_w: float = 0.30


DEFAULT_WEIGHTS = AttackScoreWeights()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _min_max(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    return min(values), max(values)


def _normalize(value: float, min_v: float, max_v: float) -> float:
    """
    Min-Max 标准化到 0~100。
    若某字段所有样本都相同，则返回 50 作为中性值，避免除零。
    """
    if abs(max_v - min_v) < 1e-12:
        return 50.0
    return (value - min_v) / (max_v - min_v) * 100.0


def _build_ranges(records: List[Dict[str, Any]], fields: List[str]) -> Dict[str, tuple[float, float]]:
    ranges = {}
    for field in fields:
        vals = [_to_float(r.get(field, 0.0)) for r in records]
        ranges[field] = _min_max(vals)
    return ranges


def _build_prior_map(action_priors: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    if action_priors is None:
        return {}
    prior_map: Dict[str, Dict[str, Any]] = {}
    for item in action_priors:
        code = item.get("code", "")
        if code:
            prior_map[code] = dict(item)
    return prior_map


def _merge_dynamics_with_priors(
    dynamics_results: Iterable[Dict[str, Any]],
    action_priors: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    将 attack_dynamics.py 的结果与 action_library.py 的原始先验按 code 合并。
    当前版优先保留 dynamics 里的派生量，同时补齐动作库中的连续量与 1-5 映射量。
    """
    prior_map = _build_prior_map(action_priors)
    merged_records: List[Dict[str, Any]] = []

    for dyn in dynamics_results:
        code = dyn.get("code", "")
        merged = dict(dyn)

        if code in prior_map:
            for k, v in prior_map[code].items():
                if k not in merged:
                    merged[k] = v

        merged_records.append(merged)

    return merged_records


def compute_attack_gain_score(
    record: Dict[str, Any],
    ranges: Dict[str, tuple[float, float]],
    weights: AttackScoreWeights = DEFAULT_WEIGHTS,
) -> float:
    """
    计算攻击收益分：数值越高越好。
    以最终表中的连续变量为主，兼容 1-5 战术映射分。
    """
    score = 0.0

    score += weights.force_w * _normalize(
        _to_float(record.get("F", record.get("avg_impact_force", 0.0))),
        *ranges["F"],
    )
    score += weights.kinetic_energy_w * _normalize(
        _to_float(record.get("Ek", record.get("kinetic_energy", 0.0))),
        *ranges["Ek"],
    )
    score += weights.momentum_w * _normalize(
        _to_float(record.get("momentum", 0.0)),
        *ranges["momentum"],
    )
    score += weights.break_w * _normalize(
        _to_float(record.get("Kbreak", record.get("balance_break_level", 0.0))),
        *ranges["Kbreak"],
    )
    score += weights.combo_w * _normalize(
        _to_float(record.get("Ccombo", record.get("combo_potential", 0.0))),
        *ranges["Ccombo"],
    )
    score += weights.accuracy_w * _normalize(
        _to_float(record.get("accuracy_level", 0.0)),
        *ranges["accuracy_level"],
    )
    score += weights.pressure_w * _normalize(
        _to_float(record.get("pressure_potential", 0.0)),
        *ranges["pressure_potential"],
    )
    score += weights.reach_w * _normalize(
        _to_float(record.get("Lreach", 0.0)),
        *ranges["Lreach"],
    )

    return score


def compute_stability_penalty(
    record: Dict[str, Any],
    ranges: Dict[str, tuple[float, float]],
    weights: AttackScoreWeights = DEFAULT_WEIGHTS,
) -> float:
    """
    计算稳定性惩罚：数值越高越差。
    以最终表中的失稳率/暴露度/硬直为主，并兼容 dynamics 代理量。
    """
    penalty = 0.0

    penalty += weights.pfall_w * _normalize(
        _to_float(record.get("Pfall", record.get("p_fall", 0.0))),
        *ranges["Pfall"],
    )
    penalty += weights.vuln_w * _normalize(
        _to_float(record.get("Vvul", record.get("counter_risk", 0.0))),
        *ranges["Vvul"],
    )
    penalty += weights.trec_w * _normalize(
        _to_float(record.get("Trec", record.get("recovery_time", 0.0))),
        *ranges["Trec"],
    )
    penalty += weights.support_loss_w * _normalize(
        _to_float(record.get("support_loss_proxy", 0.0)),
        *ranges["support_loss_proxy"],
    )
    penalty += weights.rotation_risk_w * _normalize(
        _to_float(record.get("rotation_risk_proxy", 0.0)),
        *ranges["rotation_risk_proxy"],
    )
    penalty += weights.recovery_burden_w * _normalize(
        _to_float(record.get("recovery_burden_proxy", 0.0)),
        *ranges["recovery_burden_proxy"],
    )
    penalty += weights.exposure_w * _normalize(
        _to_float(record.get("exposure_proxy", 0.0)),
        *ranges["exposure_proxy"],
    )

    return penalty


def compute_energy_penalty(
    record: Dict[str, Any],
    ranges: Dict[str, tuple[float, float]],
    weights: AttackScoreWeights = DEFAULT_WEIGHTS,
) -> float:
    """
    计算能耗惩罚：数值越高越差。
    当前版同时考虑最终表中的 Cstam 与 time_cost。
    """
    cstam_penalty = _normalize(
        _to_float(record.get("Cstam", record.get("stamina_cost_raw", record.get("energy_cost", 0.0)))),
        *ranges["Cstam"],
    )
    time_penalty = _normalize(
        _to_float(record.get("time_cost", 0.0)),
        *ranges["time_cost"],
    )
    return weights.cstam_w * cstam_penalty + weights.time_cost_w * time_penalty


def compute_total_score(
    record: Dict[str, Any],
    ranges: Dict[str, tuple[float, float]],
    weights: AttackScoreWeights = DEFAULT_WEIGHTS,
) -> Dict[str, float]:
    attack_gain_score = compute_attack_gain_score(record, ranges, weights)
    stability_penalty_score = compute_stability_penalty(record, ranges, weights)
    energy_penalty_score = compute_energy_penalty(record, ranges, weights)

    stability_benefit_score = 100.0 - stability_penalty_score
    energy_efficiency_score = 100.0 - energy_penalty_score

    total_score = (
        weights.attack_gain_weight * attack_gain_score
        + weights.stability_weight * stability_benefit_score
        + weights.energy_efficiency_weight * energy_efficiency_score
    )

    return {
        "attack_gain_score": attack_gain_score,
        "stability_penalty_score": stability_penalty_score,
        "energy_penalty_score": energy_penalty_score,
        "stability_benefit_score": stability_benefit_score,
        "energy_efficiency_score": energy_efficiency_score,
        "total_score": total_score,
    }


def score_actions(
    dynamics_results: Iterable[Dict[str, Any]],
    action_priors: Optional[Iterable[Dict[str, Any]]] = None,
    weights: AttackScoreWeights = DEFAULT_WEIGHTS,
) -> List[Dict[str, Any]]:
    """
    对全部动作进行综合评分并返回排序结果。

    参数：
    - dynamics_results: 来自 attack_dynamics.py 的输出
    - action_priors: 可选，来自 action_library.py 的动作原始先验
    """
    merged_records = _merge_dynamics_with_priors(dynamics_results, action_priors)

    score_fields = [
        "F",
        "Ek",
        "momentum",
        "Kbreak",
        "Ccombo",
        "accuracy_level",
        "pressure_potential",
        "Lreach",
        "Pfall",
        "Vvul",
        "Trec",
        "support_loss_proxy",
        "rotation_risk_proxy",
        "recovery_burden_proxy",
        "exposure_proxy",
        "Cstam",
        "time_cost",
    ]

    ranges = _build_ranges(merged_records, score_fields)

    scored_results: List[Dict[str, Any]] = []
    for record in merged_records:
        score_info = compute_total_score(record, ranges, weights)
        merged = dict(record)
        merged.update(score_info)
        scored_results.append(merged)

    scored_results.sort(key=lambda x: x["total_score"], reverse=True)

    for idx, item in enumerate(scored_results, start=1):
        item["rank"] = idx

    return scored_results


def get_top_actions(scored_results: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    return scored_results[:top_n]


if __name__ == "__main__":
    try:
        from config.action_library import get_action_list
        from models.attack_dynamics import batch_calculate_dynamics

        actions = get_action_list()
        dynamics_results = batch_calculate_dynamics(actions)
        scored = score_actions(dynamics_results, action_priors=actions)

        print("attack_scoring.py 自测开始")
        print(f"动作数量: {len(scored)}")
        print("Top 5 动作：")
        for item in scored[:5]:
            print(
                f"Rank {item['rank']}: {item['name']}"
                f" | total={item['total_score']:.2f}"
                f" | attack={item['attack_gain_score']:.2f}"
                f" | stability_penalty={item['stability_penalty_score']:.2f}"
                f" | energy_penalty={item['energy_penalty_score']:.2f}"
            )
        print("自测完成")
    except Exception as e:
        print("自测失败：", e)
