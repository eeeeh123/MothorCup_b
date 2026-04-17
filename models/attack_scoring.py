from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Any, Optional


@dataclass
class AttackScoreWeights:
    """
    Q1 综合评分权重配置。

    设计思想：
    1. 总分 = 攻击收益 + 稳定性收益 + 能耗收益
    2. 其中稳定性收益 = 100 - 稳定性惩罚
    3. 能耗收益 = 100 - 能耗惩罚

    这样最终 total_score 更直观，范围通常在 0~100 附近。
    """

    # 三大项总权重（建议和为 1）
    attack_gain_weight: float = 0.60
    stability_weight: float = 0.25
    energy_efficiency_weight: float = 0.15

    # 攻击收益内部权重（建议和为 1）
    kinetic_energy_w: float = 0.28
    avg_impact_force_w: float = 0.28
    momentum_w: float = 0.16
    impact_level_w: float = 0.10
    accuracy_level_w: float = 0.07
    balance_break_level_w: float = 0.06
    combo_potential_w: float = 0.03
    pressure_potential_w: float = 0.02

    # 稳定性惩罚内部权重（建议和为 1）
    support_loss_w: float = 0.32
    rotation_risk_w: float = 0.28
    recovery_burden_w: float = 0.22
    exposure_w: float = 0.18


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
    """
    将 action_library 的原始动作先验转成 code -> dict 的映射。
    """
    if action_priors is None:
        return {}
    prior_map = {}
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
    优点：
    - dynamics 负责物理代理量
    - priors 提供 impact_level / accuracy_level / combo_potential 等战术先验
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


def compute_impact_score(
    record: Dict[str, Any],
    ranges: Dict[str, tuple[float, float]],
    weights: AttackScoreWeights = DEFAULT_WEIGHTS,
) -> float:
    """
    计算攻击收益分：数值越高越好。
    同时融合动力学指标和动作先验指标。
    """
    score = 0.0

    score += weights.kinetic_energy_w * _normalize(
        _to_float(record.get("kinetic_energy", 0.0)), *ranges["kinetic_energy"]
    )
    score += weights.avg_impact_force_w * _normalize(
        _to_float(record.get("avg_impact_force", 0.0)), *ranges["avg_impact_force"]
    )
    score += weights.momentum_w * _normalize(
        _to_float(record.get("momentum", 0.0)), *ranges["momentum"]
    )
    score += weights.impact_level_w * _normalize(
        _to_float(record.get("impact_level", 0.0)), *ranges["impact_level"]
    )
    score += weights.accuracy_level_w * _normalize(
        _to_float(record.get("accuracy_level", 0.0)), *ranges["accuracy_level"]
    )
    score += weights.balance_break_level_w * _normalize(
        _to_float(record.get("balance_break_level", 0.0)), *ranges["balance_break_level"]
    )
    score += weights.combo_potential_w * _normalize(
        _to_float(record.get("combo_potential", 0.0)), *ranges["combo_potential"]
    )
    score += weights.pressure_potential_w * _normalize(
        _to_float(record.get("pressure_potential", 0.0)), *ranges["pressure_potential"]
    )

    return score


def compute_stability_penalty(
    record: Dict[str, Any],
    ranges: Dict[str, tuple[float, float]],
    weights: AttackScoreWeights = DEFAULT_WEIGHTS,
) -> float:
    """
    计算稳定性惩罚：数值越高越差。
    """
    penalty = 0.0

    penalty += weights.support_loss_w * _normalize(
        _to_float(record.get("support_loss_proxy", 0.0)), *ranges["support_loss_proxy"]
    )
    penalty += weights.rotation_risk_w * _normalize(
        _to_float(record.get("rotation_risk_proxy", 0.0)), *ranges["rotation_risk_proxy"]
    )
    penalty += weights.recovery_burden_w * _normalize(
        _to_float(record.get("recovery_burden_proxy", 0.0)), *ranges["recovery_burden_proxy"]
    )
    penalty += weights.exposure_w * _normalize(
        _to_float(record.get("exposure_proxy", 0.0)), *ranges["exposure_proxy"]
    )

    return penalty


def compute_energy_penalty(
    record: Dict[str, Any],
    ranges: Dict[str, tuple[float, float]],
) -> float:
    """
    计算能耗惩罚：数值越高越差。
    这里先仅使用 energy_cost，后续也可以扩展加入 time_cost。
    """
    return _normalize(
        _to_float(record.get("energy_cost", 0.0)),
        *ranges["energy_cost"]
    )


def compute_total_score(
    record: Dict[str, Any],
    ranges: Dict[str, tuple[float, float]],
    weights: AttackScoreWeights = DEFAULT_WEIGHTS,
) -> Dict[str, float]:
    """
    计算单个动作的三项分数与总分。
    """
    attack_gain_score = compute_impact_score(record, ranges, weights)
    stability_penalty_score = compute_stability_penalty(record, ranges, weights)
    energy_penalty_score = compute_energy_penalty(record, ranges)

    # 惩罚转收益，越稳定/越省能越接近 100
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
                     建议传入，这样可以把 impact_level 等一并纳入评分
    """
    merged_records = _merge_dynamics_with_priors(dynamics_results, action_priors)

    # 用于评分的字段
    score_fields = [
        "kinetic_energy",
        "avg_impact_force",
        "momentum",
        "impact_level",
        "accuracy_level",
        "balance_break_level",
        "combo_potential",
        "pressure_potential",
        "support_loss_proxy",
        "rotation_risk_proxy",
        "recovery_burden_proxy",
        "exposure_proxy",
        "energy_cost",
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
    # 建议在项目根目录使用模块方式运行：
    # python -m models.attack_scoring
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