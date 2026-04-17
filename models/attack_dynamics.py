from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Any


@dataclass
class RobotPhysicalConfig:
    """
    机器人简化物理参数配置。
    这些参数是第一版代理建模参数，后续可依据资料进一步校准。
    """
    total_mass: float = 42.0          # kg，题面给出 PM01 含电池约 42kg
    torso_mass_ratio: float = 0.50    # 躯干质量占比
    arm_mass_ratio: float = 0.05      # 单臂质量占比
    leg_mass_ratio: float = 0.20      # 单腿质量占比

    arm_length: float = 0.50          # m，手臂等效长度
    leg_length: float = 0.70          # m，腿部等效长度
    torso_buffer_length: float = 0.30 # m，冲撞等躯干动作的等效位移/缓冲尺度


DEFAULT_CONFIG = RobotPhysicalConfig()


@dataclass
class ActionDynamicsResult:
    """
    单个动作的动力学与稳定性代理结果。
    """
    code: str
    name: str
    category: str
    part: str

    # 基础动力学量
    limb_mass: float
    limb_length: float
    equivalent_mass: float
    terminal_speed: float
    momentum: float
    kinetic_energy: float
    avg_impact_force: float

    # 稳定性代理量（非最终评分，只是供后续 attack_scoring.py 使用）
    support_loss_proxy: float
    rotation_risk_proxy: float
    recovery_burden_proxy: float
    exposure_proxy: float

    # 保留原始先验，便于追踪
    eta: float
    omega: float
    dt: float
    balance_risk: float
    time_cost: float
    energy_cost: float
    counter_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_part_mass_and_length(part: str, cfg: RobotPhysicalConfig) -> tuple[float, float]:
    """
    根据发力部位返回等效肢体质量和等效长度。
    """
    torso_mass = cfg.total_mass * cfg.torso_mass_ratio
    arm_mass = cfg.total_mass * cfg.arm_mass_ratio
    leg_mass = cfg.total_mass * cfg.leg_mass_ratio

    if part == "arm":
        return arm_mass, cfg.arm_length
    if part == "leg":
        return leg_mass, cfg.leg_length
    if part == "torso":
        return torso_mass, cfg.torso_buffer_length

    raise ValueError(f"未知 part 类型: {part}")


def _safe_get(action: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    兼容不同阶段动作字典中字段可能缺失的情况。
    """
    value = action.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def calculate_action_dynamics(
    action: Dict[str, Any],
    cfg: RobotPhysicalConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    计算单个动作的动力学代理指标。

    输入 action 推荐至少包含：
    - code, name, category, part
    - eta, omega, dt
    - balance_risk, time_cost, energy_cost, counter_risk

    返回的是“中间层结果”，不做最终排名。
    """
    code = action.get("code", "")
    name = action.get("name", "")
    category = action.get("category", "")
    part = action.get("part", "")

    eta = _safe_get(action, "eta", 0.0)
    omega = _safe_get(action, "omega", 0.0)
    dt = _safe_get(action, "dt", 0.1)

    balance_risk = _safe_get(action, "balance_risk", 0.0)
    time_cost = _safe_get(action, "time_cost", 0.0)
    energy_cost = _safe_get(action, "energy_cost", 0.0)
    counter_risk = _safe_get(action, "counter_risk", 0.0)

    if dt <= 0:
        raise ValueError(f"{name or code} 的 dt 必须大于 0，当前 dt={dt}")
    if omega < 0:
        raise ValueError(f"{name or code} 的 omega 不应为负，当前 omega={omega}")

    torso_mass = cfg.total_mass * cfg.torso_mass_ratio
    limb_mass, limb_length = _get_part_mass_and_length(part, cfg)

    # -----------------------------
    # 一、保留原 Q1.py 中可用的动力学思路
    # -----------------------------
    # 1) 末端线速度（简化版）
    terminal_speed = omega * limb_length

    # 2) 等效质量：肢体质量 + 躯干传导质量
    equivalent_mass = limb_mass + eta * torso_mass

    # 3) 动量
    momentum = equivalent_mass * terminal_speed

    # 4) 动能（沿用原脚本的快速代理口径）
    kinetic_energy = 0.5 * limb_mass * (terminal_speed ** 2)

    # 5) 平均冲击力 / 平均冲量率（不是峰值力）
    avg_impact_force = momentum / dt

    # -----------------------------
    # 二、为 B 题问题 1 增补稳定性代理量
    # -----------------------------
    # 题目要求：不仅追求冲击效果，还要兼顾自身平衡
    # 这里先给出“原始代理量”，后续统一由 attack_scoring.py 做标准化和加权。
    #
    # support_loss_proxy：支撑损失代理
    # - 腿法和转体动作通常更依赖单腿/动态支撑
    # - 这里先把 balance_risk 本身作为核心输入，再叠加 part 特征
    part_support_factor = {
        "arm": 0.8,
        "leg": 1.2,
        "torso": 1.0,
    }.get(part, 1.0)

    support_loss_proxy = balance_risk * part_support_factor

    # rotation_risk_proxy：转体失稳代理
    # - omega 越大，balance_risk 越高，动作后姿态恢复难度通常越大
    rotation_risk_proxy = balance_risk * omega

    # recovery_burden_proxy：动作后恢复负担
    # - 与动作耗时、能耗、碰撞作用时间共同相关
    recovery_burden_proxy = 0.45 * time_cost + 0.35 * energy_cost + 0.20 * (dt * 10.0)

    # exposure_proxy：出招后的暴露风险
    # - 结合被反制风险和自身平衡风险
    exposure_proxy = 0.6 * counter_risk + 0.4 * balance_risk

    result = ActionDynamicsResult(
        code=code,
        name=name,
        category=category,
        part=part,
        limb_mass=limb_mass,
        limb_length=limb_length,
        equivalent_mass=equivalent_mass,
        terminal_speed=terminal_speed,
        momentum=momentum,
        kinetic_energy=kinetic_energy,
        avg_impact_force=avg_impact_force,
        support_loss_proxy=support_loss_proxy,
        rotation_risk_proxy=rotation_risk_proxy,
        recovery_burden_proxy=recovery_burden_proxy,
        exposure_proxy=exposure_proxy,
        eta=eta,
        omega=omega,
        dt=dt,
        balance_risk=balance_risk,
        time_cost=time_cost,
        energy_cost=energy_cost,
        counter_risk=counter_risk,
    )

    return result.to_dict()


def batch_calculate_dynamics(
    actions: Iterable[Dict[str, Any]],
    cfg: RobotPhysicalConfig = DEFAULT_CONFIG,
) -> List[Dict[str, Any]]:
    """
    批量计算多个动作的动力学结果。
    """
    return [calculate_action_dynamics(action, cfg=cfg) for action in actions]


if __name__ == "__main__":
    # 本地最小自测
    try:
        from config.action_library import get_action_list

        actions = get_action_list()
        results = batch_calculate_dynamics(actions)

        print("attack_dynamics.py 自测开始")
        print(f"动作数量: {len(results)}")
        if results:
            first = results[0]
            print("第一个动作:", first["name"])
            print("末端速度:", round(first["terminal_speed"], 4))
            print("动能:", round(first["kinetic_energy"], 4))
            print("平均冲击力:", round(first["avg_impact_force"], 4))
            print("支撑损失代理:", round(first["support_loss_proxy"], 4))
        print("自测完成")
    except Exception as e:
        print("自测失败：", e)