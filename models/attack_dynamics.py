from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Any


@dataclass
class RobotPhysicalConfig:
    """
    机器人简化物理参数配置。

    注意：
    1. 当前版本中，这些参数主要作为“缺失值回退计算”的兜底配置；
    2. 若动作字典中已经提供 Ek / F / Tpre / Trec / Cstam / Pfall / Vvul 等最终表字段，
       则优先直接使用表中数据，而不是再次用旧公式覆盖。
    """
    total_mass: float = 42.0          # kg，PM01 含电池约 42kg
    torso_mass_ratio: float = 0.50    # 躯干质量占比（兜底）
    arm_mass_ratio: float = 0.05      # 单臂质量占比（兜底）
    leg_mass_ratio: float = 0.20      # 单腿质量占比（兜底）

    arm_length: float = 0.50          # m，兜底等效长度
    leg_length: float = 0.70          # m，兜底等效长度
    torso_buffer_length: float = 0.30 # m，冲撞等躯干动作缓冲尺度


DEFAULT_CONFIG = RobotPhysicalConfig()


@dataclass
class ActionDynamicsResult:
    """
    单个动作的动力学与稳定性代理结果。

    兼容旧版字段，同时补入最终表中的关键连续变量。
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

    # 稳定性代理量（供 attack_scoring.py 使用）
    support_loss_proxy: float
    rotation_risk_proxy: float
    recovery_burden_proxy: float
    exposure_proxy: float

    # 保留原始输入 / 兼容旧版评分器
    eta: float
    omega: float
    dt: float
    balance_risk: float
    time_cost: float
    energy_cost: float
    counter_risk: float

    # 新版底层连续变量（来源于最终表）
    Ek: float
    F: float
    Tpre: float
    Lreach: float
    theta_sweep: float
    Kbreak: float
    Ccombo: float
    Vvul: float
    Pfall: float
    Trec: float
    Cstam: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # 下游兼容别名
        data.update({
            "support_loss": self.support_loss_proxy,
            "rotation_risk": self.rotation_risk_proxy,
            "recovery_burden": self.recovery_burden_proxy,
            "exposure": self.exposure_proxy,
            "recovery_time": self.Trec,
            "stamina_cost_raw": self.Cstam,
            "p_fall": self.Pfall,
        })
        return data


# -----------------------------
# 基础工具函数
# -----------------------------
def _safe_get(action: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = action.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def _map_linear(x: float, src_low: float, src_high: float, dst_low: float = 1.0, dst_high: float = 5.0) -> float:
    """
    线性映射到指定区间，主要用于兜底生成兼容旧版的 1~5 字段。
    """
    if src_high - src_low <= 1e-12:
        return (dst_low + dst_high) / 2.0
    ratio = (x - src_low) / (src_high - src_low)
    ratio = _clamp(ratio, 0.0, 1.0)
    return dst_low + ratio * (dst_high - dst_low)


# -----------------------------
# 机器人分段质量与长度
# -----------------------------
def _get_part_mass_and_length(part: str, cfg: RobotPhysicalConfig) -> tuple[float, float]:
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


# -----------------------------
# 主计算函数
# -----------------------------
def calculate_action_dynamics(
    action: Dict[str, Any],
    cfg: RobotPhysicalConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    计算单个动作的动力学代理指标。

    当前版本的优先级规则：
    1. 若动作字典中已经提供最终表连续变量（Ek/F/Tpre/Trec/Cstam/Pfall/Vvul 等），优先使用；
    2. 仅在缺失时才使用旧的 eta/omega/dt + 简化刚体公式做回退推导；
    3. 输出继续兼容旧版 attack_scoring.py 的字段要求。
    """
    code = action.get("code", "")
    name = action.get("name", "")
    category = action.get("category", "")
    part = action.get("part", "")

    # 旧版基础输入
    eta = _safe_get(action, "eta", 0.0)
    omega = _safe_get(action, "omega", 0.0)
    dt = _safe_get(action, "dt", 0.1)

    # 新版最终表连续量（若已存在则优先使用）
    Ek = _safe_get(action, "Ek", action.get("kinetic_energy", 0.0))
    F = _safe_get(action, "F", action.get("avg_impact_force", 0.0))
    Tpre = _safe_get(action, "Tpre", 0.0)
    Lreach = _safe_get(action, "Lreach", 0.0)
    theta_sweep = _safe_get(action, "theta_sweep", action.get("θsweep", 0.0))
    Kbreak = _safe_get(action, "Kbreak", 0.0)
    Ccombo = _safe_get(action, "Ccombo", 0.0)
    Vvul = _safe_get(action, "Vvul", 0.0)
    Pfall = _safe_get(action, "Pfall", action.get("p_fall", 0.0))
    Trec = _safe_get(action, "Trec", action.get("recovery_time", 0.0))
    Cstam = _safe_get(action, "Cstam", action.get("stamina_cost_raw", 0.0))

    if dt <= 0:
        raise ValueError(f"{name or code} 的 dt 必须大于 0，当前 dt={dt}")
    if omega < 0:
        raise ValueError(f"{name or code} 的 omega 不应为负，当前 omega={omega}")

    torso_mass = cfg.total_mass * cfg.torso_mass_ratio
    limb_mass, limb_length = _get_part_mass_and_length(part, cfg)

    # 一、几何/动力学兜底量
    terminal_speed = omega * limb_length
    equivalent_mass = limb_mass + eta * torso_mass

    # 二、动量 / 动能 / 平均冲击力：优先用最终表
    # momentum 若表中没有，优先由 F*dt 回推；再不行用等效质量 * 末端速度。
    if F > 0:
        momentum = F * dt
    else:
        momentum = equivalent_mass * terminal_speed

    kinetic_energy = Ek if Ek > 0 else 0.5 * limb_mass * (terminal_speed ** 2)
    avg_impact_force = F if F > 0 else momentum / dt

    # 三、兼容旧版评分器的代价字段
    # 若 action_library 中已给出映射后的 1~5 字段，则直接采用；
    # 否则从连续变量做线性兜底映射。
    balance_risk = _safe_get(action, "balance_risk", _map_linear(Pfall, 0.0, 0.50, 1.0, 5.0))

    # time_cost 旧版是 1~5 代价项；若缺失，则用 Tpre + 0.5*Trec 派生。
    raw_time_cost = Tpre + 0.5 * Trec
    time_cost = _safe_get(action, "time_cost", _map_linear(raw_time_cost, 0.15, 1.40, 1.0, 5.0))

    # energy_cost 旧版是 1~5；若缺失，则用 Cstam 做兜底映射。
    energy_cost = _safe_get(action, "energy_cost", _map_linear(Cstam, 100.0, 800.0, 1.0, 5.0))

    # counter_risk 旧版是 1~5；若缺失，则用 Vvul 做兜底映射。
    counter_risk = _safe_get(action, "counter_risk", _map_linear(Vvul, 0.0, 1.0, 1.0, 5.0))

    # 四、稳定性代理量（由最终表连续变量驱动）
    # 1) 支撑损失代理：以 Pfall 为主，腿法通常更高
    part_support_factor = {
        "arm": 0.8,
        "leg": 1.2,
        "torso": 1.0,
    }.get(part, 1.0)
    support_loss_proxy = Pfall * part_support_factor

    # 2) 转体失稳代理：由角速度 + 覆盖角 + 失稳率共同驱动
    rotation_risk_proxy = Pfall * (0.7 * omega + 0.3 * theta_sweep * 10.0)

    # 3) 恢复负担代理：由前摇、硬直、能耗组成
    recovery_burden_proxy = 0.25 * Tpre * 10.0 + 0.45 * Trec + 0.30 * (Cstam / 100.0)

    # 4) 暴露风险代理：由暴露度 + 失稳率共同驱动
    exposure_proxy = 0.6 * Vvul + 0.4 * Pfall

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
        Ek=kinetic_energy,
        F=avg_impact_force,
        Tpre=Tpre,
        Lreach=Lreach,
        theta_sweep=theta_sweep,
        Kbreak=Kbreak,
        Ccombo=Ccombo,
        Vvul=Vvul,
        Pfall=Pfall,
        Trec=Trec,
        Cstam=Cstam,
    )

    return result.to_dict()


def batch_calculate_dynamics(
    actions: Iterable[Dict[str, Any]],
    cfg: RobotPhysicalConfig = DEFAULT_CONFIG,
) -> List[Dict[str, Any]]:
    """批量计算多个动作的动力学结果。"""
    return [calculate_action_dynamics(action, cfg=cfg) for action in actions]


if __name__ == "__main__":
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
