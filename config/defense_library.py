from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class DefenseAction:
    """
    单个防守动作的结构化定义。

    当前版本已与《Q2_防守动作采集》最终表对齐：
    - 原始字段全部来自最终表；
    - stability_recovery / mobility_loss 为代码侧派生字段，
      以兼容现有 transition.py 的接口；
    - energy_cost 保持原始单位 J，不做 1-5 档离散化。
    """
    code: str
    name: str
    category: str
    category_cn: str
    mechanism_keywords: str
    applicable_attack_types: List[str]
    main_cost: str

    # 原始数值字段（来自最终表）
    direct_defense_level: float
    counter_attack_potential: float
    timing_difficulty: float
    risk_if_failed: float
    energy_cost: float
    recovery_time: float

    # 派生字段（为兼容旧模型接口）
    stability_recovery: float
    mobility_loss: float

    data_source: str
    confidence: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # 兼容部分旧代码/中文说明
        data.update({
            "防护量级": self.direct_defense_level,
            "反击潜力加成": self.counter_attack_potential,
            "时机难度": self.timing_difficulty,
            "失误惩罚": self.risk_if_failed,
            "能耗": self.energy_cost,
            "恢复时间": self.recovery_time,
        })
        return data


# 主要代价 -> mobility_loss 的代码侧映射
# 说明：最终表没有显式给出 mobility_loss，因此这里用“主要代价”做兼容派生。
_MOBILITY_LOSS_MAP: Dict[str, float] = {'机动性下降': 0.85, '时机要求高': 0.3, '反击衔接一般': 0.25, '对高位攻击无效': 0.2, '近身风险高': 0.5, '需横向空间': 0.8, '易被追击下段': 0.45, '可能失去主动': 0.6, '稳定要求高': 0.45, '时间窗口短': 0.35, '视线受限': 0.2, '移动迟缓': 0.8, '侧向盲区大': 0.5, '耗能极大': 0.45, '让出空间': 0.35, '下盘不稳': 0.55, '直接送分': 0.1, '耗能与破绽极大': 0.65, '无法反击': 0.15, '难度极高': 0.55, '体能消耗顶峰': 0.7, '空间要求高': 0.75}

# 防守类别 -> stability_recovery 的附加校正
# 说明：最终表没有显式给出 stability_recovery，因此这里结合防护量级、失误惩罚、
# 恢复时间、防守类别做兼容派生。
_CATEGORY_STABILITY_BONUS: Dict[str, float] = {'格挡': 0.1, '闪避': 0.0, '姿态': 0.15, '平衡': 0.35, '倒地': 0.05, '复合': 0.12}


def _clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def derive_stability_recovery(
    direct_defense_level: float,
    risk_if_failed: float,
    recovery_time: float,
    category_cn: str,
) -> float:
    value = (
        0.45 * float(direct_defense_level)
        + 0.30 * (1.0 - float(risk_if_failed))
        + 0.15 * (1.0 - float(recovery_time))
        + _CATEGORY_STABILITY_BONUS.get(category_cn, 0.10)
    )
    return round(_clamp(value), 3)


def derive_mobility_loss(main_cost: str) -> float:
    return _MOBILITY_LOSS_MAP.get(main_cost, 0.40)


DEFENSE_ACTIONS: Dict[str, DefenseAction] = {
    "D01": DefenseAction(
        code="D01",
        name="十字格挡",
        category="block",
        category_cn="格挡",
        mechanism_keywords='正面封挡',
        applicable_attack_types=['直拳', '摆拳'],
        main_cost='机动性下降',
        direct_defense_level=0.85,
        counter_attack_potential=0.1,
        timing_difficulty=0.3,
        risk_if_failed=0.4,
        energy_cost=100.0,
        recovery_time=0.4,
        stability_recovery=0.752,
        mobility_loss=0.85,
        data_source='刚体碰撞推演',
        confidence='高',
        notes='牺牲机动性换取高防护，反击潜力低',
    ),
    "D02": DefenseAction(
        code="D02",
        name="单手拍挡",
        category="block",
        category_cn="格挡",
        mechanism_keywords='外拨格挡',
        applicable_attack_types=['直线攻击', '前踢'],
        main_cost='时机要求高',
        direct_defense_level=0.6,
        counter_attack_potential=0.3,
        timing_difficulty=0.6,
        risk_if_failed=0.6,
        energy_cost=80.0,
        recovery_time=0.3,
        stability_recovery=0.595,
        mobility_loss=0.3,
        data_source='刚体碰撞推演',
        confidence='高',
        notes='改变攻击轨迹，能耗低，恢复极快',
    ),
    "D03": DefenseAction(
        code="D03",
        name="肘挡",
        category="block",
        category_cn="格挡",
        mechanism_keywords='护头护胸',
        applicable_attack_types=['勾拳', '摆拳', '撞击'],
        main_cost='反击衔接一般',
        direct_defense_level=0.7,
        counter_attack_potential=0.2,
        timing_difficulty=0.4,
        risk_if_failed=0.5,
        energy_cost=110.0,
        recovery_time=0.4,
        stability_recovery=0.655,
        mobility_loss=0.25,
        data_source='刚体碰撞推演',
        confidence='中',
        notes='骨架硬抗，防护量级高于单手拍挡',
    ),
    "D04": DefenseAction(
        code="D04",
        name="下压格挡",
        category="block",
        category_cn="格挡",
        mechanism_keywords='下段封挡',
        applicable_attack_types=['低扫腿', '低段踢'],
        main_cost='对高位攻击无效',
        direct_defense_level=0.7,
        counter_attack_potential=0.2,
        timing_difficulty=0.5,
        risk_if_failed=0.7,
        energy_cost=120.0,
        recovery_time=0.5,
        stability_recovery=0.58,
        mobility_loss=0.2,
        data_source='刚体碰撞推演',
        confidence='中',
        notes='针对下盘的特化防御，失误容易挨重击',
    ),
    "D05": DefenseAction(
        code="D05",
        name="钳制格挡",
        category="block",
        category_cn="格挡",
        mechanism_keywords='锁臂限制',
        applicable_attack_types=['连续出拳'],
        main_cost='近身风险高',
        direct_defense_level=0.5,
        counter_attack_potential=0.8,
        timing_difficulty=0.85,
        risk_if_failed=0.9,
        energy_cost=300.0,
        recovery_time=0.8,
        stability_recovery=0.385,
        mobility_loss=0.5,
        data_source='博弈常识推演',
        confidence='中',
        notes='极高收益极高风险，成功可直接中断对方连招',
    ),
    "D06": DefenseAction(
        code="D06",
        name="左右侧闪",
        category="evade",
        category_cn="闪避",
        mechanism_keywords='横移规避',
        applicable_attack_types=['正面直击'],
        main_cost='需横向空间',
        direct_defense_level=0.0,
        counter_attack_potential=0.6,
        timing_difficulty=0.8,
        risk_if_failed=0.9,
        energy_cost=200.0,
        recovery_time=0.4,
        stability_recovery=0.12,
        mobility_loss=0.8,
        data_source='运动学推演',
        confidence='高',
        notes='纯闪避动作（无物理接触缓冲），让出身位反击',
    ),
    "D07": DefenseAction(
        code="D07",
        name="下潜闪避",
        category="evade",
        category_cn="闪避",
        mechanism_keywords='降重心躲避',
        applicable_attack_types=['摆拳', '高位攻击'],
        main_cost='易被追击下段',
        direct_defense_level=0.0,
        counter_attack_potential=0.5,
        timing_difficulty=0.75,
        risk_if_failed=0.85,
        energy_cost=250.0,
        recovery_time=0.6,
        stability_recovery=0.105,
        mobility_loss=0.45,
        data_source='运动学推演',
        confidence='高',
        notes='降重心躲避，大腿电机负荷高（耗能250J）',
    ),
    "D08": DefenseAction(
        code="D08",
        name="后跳/撤步",
        category="evade",
        category_cn="闪避",
        mechanism_keywords='拉开距离',
        applicable_attack_types=['近身冲击'],
        main_cost='可能失去主动',
        direct_defense_level=0.0,
        counter_attack_potential=0.2,
        timing_difficulty=0.5,
        risk_if_failed=0.5,
        energy_cost=180.0,
        recovery_time=0.5,
        stability_recovery=0.225,
        mobility_loss=0.6,
        data_source='运动学推演',
        confidence='高',
        notes='拉开距离，安全但丧失反击极佳身位',
    ),
    "D09": DefenseAction(
        code="D09",
        name="转身闪避",
        category="evade",
        category_cn="闪避",
        mechanism_keywords='旋身规避',
        applicable_attack_types=['侧踢', '回旋踢'],
        main_cost='稳定要求高',
        direct_defense_level=0.0,
        counter_attack_potential=0.4,
        timing_difficulty=0.85,
        risk_if_failed=0.8,
        energy_cost=250.0,
        recovery_time=0.7,
        stability_recovery=0.105,
        mobility_loss=0.45,
        data_source='运动学推演',
        confidence='中',
        notes='应对大面积扫掠，旋转电机耗能高',
    ),
    "D10": DefenseAction(
        code="D10",
        name="滑步环绕",
        category="evade",
        category_cn="闪避",
        mechanism_keywords='绕侧/绕后',
        applicable_attack_types=['冲撞', '直线攻击'],
        main_cost='时间窗口短',
        direct_defense_level=0.0,
        counter_attack_potential=0.9,
        timing_difficulty=0.7,
        risk_if_failed=0.7,
        energy_cost=220.0,
        recovery_time=0.6,
        stability_recovery=0.15,
        mobility_loss=0.35,
        data_source='博弈常识推演',
        confidence='中',
        notes='最优战术位，绕后反击潜力极高',
    ),
    "D11": DefenseAction(
        code="D11",
        name="护头防御",
        category="stance",
        category_cn="姿态",
        mechanism_keywords='静态护首',
        applicable_attack_types=['高位打击'],
        main_cost='视线受限',
        direct_defense_level=0.9,
        counter_attack_potential=0.0,
        timing_difficulty=0.1,
        risk_if_failed=0.2,
        energy_cost=50.0,
        recovery_time=0.2,
        stability_recovery=0.915,
        mobility_loss=0.2,
        data_source='刚体碰撞推演',
        confidence='高',
        notes='纯静态防御，容错率最高，无反击加成',
    ),
    "D12": DefenseAction(
        code="D12",
        name="沉身防御",
        category="stance",
        category_cn="姿态",
        mechanism_keywords='降低重心',
        applicable_attack_types=['中高位打击'],
        main_cost='移动迟缓',
        direct_defense_level=0.8,
        counter_attack_potential=0.1,
        timing_difficulty=0.2,
        risk_if_failed=0.3,
        energy_cost=150.0,
        recovery_time=0.4,
        stability_recovery=0.81,
        mobility_loss=0.8,
        data_source='运动学推演',
        confidence='高',
        notes='压缩受击面积，电机保持力矩耗能',
    ),
    "D13": DefenseAction(
        code="D13",
        name="侧身防御",
        category="stance",
        category_cn="姿态",
        mechanism_keywords='减少暴露面',
        applicable_attack_types=['直线刺击'],
        main_cost='侧向盲区大',
        direct_defense_level=0.75,
        counter_attack_potential=0.3,
        timing_difficulty=0.4,
        risk_if_failed=0.4,
        energy_cost=120.0,
        recovery_time=0.3,
        stability_recovery=0.773,
        mobility_loss=0.5,
        data_source='运动学推演',
        confidence='中',
        notes='减少正面暴露，适度反击',
    ),
    "D14": DefenseAction(
        code="D14",
        name="重心补偿",
        category="balance",
        category_cn="平衡",
        mechanism_keywords='动态抗冲击',
        applicable_attack_types=['重击', '冲撞'],
        main_cost='耗能极大',
        direct_defense_level=0.5,
        counter_attack_potential=0.0,
        timing_difficulty=0.9,
        risk_if_failed=0.8,
        energy_cost=300.0,
        recovery_time=0.8,
        stability_recovery=0.665,
        mobility_loss=0.45,
        data_source='控制工程推演',
        confidence='中',
        notes='强行对抗冲击力，极大消耗电机功率',
    ),
    "D15": DefenseAction(
        code="D15",
        name="卸力缓冲",
        category="balance",
        category_cn="平衡",
        mechanism_keywords='顺势后移',
        applicable_attack_types=['各种钝击'],
        main_cost='让出空间',
        direct_defense_level=0.7,
        counter_attack_potential=0.2,
        timing_difficulty=0.6,
        risk_if_failed=0.5,
        energy_cost=100.0,
        recovery_time=0.5,
        stability_recovery=0.89,
        mobility_loss=0.35,
        data_source='刚体碰撞推演',
        confidence='中',
        notes='顺势移动减伤',
    ),
    "D16": DefenseAction(
        code="D16",
        name="步点调整",
        category="balance",
        category_cn="平衡",
        mechanism_keywords='快速碎步',
        applicable_attack_types=['连续轻击'],
        main_cost='下盘不稳',
        direct_defense_level=0.4,
        counter_attack_potential=0.5,
        timing_difficulty=0.7,
        risk_if_failed=0.6,
        energy_cost=200.0,
        recovery_time=0.5,
        stability_recovery=0.725,
        mobility_loss=0.55,
        data_source='控制工程推演',
        confidence='中',
        notes='动态平衡调整',
    ),
    "D17": DefenseAction(
        code="D17",
        name="受控倒地",
        category="ground",
        category_cn="倒地",
        mechanism_keywords='顺应重力',
        applicable_attack_types=['即将被KO'],
        main_cost='直接送分',
        direct_defense_level=0.95,
        counter_attack_potential=0.0,
        timing_difficulty=0.4,
        risk_if_failed=1.0,
        energy_cost=50.0,
        recovery_time=1.0,
        stability_recovery=0.477,
        mobility_loss=0.1,
        data_source='博弈常识推演',
        confidence='高',
        notes='舍弃当前回合节奏，换取免受机体硬损',
    ),
    "D18": DefenseAction(
        code="D18",
        name="快速起身",
        category="ground",
        category_cn="倒地",
        mechanism_keywords='爆发复位',
        applicable_attack_types=['倒地状态'],
        main_cost='耗能与破绽极大',
        direct_defense_level=0.0,
        counter_attack_potential=0.2,
        timing_difficulty=0.3,
        risk_if_failed=0.8,
        energy_cost=400.0,
        recovery_time=1.0,
        stability_recovery=0.11,
        mobility_loss=0.65,
        data_source='运动学推演',
        confidence='高',
        notes='极度耗能的恢复动作',
    ),
    "D19": DefenseAction(
        code="D19",
        name="倒地防御",
        category="ground",
        category_cn="倒地",
        mechanism_keywords='地面蜷缩',
        applicable_attack_types=['倒地被追击'],
        main_cost='无法反击',
        direct_defense_level=0.8,
        counter_attack_potential=0.0,
        timing_difficulty=0.2,
        risk_if_failed=0.9,
        energy_cost=80.0,
        recovery_time=0.5,
        stability_recovery=0.515,
        mobility_loss=0.15,
        data_source='运动学推演',
        confidence='高',
        notes='倒地状态下的被动掩护',
    ),
    "D20": DefenseAction(
        code="D20",
        name="闪→挡→反",
        category="composite",
        category_cn="复合",
        mechanism_keywords='闪避接格挡',
        applicable_attack_types=['复合连招'],
        main_cost='难度极高',
        direct_defense_level=0.6,
        counter_attack_potential=0.7,
        timing_difficulty=0.85,
        risk_if_failed=0.8,
        energy_cost=350.0,
        recovery_time=0.8,
        stability_recovery=0.48,
        mobility_loss=0.55,
        data_source='博弈常识推演',
        confidence='中',
        notes='综合防守体系',
    ),
    "D21": DefenseAction(
        code="D21",
        name="潜→闪→踢",
        category="composite",
        category_cn="复合",
        mechanism_keywords='下潜接反击',
        applicable_attack_types=['高位空挡'],
        main_cost='体能消耗顶峰',
        direct_defense_level=0.4,
        counter_attack_potential=0.8,
        timing_difficulty=0.9,
        risk_if_failed=0.9,
        energy_cost=450.0,
        recovery_time=1.0,
        stability_recovery=0.33,
        mobility_loss=0.7,
        data_source='博弈常识推演',
        confidence='中',
        notes='最复杂的防反连招，失败代价极大',
    ),
    "D22": DefenseAction(
        code="D22",
        name="挡→撤→绕",
        category="composite",
        category_cn="复合",
        mechanism_keywords='退防反打',
        applicable_attack_types=['持续压迫'],
        main_cost='空间要求高',
        direct_defense_level=0.8,
        counter_attack_potential=0.6,
        timing_difficulty=0.75,
        risk_if_failed=0.6,
        energy_cost=300.0,
        recovery_time=0.9,
        stability_recovery=0.615,
        mobility_loss=0.75,
        data_source='博弈常识推演',
        confidence='中',
        notes='退防反打体系',
    ),
}


DEFENSE_ORDER: List[str] = [
'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22'
]


def get_defense_dict() -> Dict[str, Dict[str, Any]]:
    """返回 code -> dict 的防守动作映射。"""
    return {k: v.to_dict() for k, v in DEFENSE_ACTIONS.items()}


def get_defense_list() -> List[Dict[str, Any]]:
    """返回按 DEFENSE_ORDER 排序的防守动作列表。"""
    return [DEFENSE_ACTIONS[k].to_dict() for k in DEFENSE_ORDER]


def get_defense_by_code(code: str) -> Optional[Dict[str, Any]]:
    action = DEFENSE_ACTIONS.get(code)
    return action.to_dict() if action else None


def get_defense_by_name(name: str) -> Optional[Dict[str, Any]]:
    for code in DEFENSE_ORDER:
        action = DEFENSE_ACTIONS[code]
        if action.name == name:
            return action.to_dict()
    return None


def prepare_defense_records() -> List[Dict[str, Any]]:
    """
    兼容 transition.py / matchup_matrix.py 的旧接口。
    """
    return get_defense_list()


if __name__ == "__main__":
    for item in get_defense_list():
        print(
            item["code"],
            item["name"],
            item["direct_defense_level"],
            item["counter_attack_potential"],
            item["timing_difficulty"],
            item["risk_if_failed"],
            item["energy_cost"],
            item["recovery_time"],
            item["stability_recovery"],
            item["mobility_loss"],
        )
