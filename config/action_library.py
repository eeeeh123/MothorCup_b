from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class AttackAction:
    """
    单个攻击动作的结构化定义。

    说明：
    1. 本文件中的数值是“第一版先验参数”，用于让 Q1 先跑通。
    2. 后续应由资料检索、公开视频观察、团队评估或敏感性分析进行校准。
    3. 这里的字段既服务于 Q1 动作动力学分析，也为 Q2/Q3/Q4 预留接口。
    """

    # 基本信息
    code: str
    name: str
    category: str  # 拳法 / 腿法 / 组合技 / 特殊技
    part: str      # arm / leg / torso
    target_zones: List[str]
    attack_range: str  # near / mid / far / near-mid / mid-far

    # 物理代理参数（第一版先验）
    eta: float     # 躯干质量传导系数（0~1）
    omega: float   # 发力角速度/等效角速度（rad/s）
    dt: float      # 有效碰撞或作用时间（s）

    # 战术与稳定性相关参数（建议 1~5 分）
    impact_level: float          # 冲击强度
    accuracy_level: float        # 命中概率
    balance_break_level: float   # 破坏对手平衡能力
    combo_potential: float       # 连续追击潜力
    time_cost: float             # 动作耗时（高=更慢）
    balance_risk: float          # 自身失衡风险（高=更危险）
    energy_cost: float           # 能耗水平
    counter_risk: float          # 被反制风险
    pressure_potential: float    # 压制得分潜力

    # 备注
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# -------------------------------------------------------------------
# 设计口径说明
# -------------------------------------------------------------------
# 根据题目附录一：
# - “左直拳/右直拳”合并为：直拳
# - “左勾拳/右勾拳”合并为：勾拳
# - “回旋踢/转身踢”合并为：回旋踢
#
# 因而使用如下 13 类动作：
# 1 直拳
# 2 勾拳
# 3 组合拳
# 4 摆拳
# 5 前踢
# 6 侧踢
# 7 回旋踢
# 8 低扫腿
# 9 膝撞
# 10 拳腿组合
# 11 五连踢
# 12 冲撞
# 13 倒地反击
#
# 这与题面“13种攻击动作”的说法保持一致。
# -------------------------------------------------------------------


ATTACK_ACTIONS: Dict[str, AttackAction] = {
    "A01": AttackAction(
        code="A01",
        name="直拳",
        category="拳法",
        part="arm",
        target_zones=["头部", "躯干"],
        attack_range="near-mid",
        eta=0.22,
        omega=12.0,
        dt=0.05,
        impact_level=3.0,
        accuracy_level=4.5,
        balance_break_level=2.5,
        combo_potential=4.0,
        time_cost=1.5,
        balance_risk=1.5,
        energy_cost=1.8,
        counter_risk=2.0,
        pressure_potential=3.2,
        notes="直线快速前伸，命中率高，适合起手和衔接。"
    ),
    "A02": AttackAction(
        code="A02",
        name="勾拳",
        category="拳法",
        part="arm",
        target_zones=["头部", "躯干"],
        attack_range="near",
        eta=0.40,
        omega=10.0,
        dt=0.08,
        impact_level=3.5,
        accuracy_level=3.2,
        balance_break_level=3.0,
        combo_potential=3.4,
        time_cost=2.2,
        balance_risk=2.2,
        energy_cost=2.2,
        counter_risk=2.8,
        pressure_potential=3.3,
        notes="侧向弧线攻击，近身威胁较大，但更依赖时机。"
    ),
    "A03": AttackAction(
        code="A03",
        name="组合拳",
        category="拳法",
        part="arm",
        target_zones=["头部", "躯干"],
        attack_range="near-mid",
        eta=0.30,
        omega=11.0,
        dt=0.06,
        impact_level=3.8,
        accuracy_level=3.8,
        balance_break_level=3.2,
        combo_potential=4.6,
        time_cost=2.5,
        balance_risk=2.0,
        energy_cost=2.6,
        counter_risk=2.8,
        pressure_potential=4.0,
        notes="直拳+勾拳连续连击，重在节奏压制与连续得分。"
    ),
    "A04": AttackAction(
        code="A04",
        name="摆拳",
        category="拳法",
        part="arm",
        target_zones=["头部", "躯干"],
        attack_range="near-mid",
        eta=0.60,
        omega=8.0,
        dt=0.10,
        impact_level=4.0,
        accuracy_level=2.8,
        balance_break_level=3.6,
        combo_potential=2.6,
        time_cost=3.0,
        balance_risk=3.0,
        energy_cost=2.8,
        counter_risk=3.5,
        pressure_potential=3.4,
        notes="大弧度摆臂，威力较强，但前摇较大、容易被预判。"
    ),
    "A05": AttackAction(
        code="A05",
        name="前踢",
        category="腿法",
        part="leg",
        target_zones=["躯干", "四肢"],
        attack_range="mid",
        eta=0.30,
        omega=14.0,
        dt=0.06,
        impact_level=4.0,
        accuracy_level=3.8,
        balance_break_level=3.5,
        combo_potential=3.0,
        time_cost=2.4,
        balance_risk=2.6,
        energy_cost=3.0,
        counter_risk=2.8,
        pressure_potential=3.5,
        notes="直线踢击，兼顾距离与爆发，适合中距离压迫。"
    ),
    "A06": AttackAction(
        code="A06",
        name="侧踢",
        category="腿法",
        part="leg",
        target_zones=["躯干"],
        attack_range="mid",
        eta=0.70,
        omega=15.0,
        dt=0.07,
        impact_level=4.8,
        accuracy_level=3.2,
        balance_break_level=4.5,
        combo_potential=2.4,
        time_cost=3.0,
        balance_risk=3.8,
        energy_cost=3.8,
        counter_risk=3.5,
        pressure_potential=4.2,
        notes="爆发力强，破坏平衡能力突出，但单腿支撑风险较高。"
    ),
    "A07": AttackAction(
        code="A07",
        name="回旋踢",
        category="腿法",
        part="leg",
        target_zones=["头部", "躯干"],
        attack_range="mid",
        eta=0.80,
        omega=8.0,
        dt=0.12,
        impact_level=4.6,
        accuracy_level=2.2,
        balance_break_level=4.3,
        combo_potential=1.8,
        time_cost=4.2,
        balance_risk=4.6,
        energy_cost=4.2,
        counter_risk=4.2,
        pressure_potential=3.8,
        notes="大角度转体带动腿部扫踢，观赏性和威慑力强，但稳定性代价大。"
    ),
    "A08": AttackAction(
        code="A08",
        name="低扫腿",
        category="腿法",
        part="leg",
        target_zones=["下肢"],
        attack_range="near-mid",
        eta=0.50,
        omega=10.0,
        dt=0.09,
        impact_level=3.6,
        accuracy_level=3.6,
        balance_break_level=4.6,
        combo_potential=3.2,
        time_cost=2.8,
        balance_risk=2.8,
        energy_cost=3.0,
        counter_risk=3.0,
        pressure_potential=3.7,
        notes="主要针对下肢，核心价值在于破坏对方重心与步态。"
    ),
    "A09": AttackAction(
        code="A09",
        name="膝撞",
        category="腿法",
        part="leg",
        target_zones=["躯干"],
        attack_range="near",
        eta=0.90,
        omega=8.0,
        dt=0.15,
        impact_level=4.2,
        accuracy_level=3.0,
        balance_break_level=4.0,
        combo_potential=2.5,
        time_cost=2.8,
        balance_risk=2.7,
        energy_cost=3.4,
        counter_risk=3.2,
        pressure_potential=3.6,
        notes="近身冲击强，适合缠斗和贴身阶段。"
    ),
    "A10": AttackAction(
        code="A10",
        name="拳腿组合",
        category="组合技",
        part="leg",
        target_zones=["头部", "躯干", "下肢"],
        attack_range="near-mid",
        eta=0.40,
        omega=12.0,
        dt=0.06,
        impact_level=4.1,
        accuracy_level=3.6,
        balance_break_level=3.8,
        combo_potential=4.7,
        time_cost=3.3,
        balance_risk=3.0,
        energy_cost=3.6,
        counter_risk=3.5,
        pressure_potential=4.4,
        notes="通过上下段切换提高压制性，但控制复杂度更高。"
    ),
    "A11": AttackAction(
        code="A11",
        name="五连踢",
        category="组合技",
        part="leg",
        target_zones=["躯干", "四肢"],
        attack_range="mid",
        eta=0.20,
        omega=10.0,
        dt=0.05,
        impact_level=3.8,
        accuracy_level=3.0,
        balance_break_level=3.8,
        combo_potential=5.0,
        time_cost=4.6,
        balance_risk=4.0,
        energy_cost=4.8,
        counter_risk=4.0,
        pressure_potential=4.6,
        notes="连续多次踢击连招，核心优势在连续压制，缺点是能耗高、节奏长。"
    ),
    "A12": AttackAction(
        code="A12",
        name="冲撞",
        category="特殊技",
        part="torso",
        target_zones=["躯干"],
        attack_range="near",
        eta=1.00,
        omega=6.0,
        dt=0.20,
        impact_level=4.4,
        accuracy_level=3.4,
        balance_break_level=4.8,
        combo_potential=2.0,
        time_cost=3.5,
        balance_risk=3.6,
        energy_cost=4.0,
        counter_risk=3.8,
        pressure_potential=4.8,
        notes="以身体整体质量形成冲击，适合制造压制与失衡。"
    ),
    "A13": AttackAction(
        code="A13",
        name="倒地反击",
        category="特殊技",
        part="arm",
        target_zones=["下肢", "近身部位"],
        attack_range="near",
        eta=0.10,
        omega=4.0,
        dt=0.15,
        impact_level=2.0,
        accuracy_level=2.8,
        balance_break_level=2.4,
        combo_potential=1.5,
        time_cost=3.8,
        balance_risk=3.5,
        energy_cost=2.5,
        counter_risk=4.2,
        pressure_potential=2.2,
        notes="特殊状态下的应急动作，更多用于减少失分或争取节奏恢复。"
    ),
}


ACTION_ORDER: List[str] = [
    "A01", "A02", "A03", "A04", "A05", "A06", "A07",
    "A08", "A09", "A10", "A11", "A12", "A13"
]


def get_action_dict() -> Dict[str, dict]:
    """
    返回适合 DataFrame 或后续计算直接使用的字典形式。
    """
    return {k: v.to_dict() for k, v in ATTACK_ACTIONS.items()}


def get_action_list() -> List[dict]:
    """
    返回按 ACTION_ORDER 排序的动作列表。
    """
    return [ATTACK_ACTIONS[k].to_dict() for k in ACTION_ORDER]


if __name__ == "__main__":
    for item in get_action_list():
        print(item["code"], item["name"], item["category"])