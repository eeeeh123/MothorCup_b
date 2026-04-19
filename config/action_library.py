from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AttackAction:
    """
    单个攻击动作的结构化定义。

    当前版本已与《Q1_攻击动作采集》最终表对齐：
    - 保留旧版 Q1/Q2/Q3 需要的 1~5 映射字段；
    - 新增底层连续变量字段，供 Q1 重新计算与 Q3/Q4 状态转移直接使用；
    - 兼容 transition.py 里使用的别名键。
    """
    code: str  # 编号，如 A01
    name: str  # 动作名称
    category: str  # 拳法 / 腿法 / 组合技 / 特殊技
    part: str  # arm / leg / torso
    target_zones: List[str]  # 主要攻击部位
    attack_range: str  # near / mid / far / near-mid / mid-far
    eta: float  # 躯干质量传导系数
    omega: float  # 等效角速度 rad/s
    dt: float  # 接触时间 s
    Ek: float  # 动能 J
    F: float  # 打击力 N
    Tpre: float  # 前摇 s
    Lreach: float  # 触达距离 m
    theta_sweep: float  # 覆盖角 rad
    Kbreak: float  # 破防系
    Ccombo: float  # 连招度
    Vvul: float  # 暴露度
    Pfall: float  # 失稳率
    Trec: float  # 硬直 s
    Cstam: float  # 能耗 J
    impact_level: float  # 映射冲击强度 1-5
    accuracy_level: float  # 映射命中/压迫 1-5
    balance_break_level: float  # 映射破坏平衡 1-5
    combo_potential: float  # 映射连续追击 1-5
    time_cost: float  # 由 Tpre+0.5*Trec 映射得到的时间成本 1-5
    balance_risk: float  # 映射自身失衡风险 1-5
    energy_cost: float  # 映射能耗 1-5
    counter_risk: float  # 被反制风险 1-5
    pressure_potential: float  # 压制得分潜力 1-5
    tactical_total_score: float  # 综合战术总分
    confidence: str  # 高 / 中 / 低
    data_source: str  # 数据来源
    notes: str  # 备注

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # 兼容旧版/下游模块常用别名
        data.update({
            "kinetic_energy": self.Ek,
            "avg_impact_force": self.F,
            "time_cost_raw": self.Tpre + 0.5 * self.Trec,
            "recovery_time": self.Trec,
            "stamina_cost_raw": self.Cstam,
            "p_fall": self.Pfall,
            "support_loss_proxy": self.balance_risk,
            "rotation_risk_proxy": self.balance_risk,
            "recovery_burden_proxy": self.time_cost,
            "exposure_proxy": self.counter_risk,
            "composite_penalty": COMPOSITE_PENALTY_MAP.get(self.code, 0.0),
        })
        return data


ATTACK_ACTIONS: Dict[str, AttackAction] = {
    "A01": AttackAction(
        code="A01",
        name="直拳",
        category="拳法",
        part="arm",
        target_zones=['头', '躯干'],
        attack_range="near-mid",
        eta=0.2,
        omega=16.0,
        dt=0.04,
        Ek=54.0,
        F=1134.0,
        Tpre=0.15,
        Lreach=0.45,
        theta_sweep=0.1,
        Kbreak=0.2,
        Ccombo=0.9,
        Vvul=0.2,
        Pfall=0.02,
        Trec=0.3,
        Cstam=150.0,
        impact_level=2.0,
        accuracy_level=5.0,
        balance_break_level=1.0,
        combo_potential=5.0,
        time_cost=1.0,
        balance_risk=1.0,
        energy_cost=1.0,
        counter_risk=1.0,
        pressure_potential=3.0,
        tactical_total_score=2.45,
        confidence="高",
        data_source="公式推演+机电映射",
        notes="基础起手技，映射总分极高体现其实用性",
    ),
    "A02": AttackAction(
        code="A02",
        name="勾拳",
        category="拳法",
        part="arm",
        target_zones=['头', '躯干'],
        attack_range="near",
        eta=0.35,
        omega=14.0,
        dt=0.06,
        Ek=25.0,
        F=771.0,
        Tpre=0.2,
        Lreach=0.35,
        theta_sweep=0.5,
        Kbreak=0.4,
        Ccombo=0.8,
        Vvul=0.3,
        Pfall=0.05,
        Trec=0.5,
        Cstam=220.0,
        impact_level=1.0,
        accuracy_level=5.0,
        balance_break_level=2.0,
        combo_potential=5.0,
        time_cost=1.55,
        balance_risk=1.0,
        energy_cost=2.0,
        counter_risk=2.0,
        pressure_potential=3.0,
        tactical_total_score=2.25,
        confidence="中",
        data_source="公式推演+机电映射",
        notes="修正了等效半径，打击力降低，近身特化",
    ),
    "A03": AttackAction(
        code="A03",
        name="组合拳",
        category="拳法",
        part="arm",
        target_zones=['头', '躯干'],
        attack_range="near-mid",
        eta=0.3,
        omega=14.5,
        dt=0.05,
        Ek=44.0,
        F=1096.0,
        Tpre=0.25,
        Lreach=0.45,
        theta_sweep=0.4,
        Kbreak=0.3,
        Ccombo=0.5,
        Vvul=0.4,
        Pfall=0.1,
        Trec=0.8,
        Cstam=400.0,
        impact_level=2.0,
        accuracy_level=4.0,
        balance_break_level=2.0,
        combo_potential=3.0,
        time_cost=2.27,
        balance_risk=2.0,
        energy_cost=3.0,
        counter_risk=3.0,
        pressure_potential=4.0,
        tactical_total_score=1.75,
        confidence="中",
        data_source="公式推演+机电映射",
        notes="大幅下调连招潜力，体现连击收尾特征",
    ),
    "A04": AttackAction(
        code="A04",
        name="摆拳",
        category="拳法",
        part="arm",
        target_zones=['头', '躯干'],
        attack_range="near-mid",
        eta=0.5,
        omega=12.0,
        dt=0.08,
        Ek=30.0,
        F=850.0,
        Tpre=0.3,
        Lreach=0.4,
        theta_sweep=1.2,
        Kbreak=0.5,
        Ccombo=0.6,
        Vvul=0.6,
        Pfall=0.08,
        Trec=0.7,
        Cstam=300.0,
        impact_level=1.0,
        accuracy_level=4.0,
        balance_break_level=3.0,
        combo_potential=4.0,
        time_cost=2.27,
        balance_risk=2.0,
        energy_cost=2.0,
        counter_risk=3.0,
        pressure_potential=3.0,
        tactical_total_score=1.9,
        confidence="高",
        data_source="公式推演+机电映射",
        notes="横扫攻击，覆盖面大但前摇较长",
    ),
    "A05": AttackAction(
        code="A05",
        name="前踢",
        category="腿法",
        part="leg",
        target_zones=['躯干', '四肢'],
        attack_range="mid",
        eta=0.4,
        omega=17.0,
        dt=0.05,
        Ek=512.0,
        F=3712.0,
        Tpre=0.25,
        Lreach=0.65,
        theta_sweep=0.2,
        Kbreak=0.6,
        Ccombo=0.7,
        Vvul=0.3,
        Pfall=0.15,
        Trec=0.6,
        Cstam=280.0,
        impact_level=3.0,
        accuracy_level=4.0,
        balance_break_level=3.0,
        combo_potential=4.0,
        time_cost=1.91,
        balance_risk=2.0,
        energy_cost=2.0,
        counter_risk=2.0,
        pressure_potential=4.0,
        tactical_total_score=2.4,
        confidence="中",
        data_source="公式推演+机电映射",
        notes="直线远距离打击，数据均衡",
    ),
    "A06": AttackAction(
        code="A06",
        name="侧踢",
        category="腿法",
        part="leg",
        target_zones=['躯干'],
        attack_range="mid",
        eta=0.7,
        omega=19.5,
        dt=0.04,
        Ek=674.0,
        F=7320.0,
        Tpre=0.35,
        Lreach=0.65,
        theta_sweep=0.3,
        Kbreak=0.9,
        Ccombo=0.4,
        Vvul=0.5,
        Pfall=0.25,
        Trec=0.9,
        Cstam=450.0,
        impact_level=5.0,
        accuracy_level=3.0,
        balance_break_level=5.0,
        combo_potential=3.0,
        time_cost=2.82,
        balance_risk=3.0,
        energy_cost=3.0,
        counter_risk=4.0,
        pressure_potential=5.0,
        tactical_total_score=2.6,
        confidence="高",
        data_source="公式推演+机电映射",
        notes="动量霸主；破防极高，是核心确反重击",
    ),
    "A07": AttackAction(
        code="A07",
        name="回旋踢",
        category="腿法",
        part="leg",
        target_zones=['头', '躯干'],
        attack_range="mid",
        eta=0.85,
        omega=21.0,
        dt=0.06,
        Ek=782.0,
        F=5971.0,
        Tpre=0.6,
        Lreach=0.7,
        theta_sweep=3.1,
        Kbreak=0.8,
        Ccombo=0.1,
        Vvul=0.8,
        Pfall=0.45,
        Trec=1.5,
        Cstam=600.0,
        impact_level=4.0,
        accuracy_level=1.0,
        balance_break_level=4.0,
        combo_potential=1.0,
        time_cost=4.82,
        balance_risk=5.0,
        energy_cost=4.0,
        counter_risk=5.0,
        pressure_potential=2.0,
        tactical_total_score=1.1,
        confidence="中",
        data_source="公式推演+机电映射",
        notes="动能霸主；但前摇极长且极易失稳，收益期望严重受限",
    ),
    "A08": AttackAction(
        code="A08",
        name="低扫腿",
        category="腿法",
        part="leg",
        target_zones=['下肢'],
        attack_range="near-mid",
        eta=0.5,
        omega=15.0,
        dt=0.08,
        Ek=399.0,
        F=2303.0,
        Tpre=0.3,
        Lreach=0.65,
        theta_sweep=1.5,
        Kbreak=0.5,
        Ccombo=0.5,
        Vvul=0.4,
        Pfall=0.2,
        Trec=0.7,
        Cstam=350.0,
        impact_level=2.0,
        accuracy_level=4.0,
        balance_break_level=3.0,
        combo_potential=3.0,
        time_cost=2.27,
        balance_risk=3.0,
        energy_cost=2.0,
        counter_risk=2.0,
        pressure_potential=3.0,
        tactical_total_score=1.9,
        confidence="高",
        data_source="公式推演+机电映射",
        notes="专攻底盘破坏平衡",
    ),
    "A09": AttackAction(
        code="A09",
        name="膝撞",
        category="腿法",
        part="leg",
        target_zones=['躯干'],
        attack_range="near",
        eta=0.8,
        omega=13.0,
        dt=0.1,
        Ek=87.0,
        F=1146.0,
        Tpre=0.2,
        Lreach=0.25,
        theta_sweep=0.2,
        Kbreak=0.7,
        Ccombo=0.6,
        Vvul=0.2,
        Pfall=0.15,
        Trec=0.8,
        Cstam=250.0,
        impact_level=2.0,
        accuracy_level=5.0,
        balance_break_level=4.0,
        combo_potential=4.0,
        time_cost=2.09,
        balance_risk=2.0,
        energy_cost=2.0,
        counter_risk=3.0,
        pressure_potential=4.0,
        tactical_total_score=2.55,
        confidence="中",
        data_source="公式推演+机电映射",
        notes="修正为大腿半长发力，动能低但传导冲击力巨大",
    ),
    "A10": AttackAction(
        code="A10",
        name="拳腿组合",
        category="组合技",
        part="leg",
        target_zones=['头', '躯干', '下肢'],
        attack_range="near-mid",
        eta=0.45,
        omega=15.5,
        dt=0.05,
        Ek=426.0,
        F=3597.0,
        Tpre=0.3,
        Lreach=0.65,
        theta_sweep=0.8,
        Kbreak=0.6,
        Ccombo=0.4,
        Vvul=0.5,
        Pfall=0.2,
        Trec=1.2,
        Cstam=500.0,
        impact_level=3.0,
        accuracy_level=4.0,
        balance_break_level=3.0,
        combo_potential=3.0,
        time_cost=3.18,
        balance_risk=3.0,
        energy_cost=3.0,
        counter_risk=4.0,
        pressure_potential=4.0,
        tactical_total_score=2.05,
        confidence="中",
        data_source="公式推演+机电映射",
        notes="综合型连段，修正后降低了后续追击潜力",
    ),
    "A11": AttackAction(
        code="A11",
        name="五连踢",
        category="组合技",
        part="leg",
        target_zones=['躯干', '四肢'],
        attack_range="mid",
        eta=0.25,
        omega=16.0,
        dt=0.05,
        Ek=454.0,
        F=2839.0,
        Tpre=0.4,
        Lreach=0.65,
        theta_sweep=0.5,
        Kbreak=0.5,
        Ccombo=0.2,
        Vvul=0.6,
        Pfall=0.35,
        Trec=1.8,
        Cstam=800.0,
        impact_level=3.0,
        accuracy_level=3.0,
        balance_break_level=3.0,
        combo_potential=2.0,
        time_cost=4.64,
        balance_risk=4.0,
        energy_cost=5.0,
        counter_risk=5.0,
        pressure_potential=3.0,
        tactical_total_score=1.45,
        confidence="中",
        data_source="公式推演+机电映射",
        notes="体力消耗极值，修正连招度Ccombo至极低0.2",
    ),
    "A12": AttackAction(
        code="A12",
        name="冲撞",
        category="特殊技",
        part="torso",
        target_zones=['躯干'],
        attack_range="near",
        eta=1.0,
        omega=2.5,
        dt=0.2,
        Ek=131.0,
        F=525.0,
        Tpre=0.5,
        Lreach=1.0,
        theta_sweep=0.6,
        Kbreak=1.0,
        Ccombo=0.3,
        Vvul=0.9,
        Pfall=0.3,
        Trec=1.5,
        Cstam=450.0,
        impact_level=1.0,
        accuracy_level=2.0,
        balance_break_level=5.0,
        combo_potential=2.0,
        time_cost=4.45,
        balance_risk=4.0,
        energy_cost=3.0,
        counter_risk=5.0,
        pressure_potential=5.0,
        tactical_total_score=1.1,
        confidence="高",
        data_source="公式推演+平动动力学",
        notes="修正为质心平动模型(v=2.5m/s)，体现重质量压迫而非高能打击",
    ),
    "A13": AttackAction(
        code="A13",
        name="倒地反击",
        category="特殊技",
        part="arm",
        target_zones=['下肢', '近身部位'],
        attack_range="near",
        eta=0.1,
        omega=8.0,
        dt=0.1,
        Ek=14.0,
        F=151.0,
        Tpre=0.4,
        Lreach=0.3,
        theta_sweep=0.5,
        Kbreak=0.1,
        Ccombo=0.2,
        Vvul=0.1,
        Pfall=0.0,
        Trec=2.0,
        Cstam=100.0,
        impact_level=1.0,
        accuracy_level=3.0,
        balance_break_level=1.0,
        combo_potential=2.0,
        time_cost=5.0,
        balance_risk=1.0,
        energy_cost=1.0,
        counter_risk=1.0,
        pressure_potential=1.0,
        tactical_total_score=1.05,
        confidence="中",
        data_source="公式推演+机电映射",
        notes="略微提升角速度使其具备基础脱困推力",
    ),
}

COMPOSITE_PENALTY_MAP = {
    "A01": 1.0,  # 直拳
    "A02": 1.5,  # 勾拳
    "A03": 2.5,  # 组合拳
    "A04": 2.5,  # 摆拳
    "A05": 2.0,  # 前踢
    "A06": 3.5,  # 侧踢
    "A07": 5.0,  # 回旋踢
    "A08": 2.5,  # 低扫腿
    "A09": 2.0,  # 膝撞
    "A10": 3.5,  # 拳腿组合
    "A11": 4.5,  # 五连踢
    "A12": 4.5,  # 冲撞
    "A13": 1.5,  # 倒地反击
}

ACTION_ORDER: List[str] = [
    "A01", "A02", "A03", "A04", "A05", "A06", "A07",
    "A08", "A09", "A10", "A11", "A12", "A13"
]


def get_action_dict() -> Dict[str, Dict[str, Any]]:
    """返回 code -> dict 的动作映射。"""
    return {k: v.to_dict() for k, v in ATTACK_ACTIONS.items()}


def get_action_list() -> List[Dict[str, Any]]:
    """返回按 ACTION_ORDER 排序的动作列表。"""
    return [ATTACK_ACTIONS[k].to_dict() for k in ACTION_ORDER]


def get_action_by_code(code: str) -> Optional[Dict[str, Any]]:
    """按编号获取动作。"""
    action = ATTACK_ACTIONS.get(code)
    return action.to_dict() if action else None


def get_action_by_name(name: str) -> Optional[Dict[str, Any]]:
    """按动作名称获取动作。"""
    for code in ACTION_ORDER:
        action = ATTACK_ACTIONS[code]
        if action.name == name:
            return action.to_dict()
    return None


def prepare_attack_records() -> List[Dict[str, Any]]:
    """兼容部分分析/矩阵模块的旧接口。"""
    return get_action_list()


if __name__ == "__main__":
    records = prepare_attack_records()
    for r in records[:3]:
        print(r["code"], r["name"], r.get("composite_penalty"))