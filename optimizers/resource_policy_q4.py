from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from models.bo3_state import BO3State, FaultLevel, ResourceType
from models.state import RoundState, PostureState, DistanceState
from models.transition import ActionDecision
from simulators.round_simulator import (
    PolicyFn,
    simple_rule_policy,
    random_policy,
    get_policy_name,
)
from optimizers.policy_q3 import greedy_q3_policy


# =========================================================
# 一、数据结构
# =========================================================
@dataclass
class ResourceDecision:
    side: str
    resource_type: ResourceType
    phase: str                  # pre_round / in_round
    reason: str


@dataclass
class RoundPolicyDecision:
    side: str
    policy: PolicyFn
    reason: str

    @property
    def policy_name(self) -> str:
        return get_policy_name(self.policy)


@dataclass
class ResourcePolicyDecisionBundle:
    pre_round_actions: List[ResourceDecision] = field(default_factory=list)
    in_round_actions: List[ResourceDecision] = field(default_factory=list)
    round_policy_decisions: List[RoundPolicyDecision] = field(default_factory=list)


@dataclass
class FastQ4ResourcePolicyConfig:
    """
    快速 Q4 资源策略配置（事件驱动版）

    设计目标：
    1. 不做在线 Monte Carlo；
    2. 只在局前 / 关键事件点 做资源判断；
    3. 通过“激进 / 平衡 / 保守”三种模板控制 Q3 行为；
    4. 保持与 bo3_simulator.py 当前接口兼容。
    """

    # -------------------------
    # 局前维修：阈值与评分
    # -------------------------
    repair_fault_levels: tuple[FaultLevel, ...] = (FaultLevel.MAJOR, FaultLevel.CRITICAL)
    repair_carry_stability_threshold: float = 18.0
    repair_carry_hp_threshold: float = 15.0
    repair_carry_energy_threshold: float = 22.0

    repair_fault_w: float = 2.0
    repair_hp_w: float = 0.06
    repair_stability_w: float = 0.08
    repair_energy_w: float = 0.05
    repair_match_point_bonus: float = 0.8
    repair_last_one_penalty: float = 0.6
    repair_trigger_score: float = 1.35

    # -------------------------
    # 本局策略模板选择
    # -------------------------
    urgency_trailing_bonus: float = 1.2
    urgency_match_point_bonus: float = 1.0
    urgency_opp_weak_bonus: float = 0.8

    risk_fault_w: float = 0.8
    risk_hp_w: float = 0.035
    risk_stability_w: float = 0.05
    risk_energy_w: float = 0.035

    aggressive_bias: float = 0.3
    conservative_lead_bonus: float = 1.0
    balanced_bias: float = 0.15

    # -------------------------
    # 局内 timeout
    # -------------------------
    timeout_stability_threshold: float = 18.0
    timeout_energy_threshold: float = 15.0
    timeout_off_balance_stability_threshold: float = 25.0
    timeout_match_point_bonus: float = 0.8
    timeout_last_one_penalty: float = 0.6
    timeout_trigger_score: float = 1.05

    # -------------------------
    # 局内 reset
    # -------------------------
    reset_downed_bonus: float = 2.0
    reset_match_point_bonus: float = 0.8
    reset_last_one_penalty: float = 0.7
    reset_trigger_score: float = 1.0

    # -------------------------
    # 模板策略中的快速行为阈值
    # -------------------------
    conservative_energy_hold_threshold: float = 32.0
    conservative_stability_recover_threshold: float = 38.0
    conservative_low_energy_threshold: float = 22.0
    aggressive_attack_energy_threshold: float = 28.0
    aggressive_attack_stability_threshold: float = 34.0


DEFAULT_CONFIG = FastQ4ResourcePolicyConfig()


# =========================================================
# 二、Q3 轻量模板策略
# =========================================================
def aggressive_q3_policy(state: RoundState, side: str, rng) -> ActionDecision:
    """
    激进模板：
    - 高能量/稳定时优先进攻
    - 根据距离快速给出强压制动作
    - 关键时刻仍允许 recover/hold 兜底
    """
    me = state.my if side == "my" else state.opp
    opp = state.opp if side == "my" else state.my

    if me.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
        return ActionDecision("recover", None)

    # 极低能量或低稳定时先自保
    if me.energy < DEFAULT_CONFIG.aggressive_attack_energy_threshold or me.stability < DEFAULT_CONFIG.aggressive_attack_stability_threshold:
        if me.stability < 26 or me.energy < 18:
            return ActionDecision("recover", None)
        return simple_rule_policy(state, side, rng)

    # 对方失衡时优先追击
    if opp.posture == PostureState.OFF_BALANCE:
        if state.distance in [DistanceState.CLINCH, DistanceState.NEAR]:
            return ActionDecision("attack", "A12" if me.energy >= 35 else "A09")
        return ActionDecision("attack", "A03")

    if state.distance == DistanceState.FAR:
        return ActionDecision("attack", "A05" if me.energy < 50 else "A06")
    if state.distance == DistanceState.MID:
        if me.energy >= 60 and me.stability >= 50:
            return ActionDecision("attack", "A06")
        return ActionDecision("attack", "A05")
    if state.distance == DistanceState.NEAR:
        if me.energy >= 38 and me.stability >= 45:
            return ActionDecision("attack", "A12")
        return ActionDecision("attack", "A03")

    # clinch
    return ActionDecision("attack", "A09" if me.energy < 45 else "A12")


def balanced_q3_policy(state: RoundState, side: str, rng) -> ActionDecision:
    """
    平衡模板：
    直接复用 simple_rule_policy，保证速度与稳定性。
    """
    return simple_rule_policy(state, side, rng)


def conservative_q3_policy(state: RoundState, side: str, rng) -> ActionDecision:
    """
    保守模板：
    - 风险高时优先防守/恢复
    - 仅在对手明显失衡或我方状态较稳时做追击
    """
    me = state.my if side == "my" else state.opp
    opp = state.opp if side == "my" else state.my

    if me.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
        return ActionDecision("recover", None)

    if opp.posture == PostureState.OFF_BALANCE and me.stability >= 42 and me.energy >= 28:
        if state.distance in [DistanceState.CLINCH, DistanceState.NEAR]:
            return ActionDecision("attack", "A03")
        return ActionDecision("attack", "A05")

    if me.stability < DEFAULT_CONFIG.conservative_stability_recover_threshold:
        return ActionDecision("recover", None)

    if me.energy < DEFAULT_CONFIG.conservative_low_energy_threshold:
        return ActionDecision("hold", None)

    if state.distance == DistanceState.FAR:
        return ActionDecision("defend", "D22")
    if state.distance == DistanceState.MID:
        return ActionDecision("defend", "D15")
    if state.distance == DistanceState.NEAR:
        return ActionDecision("defend", "D20")

    # clinch
    return ActionDecision("defend", "D12")


# =========================================================
# 三、快速 Q4 资源策略引擎
# =========================================================
class FastEventDrivenQ4ResourcePolicy:
    def __init__(self, config: FastQ4ResourcePolicyConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    # -----------------------------------------------------
    # 基础访问
    # -----------------------------------------------------
    @staticmethod
    def _get_side_state(bo3_state: BO3State, side: str):
        return bo3_state.my if side == "my" else bo3_state.opp

    @staticmethod
    def _get_other_side_state(bo3_state: BO3State, side: str):
        return bo3_state.opp if side == "my" else bo3_state.my

    @staticmethod
    def _get_round_fighter(round_state: RoundState, side: str):
        return round_state.my if side == "my" else round_state.opp

    @staticmethod
    def _fault_penalty_value(fault_level: FaultLevel) -> float:
        if fault_level == FaultLevel.HEALTHY:
            return 0.0
        if fault_level == FaultLevel.MINOR:
            return 1.0
        if fault_level == FaultLevel.MAJOR:
            return 2.0
        return 3.0

    @staticmethod
    def _is_match_point(bo3_state: BO3State, side: str) -> bool:
        """
        当前 side 是否处于“这一局赢了就拿下系列赛”的赛点。
        """
        me = bo3_state.my if side == "my" else bo3_state.opp
        return me.round_wins >= 1

    @staticmethod
    def _is_behind(bo3_state: BO3State, side: str) -> bool:
        me = bo3_state.my if side == "my" else bo3_state.opp
        opp = bo3_state.opp if side == "my" else bo3_state.my
        return me.round_wins < opp.round_wins

    @staticmethod
    def _is_ahead(bo3_state: BO3State, side: str) -> bool:
        me = bo3_state.my if side == "my" else bo3_state.opp
        opp = bo3_state.opp if side == "my" else bo3_state.my
        return me.round_wins > opp.round_wins

    # -----------------------------------------------------
    # 局前维修
    # -----------------------------------------------------
    def _repair_score(self, bo3_state: BO3State, side: str) -> float:
        me = self._get_side_state(bo3_state, side)
        score = 0.0

        if me.fault_level in self.config.repair_fault_levels:
            score += self.config.repair_fault_w * self._fault_penalty_value(me.fault_level)

        score += self.config.repair_hp_w * me.carry_hp_debt
        score += self.config.repair_stability_w * me.carry_stability_debt
        score += self.config.repair_energy_w * me.carry_energy_debt

        if self._is_match_point(bo3_state, side):
            score += self.config.repair_match_point_bonus

        if me.repairs_left <= 1:
            score -= self.config.repair_last_one_penalty

        return score

    def should_repair_pre_round(self, bo3_state: BO3State, side: str) -> Optional[str]:
        me = self._get_side_state(bo3_state, side)

        if me.repairs_left <= 0:
            return None

        # 强规则优先
        if me.fault_level in self.config.repair_fault_levels:
            return "故障等级达到 MAJOR/CRITICAL，赛前优先维修"

        if me.carry_stability_debt >= self.config.repair_carry_stability_threshold:
            return "跨局稳定性惩罚较高，赛前维修"
        if me.carry_hp_debt >= self.config.repair_carry_hp_threshold:
            return "跨局损伤惩罚较高，赛前维修"
        if me.carry_energy_debt >= self.config.repair_carry_energy_threshold:
            return "跨局能量惩罚较高，赛前维修"

        # 快速评分补充
        score = self._repair_score(bo3_state, side)
        if score >= self.config.repair_trigger_score:
            return f"赛前维修评分达阈值（score={score:.2f}）"

        return None

    def decide_pre_round_actions(self, bo3_state: BO3State) -> List[ResourceDecision]:
        decisions: List[ResourceDecision] = []

        for side in ["my", "opp"]:
            reason = self.should_repair_pre_round(bo3_state, side)
            if reason is not None:
                decisions.append(
                    ResourceDecision(
                        side=side,
                        resource_type=ResourceType.REPAIR,
                        phase="pre_round",
                        reason=reason,
                    )
                )

        return decisions

    # -----------------------------------------------------
    # 本局模板策略选择
    # -----------------------------------------------------
    def _urgency_score(self, bo3_state: BO3State, side: str) -> float:
        me = self._get_side_state(bo3_state, side)
        opp = self._get_other_side_state(bo3_state, side)

        score = 0.0
        if self._is_behind(bo3_state, side):
            score += self.config.urgency_trailing_bonus
        if self._is_match_point(bo3_state, side):
            score += self.config.urgency_match_point_bonus

        # 对手状态越差，我方越适合抢攻
        opp_weakness = 0.0
        opp_weakness += 0.04 * opp.carry_hp_debt
        opp_weakness += 0.05 * opp.carry_stability_debt
        opp_weakness += 0.03 * opp.carry_energy_debt
        opp_weakness += 0.40 * self._fault_penalty_value(opp.fault_level)
        score += min(self.config.urgency_opp_weak_bonus, opp_weakness)

        return score

    def _self_risk_score(self, bo3_state: BO3State, side: str) -> float:
        me = self._get_side_state(bo3_state, side)
        score = 0.0
        score += self.config.risk_fault_w * self._fault_penalty_value(me.fault_level)
        score += self.config.risk_hp_w * me.carry_hp_debt
        score += self.config.risk_stability_w * me.carry_stability_debt
        score += self.config.risk_energy_w * me.carry_energy_debt
        return score

    def select_round_policy(
        self,
        bo3_state: BO3State,
        side: str,
        default_policy: PolicyFn,
    ) -> RoundPolicyDecision:
        # 随机策略保持随机，避免被外层重写
        if default_policy is random_policy:
            return RoundPolicyDecision(
                side=side,
                policy=random_policy,
                reason="基准随机策略，不做外层改写",
            )

        urgency = self._urgency_score(bo3_state, side)
        self_risk = self._self_risk_score(bo3_state, side)
        lead_bonus = self.config.conservative_lead_bonus if self._is_ahead(bo3_state, side) else 0.0

        aggressive_score = self.config.aggressive_bias + urgency - 0.60 * self_risk
        balanced_score = self.config.balanced_bias + 0.50 * urgency - 0.35 * self_risk
        conservative_score = lead_bonus + 0.75 * self_risk - 0.40 * urgency

        # 强规则：严重受损时优先保守
        me = self._get_side_state(bo3_state, side)
        if me.fault_level in (FaultLevel.MAJOR, FaultLevel.CRITICAL):
            return RoundPolicyDecision(
                side=side,
                policy=conservative_q3_policy,
                reason="高故障等级，进入保守模板",
            )

        # 选择最高模板
        template_scores = [
            ("aggressive", aggressive_score, aggressive_q3_policy),
            ("balanced", balanced_score, balanced_q3_policy),
            ("conservative", conservative_score, conservative_q3_policy),
        ]
        template_scores.sort(key=lambda x: x[1], reverse=True)
        template_name, template_score, chosen_policy = template_scores[0]

        return RoundPolicyDecision(
            side=side,
            policy=chosen_policy,
            reason=(
                f"模板={template_name}"
                f" | urgency={urgency:.2f}"
                f" | risk={self_risk:.2f}"
                f" | score={template_score:.2f}"
            ),
        )

    def decide_round_policies(
        self,
        bo3_state: BO3State,
        my_default_policy: PolicyFn,
        opp_default_policy: PolicyFn,
    ) -> List[RoundPolicyDecision]:
        return [
            self.select_round_policy(bo3_state, "my", my_default_policy),
            self.select_round_policy(bo3_state, "opp", opp_default_policy),
        ]

    # -----------------------------------------------------
    # 局内 timeout / reset
    # -----------------------------------------------------
    def _timeout_score(self, bo3_state: BO3State, state: RoundState, side: str) -> float:
        fighter = self._get_round_fighter(state, side)
        side_state = self._get_side_state(bo3_state, side)

        score = 0.0
        if fighter.stability <= self.config.timeout_stability_threshold:
            score += 0.9
        if fighter.energy <= self.config.timeout_energy_threshold:
            score += 0.8
        if fighter.posture == PostureState.OFF_BALANCE and fighter.stability <= self.config.timeout_off_balance_stability_threshold:
            score += 0.7

        if self._is_match_point(bo3_state, side):
            score += self.config.timeout_match_point_bonus

        if side_state.timeouts_left <= 1:
            score -= self.config.timeout_last_one_penalty

        return score

    def should_timeout_in_round(self, bo3_state: BO3State, state: RoundState, side: str) -> Optional[str]:
        fighter = self._get_round_fighter(state, side)
        side_state = self._get_side_state(bo3_state, side)

        if state.is_finished() or side_state.timeouts_left <= 0:
            return None

        # 强规则：极端危险状态直接 timeout
        if fighter.stability <= self.config.timeout_stability_threshold:
            return "稳定性过低，触发局内暂停"
        if fighter.energy <= self.config.timeout_energy_threshold:
            return "能量过低，触发局内暂停"
        if fighter.posture == PostureState.OFF_BALANCE and fighter.stability <= self.config.timeout_off_balance_stability_threshold:
            return "失衡且稳定性偏低，触发局内暂停"

        score = self._timeout_score(bo3_state, state, side)
        if score >= self.config.timeout_trigger_score:
            return f"局内暂停评分达阈值（score={score:.2f}）"

        return None

    def _reset_score(self, bo3_state: BO3State, state: RoundState, side: str) -> float:
        fighter = self._get_round_fighter(state, side)
        side_state = self._get_side_state(bo3_state, side)

        score = 0.0
        if fighter.posture == PostureState.DOWNED:
            score += self.config.reset_downed_bonus

        if self._is_match_point(bo3_state, side):
            score += self.config.reset_match_point_bonus

        if side_state.resets_left <= 1:
            score -= self.config.reset_last_one_penalty

        return score

    def should_reset_in_round(self, bo3_state: BO3State, state: RoundState, side: str) -> Optional[str]:
        fighter = self._get_round_fighter(state, side)
        side_state = self._get_side_state(bo3_state, side)

        if state.is_finished() or side_state.resets_left <= 0:
            return None

        # 强规则：倒地优先 reset
        if fighter.posture == PostureState.DOWNED:
            return "倒地后触发人工复位"

        score = self._reset_score(bo3_state, state, side)
        if score >= self.config.reset_trigger_score:
            return f"人工复位评分达阈值（score={score:.2f}）"

        return None

    def decide_in_round_actions(
        self,
        bo3_state: BO3State,
        state: RoundState,
    ) -> List[ResourceDecision]:
        decisions: List[ResourceDecision] = []

        for side in ["my", "opp"]:
            reset_reason = self.should_reset_in_round(bo3_state, state, side)
            if reset_reason is not None:
                decisions.append(
                    ResourceDecision(
                        side=side,
                        resource_type=ResourceType.RESET,
                        phase="in_round",
                        reason=reset_reason,
                    )
                )
                # reset 优先，不叠加 timeout
                continue

            timeout_reason = self.should_timeout_in_round(bo3_state, state, side)
            if timeout_reason is not None:
                decisions.append(
                    ResourceDecision(
                        side=side,
                        resource_type=ResourceType.TIMEOUT,
                        phase="in_round",
                        reason=timeout_reason,
                    )
                )

        return decisions

    # -----------------------------------------------------
    # 整体决策打包
    # -----------------------------------------------------
    def decide_bundle(
        self,
        bo3_state: BO3State,
        my_default_policy: PolicyFn,
        opp_default_policy: PolicyFn,
        current_round_state: Optional[RoundState] = None,
    ) -> ResourcePolicyDecisionBundle:
        bundle = ResourcePolicyDecisionBundle()
        bundle.pre_round_actions = self.decide_pre_round_actions(bo3_state)
        bundle.round_policy_decisions = self.decide_round_policies(
            bo3_state=bo3_state,
            my_default_policy=my_default_policy,
            opp_default_policy=opp_default_policy,
        )

        if current_round_state is not None:
            bundle.in_round_actions = self.decide_in_round_actions(
                bo3_state=bo3_state,
                state=current_round_state,
            )

        return bundle


# =========================================================
# 四、对外暴露的默认策略对象与包装函数
# =========================================================
_default_resource_policy_engine = FastEventDrivenQ4ResourcePolicy(DEFAULT_CONFIG)


def decide_pre_round_resource_actions(bo3_state: BO3State) -> List[ResourceDecision]:
    return _default_resource_policy_engine.decide_pre_round_actions(bo3_state)


def decide_round_policies(
    bo3_state: BO3State,
    my_default_policy: PolicyFn,
    opp_default_policy: PolicyFn,
) -> List[RoundPolicyDecision]:
    return _default_resource_policy_engine.decide_round_policies(
        bo3_state=bo3_state,
        my_default_policy=my_default_policy,
        opp_default_policy=opp_default_policy,
    )


def decide_in_round_resource_actions(
    bo3_state: BO3State,
    current_round_state: RoundState,
) -> List[ResourceDecision]:
    return _default_resource_policy_engine.decide_in_round_actions(
        bo3_state=bo3_state,
        state=current_round_state,
    )


# =========================================================
# 五、把决策真正执行到 BO3State 上
# =========================================================
def apply_resource_decisions_to_bo3_state(
    bo3_state: BO3State,
    decisions: List[ResourceDecision],
) -> List[ResourceDecision]:
    applied: List[ResourceDecision] = []

    for item in decisions:
        ok = False
        if item.resource_type == ResourceType.REPAIR:
            ok = bo3_state.use_emergency_repair(item.side)
        elif item.resource_type == ResourceType.RESET:
            ok = bo3_state.use_manual_reset(item.side)
        elif item.resource_type == ResourceType.TIMEOUT:
            ok = bo3_state.use_timeout(item.side)

        if ok:
            applied.append(item)

    return applied


# =========================================================
# 六、自测
# =========================================================
if __name__ == "__main__":
    from models.bo3_state import create_initial_bo3_state

    bo3 = create_initial_bo3_state()

    print("resource_policy_q4.py（快速事件驱动版）自测开始")
    print("初始比分:", bo3.scoreline())
    print("我方资源:", bo3.my.to_dict())
    print("对方资源:", bo3.opp.to_dict())

    # 构造一组跨局损伤
    bo3.my.carry_stability_debt = 20.0
    bo3.my.carry_energy_debt = 18.0
    bo3.my.accumulated_fault_score = 35.0
    bo3._refresh_fault_level(bo3.my)

    pre_actions = decide_pre_round_resource_actions(bo3)
    print("\n局前资源决策：")
    for a in pre_actions:
        print(a.side, a.resource_type.value, a.reason)

    applied = apply_resource_decisions_to_bo3_state(bo3, pre_actions)
    print("成功执行数量:", len(applied))
    print("维修后我方资源:", bo3.my.to_dict())

    policy_decisions = decide_round_policies(
        bo3_state=bo3,
        my_default_policy=greedy_q3_policy,
        opp_default_policy=simple_rule_policy,
    )
    print("\n本局模板策略决策：")
    for p in policy_decisions:
        print(p.side, p.policy_name, p.reason)

    print("\nresource_policy_q4.py（快速事件驱动版）自测完成")
