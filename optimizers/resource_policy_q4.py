from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.bo3_state import BO3State, FaultLevel, ResourceType
from models.state import RoundState, PostureState, DistanceState, InitiativeState
from models.transition import ActionDecision
from simulators.round_simulator import (
    PolicyFn,
    simple_rule_policy,
    random_policy,
    get_policy_name,
)
from optimizers.policy_q3 import greedy_q3_policy


@dataclass
class ResourceDecision:
    side: str
    resource_type: ResourceType
    phase: str
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
class Q4ResourcePolicyConfig:
    repair_fault_levels: tuple[FaultLevel, ...] = (FaultLevel.MAJOR, FaultLevel.CRITICAL)
    repair_carry_stability_threshold: float = 18.0
    repair_carry_hp_threshold: float = 15.0
    repair_carry_energy_threshold: float = 22.0

    timeout_stability_threshold: float = 18.0
    timeout_energy_threshold: float = 15.0
    timeout_off_balance_stability_threshold: float = 25.0

    heavy_damage_fault_levels: tuple[FaultLevel, ...] = (FaultLevel.MAJOR, FaultLevel.CRITICAL)
    heavy_damage_carry_stability_threshold: float = 16.0
    heavy_damage_carry_hp_threshold: float = 12.0
    heavy_damage_carry_energy_threshold: float = 20.0

    greedy_to_simple_when_leading_stability_threshold: float = 10.0
    greedy_to_simple_when_leading_energy_threshold: float = 14.0

    protective_stability_threshold: float = 38.0
    protective_energy_threshold: float = 22.0

    momentum_energy_threshold: float = 34.0
    momentum_stability_threshold: float = 40.0

    risk_high_risk_actions: tuple[str, ...] = ("A06", "A07", "A10", "A11", "A12")
    risk_aware_energy_threshold: float = 42.0
    risk_aware_stability_threshold: float = 46.0


DEFAULT_CONFIG = Q4ResourcePolicyConfig()


def protective_q3_policy(state: RoundState, side: str, rng) -> ActionDecision:
    me = state.my if side == "my" else state.opp
    opp = state.opp if side == "my" else state.my

    if me.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
        return ActionDecision("recover", None)

    if opp.posture == PostureState.OFF_BALANCE and me.stability >= 42 and me.energy >= 26:
        if state.distance in [DistanceState.CLINCH, DistanceState.NEAR]:
            return ActionDecision("attack", "A03")
        return ActionDecision("attack", "A05")

    if me.stability < DEFAULT_CONFIG.protective_stability_threshold:
        return ActionDecision("recover", None)

    if me.energy < DEFAULT_CONFIG.protective_energy_threshold:
        return ActionDecision("hold", None)

    if state.distance == DistanceState.FAR:
        return ActionDecision("defend", "D22")
    if state.distance == DistanceState.MID:
        return ActionDecision("defend", "D15")
    if state.distance == DistanceState.NEAR:
        return ActionDecision("defend", "D20")
    return ActionDecision("defend", "D12")


def momentum_q3_policy(state: RoundState, side: str, rng) -> ActionDecision:
    me = state.my if side == "my" else state.opp
    opp = state.opp if side == "my" else state.my

    if me.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
        return ActionDecision("recover", None)

    low_state = (
        me.energy < DEFAULT_CONFIG.momentum_energy_threshold
        or me.stability < DEFAULT_CONFIG.momentum_stability_threshold
    )
    if low_state:
        return simple_rule_policy(state, side, rng)

    if opp.posture == PostureState.OFF_BALANCE:
        return greedy_q3_policy(state, side, rng)

    has_initiative = (
        (side == "my" and state.initiative == InitiativeState.MY)
        or (side == "opp" and state.initiative == InitiativeState.OPP)
    )
    if has_initiative:
        return greedy_q3_policy(state, side, rng)

    if state.distance == DistanceState.MID:
        return ActionDecision("attack", "A05")
    if state.distance == DistanceState.NEAR:
        return ActionDecision("attack", "A03")

    return simple_rule_policy(state, side, rng)


def risk_aware_q3_policy(state: RoundState, side: str, rng) -> ActionDecision:
    me = state.my if side == "my" else state.opp

    if me.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
        return ActionDecision("recover", None)

    greedy_action = greedy_q3_policy(state, side, rng)

    if greedy_action.action_type == "attack" and greedy_action.action_key in DEFAULT_CONFIG.risk_high_risk_actions:
        if me.energy < DEFAULT_CONFIG.risk_aware_energy_threshold or me.stability < DEFAULT_CONFIG.risk_aware_stability_threshold:
            return simple_rule_policy(state, side, rng)

    return greedy_action


Q4_COMPARE_POLICY_LIBRARY: Dict[str, PolicyFn] = {
    "greedy_q3_policy": greedy_q3_policy,
    "simple_rule_policy": simple_rule_policy,
    "protective_q3_policy": protective_q3_policy,
    "momentum_q3_policy": momentum_q3_policy,
    "risk_aware_q3_policy": risk_aware_q3_policy,
    "random_policy": random_policy,
}

Q4_COMPARE_POLICY_DESCRIPTION: Dict[str, str] = {
    "greedy_q3_policy": "一步前瞻的主策略基线",
    "simple_rule_policy": "轻量规则策略基线",
    "protective_q3_policy": "保护型策略：优先防守/恢复，仅在明显优势时追击",
    "momentum_q3_policy": "动量型策略：主动权/对手失衡时加强压制，否则退回 simple_rule",
    "risk_aware_q3_policy": "风险感知策略：greedy 为主，但在高风险动作上做状态过滤",
    "random_policy": "随机策略，仅用于下界对照",
}


def get_q4_compare_policy_library(include_random: bool = True) -> Dict[str, PolicyFn]:
    if include_random:
        return dict(Q4_COMPARE_POLICY_LIBRARY)
    return {k: v for k, v in Q4_COMPARE_POLICY_LIBRARY.items() if k != "random_policy"}


class RuleBasedQ4ResourcePolicy:
    def __init__(self, config: Q4ResourcePolicyConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._round_policy_cache: Dict[tuple, List[RoundPolicyDecision]] = {}

    @staticmethod
    def _get_side_state(bo3_state: BO3State, side: str):
        return bo3_state.my if side == "my" else bo3_state.opp

    @staticmethod
    def _get_other_side_state(bo3_state: BO3State, side: str):
        return bo3_state.opp if side == "my" else bo3_state.my

    @staticmethod
    def _get_round_fighter(round_state: RoundState, side: str):
        return round_state.my if side == "my" else round_state.opp

    def should_repair_pre_round(self, bo3_state: BO3State, side: str) -> Optional[str]:
        side_state = self._get_side_state(bo3_state, side)

        if side_state.repairs_left <= 0:
            return None

        if side_state.fault_level in self.config.repair_fault_levels:
            return "故障等级达到 MAJOR/CRITICAL，赛前优先维修"
        if side_state.carry_stability_debt >= self.config.repair_carry_stability_threshold:
            return "跨局稳定性惩罚较高，赛前维修"
        if side_state.carry_hp_debt >= self.config.repair_carry_hp_threshold:
            return "跨局损伤惩罚较高，赛前维修"
        if side_state.carry_energy_debt >= self.config.repair_carry_energy_threshold:
            return "跨局能量惩罚较高，赛前维修"

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

    def _is_heavy_damage(self, me) -> bool:
        return (
            me.fault_level in self.config.heavy_damage_fault_levels
            or me.carry_stability_debt >= self.config.heavy_damage_carry_stability_threshold
            or me.carry_hp_debt >= self.config.heavy_damage_carry_hp_threshold
            or me.carry_energy_debt >= self.config.heavy_damage_carry_energy_threshold
        )

    def select_round_policy(self, bo3_state: BO3State, side: str, default_policy: PolicyFn) -> RoundPolicyDecision:
        me = self._get_side_state(bo3_state, side)
        opp = self._get_other_side_state(bo3_state, side)

        if default_policy is random_policy:
            return RoundPolicyDecision(side=side, policy=random_policy, reason="基准随机策略，不做外层改写")

        trailing = me.round_wins < opp.round_wins
        leading = me.round_wins > opp.round_wins
        heavy_damage = self._is_heavy_damage(me)

        if heavy_damage:
            if default_policy in (greedy_q3_policy, momentum_q3_policy, risk_aware_q3_policy):
                return RoundPolicyDecision(side=side, policy=simple_rule_policy, reason="高故障/高跨局惩罚，激进型策略降级为 simple_rule")
            return RoundPolicyDecision(side=side, policy=protective_q3_policy, reason="高故障/高跨局惩罚，进入保护型策略")

        if trailing and default_policy is simple_rule_policy:
            return RoundPolicyDecision(side=side, policy=greedy_q3_policy, reason="系列赛落后，将 simple_rule 提升为 greedy")

        if trailing and default_policy is protective_q3_policy:
            return RoundPolicyDecision(side=side, policy=momentum_q3_policy, reason="系列赛落后，将 protective 提升为 momentum")

        if leading and default_policy is greedy_q3_policy and (
            me.carry_stability_debt >= self.config.greedy_to_simple_when_leading_stability_threshold
            or me.carry_energy_debt >= self.config.greedy_to_simple_when_leading_energy_threshold
        ):
            return RoundPolicyDecision(side=side, policy=simple_rule_policy, reason="系列赛领先且损耗偏高，greedy 降级为 simple_rule")

        if leading and default_policy is momentum_q3_policy and (
            me.carry_stability_debt >= self.config.greedy_to_simple_when_leading_stability_threshold
            or me.carry_energy_debt >= self.config.greedy_to_simple_when_leading_energy_threshold
        ):
            return RoundPolicyDecision(side=side, policy=protective_q3_policy, reason="系列赛领先且损耗偏高，momentum 降级为 protective")

        return RoundPolicyDecision(side=side, policy=default_policy, reason="保持默认单局策略")

    def _make_round_cache_key(self, bo3_state: BO3State, my_default_policy: PolicyFn, opp_default_policy: PolicyFn) -> tuple:
        return (
            bo3_state.current_round_index,
            bo3_state.my.round_wins,
            bo3_state.opp.round_wins,
            bo3_state.my.fault_level.value,
            bo3_state.opp.fault_level.value,
            round(bo3_state.my.carry_hp_debt, 1),
            round(bo3_state.my.carry_stability_debt, 1),
            round(bo3_state.my.carry_energy_debt, 1),
            round(bo3_state.opp.carry_hp_debt, 1),
            round(bo3_state.opp.carry_stability_debt, 1),
            round(bo3_state.opp.carry_energy_debt, 1),
            get_policy_name(my_default_policy),
            get_policy_name(opp_default_policy),
        )

    def decide_round_policies(self, bo3_state: BO3State, my_default_policy: PolicyFn, opp_default_policy: PolicyFn) -> List[RoundPolicyDecision]:
        cache_key = self._make_round_cache_key(bo3_state, my_default_policy, opp_default_policy)
        if cache_key in self._round_policy_cache:
            return self._round_policy_cache[cache_key]

        result = [
            self.select_round_policy(bo3_state, "my", my_default_policy),
            self.select_round_policy(bo3_state, "opp", opp_default_policy),
        ]
        self._round_policy_cache[cache_key] = result
        return result

    def should_reset_in_round(self, bo3_state: BO3State, state: RoundState, side: str) -> Optional[str]:
        fighter = self._get_round_fighter(state, side)
        side_state = self._get_side_state(bo3_state, side)

        if state.is_finished():
            return None
        if side_state.resets_left <= 0:
            return None

        if fighter.posture == PostureState.DOWNED:
            return "倒地后触发人工复位"
        return None

    def should_timeout_in_round(self, bo3_state: BO3State, state: RoundState, side: str) -> Optional[str]:
        fighter = self._get_round_fighter(state, side)
        side_state = self._get_side_state(bo3_state, side)

        if state.is_finished():
            return None
        if side_state.timeouts_left <= 0:
            return None

        if fighter.stability <= self.config.timeout_stability_threshold:
            return "稳定性过低，触发局内暂停"
        if fighter.energy <= self.config.timeout_energy_threshold:
            return "能量过低，触发局内暂停"
        if fighter.posture == PostureState.OFF_BALANCE and fighter.stability <= self.config.timeout_off_balance_stability_threshold:
            return "失衡且稳定性偏低，触发局内暂停"

        return None

    def decide_in_round_actions(self, bo3_state: BO3State, state: RoundState) -> List[ResourceDecision]:
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

    def decide_bundle(self, bo3_state: BO3State, my_default_policy: PolicyFn, opp_default_policy: PolicyFn, current_round_state: Optional[RoundState] = None) -> ResourcePolicyDecisionBundle:
        bundle = ResourcePolicyDecisionBundle()
        bundle.pre_round_actions = self.decide_pre_round_actions(bo3_state)
        bundle.round_policy_decisions = self.decide_round_policies(bo3_state, my_default_policy, opp_default_policy)

        if current_round_state is not None:
            bundle.in_round_actions = self.decide_in_round_actions(bo3_state, current_round_state)

        return bundle


_default_resource_policy_engine = RuleBasedQ4ResourcePolicy(DEFAULT_CONFIG)


def decide_pre_round_resource_actions(bo3_state: BO3State) -> List[ResourceDecision]:
    return _default_resource_policy_engine.decide_pre_round_actions(bo3_state)


def decide_round_policies(bo3_state: BO3State, my_default_policy: PolicyFn, opp_default_policy: PolicyFn) -> List[RoundPolicyDecision]:
    return _default_resource_policy_engine.decide_round_policies(bo3_state, my_default_policy, opp_default_policy)


def decide_in_round_resource_actions(bo3_state: BO3State, current_round_state: RoundState) -> List[ResourceDecision]:
    return _default_resource_policy_engine.decide_in_round_actions(bo3_state, current_round_state)


def apply_resource_decisions_to_bo3_state(bo3_state: BO3State, decisions: List[ResourceDecision]) -> List[ResourceDecision]:
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


if __name__ == "__main__":
    from models.bo3_state import create_initial_bo3_state

    bo3 = create_initial_bo3_state()

    print("resource_policy_q4.py（规则恢复+策略扩展版）自测开始")
    print("初始比分:", bo3.scoreline())
    print("我方资源:", bo3.my.to_dict())
    print("对方资源:", bo3.opp.to_dict())

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
    print("\n本局策略决策：")
    for p in policy_decisions:
        print(p.side, p.policy_name, p.reason)

    print("\n可对比策略库：")
    for k in get_q4_compare_policy_library():
        print("-", k, ":", Q4_COMPARE_POLICY_DESCRIPTION.get(k, ""))

    print("\nresource_policy_q4.py（规则恢复+策略扩展版）自测完成")
