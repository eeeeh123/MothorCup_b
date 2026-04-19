from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from models.bo3_state import BO3State, FaultLevel, ResourceType
from models.state import RoundState, PostureState
from simulators.round_simulator import PolicyFn, simple_rule_policy, random_policy, get_policy_name
from optimizers.policy_q3 import greedy_q3_policy


@dataclass(frozen=True)
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

    enable_mc_for_round_policy: bool = True
    enable_mc_for_in_round_actions: bool = True
    mc_rollouts_round_policy: int = 6
    mc_rollouts_in_round_action: int = 4
    mc_rollout_mode: str = "sample"
    mc_rollout_max_steps_per_round: int = 300
    mc_seed_base: int = 20260419
    mc_min_improvement_round_policy: float = 0.03
    mc_min_improvement_in_round_action: float = 0.02

    score_series_win: float = 1.0
    score_series_draw: float = 0.0
    score_series_loss: float = -1.0
    score_round_margin_weight: float = 0.15
    score_resource_saving_weight: float = 0.02
    score_fault_penalty_weight: float = 0.03


DEFAULT_CONFIG = Q4ResourcePolicyConfig()


class RuleBasedQ4ResourcePolicy:
    def __init__(self, config: Q4ResourcePolicyConfig = DEFAULT_CONFIG) -> None:
        self.config = config

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
                decisions.append(ResourceDecision(side, ResourceType.REPAIR, "pre_round", reason))
        return decisions

    def select_round_policy(self, bo3_state: BO3State, side: str, default_policy: PolicyFn) -> RoundPolicyDecision:
        me = self._get_side_state(bo3_state, side)
        opp = self._get_other_side_state(bo3_state, side)

        if default_policy is random_policy:
            return RoundPolicyDecision(side, random_policy, "基准随机策略，不做外层改写")

        heavy_damage = (
            me.fault_level in self.config.heavy_damage_fault_levels
            or me.carry_stability_debt >= self.config.heavy_damage_carry_stability_threshold
            or me.carry_hp_debt >= self.config.heavy_damage_carry_hp_threshold
            or me.carry_energy_debt >= self.config.heavy_damage_carry_energy_threshold
        )
        trailing = me.round_wins < opp.round_wins
        leading = me.round_wins > opp.round_wins

        if heavy_damage:
            return RoundPolicyDecision(side, simple_rule_policy, "高故障/高跨局惩罚，切换为保守策略")
        if trailing and default_policy is simple_rule_policy:
            return RoundPolicyDecision(side, greedy_q3_policy, "系列赛落后，将 simple_rule 提升为 greedy")
        if leading and default_policy is greedy_q3_policy and (
            me.carry_stability_debt >= self.config.greedy_to_simple_when_leading_stability_threshold
            or me.carry_energy_debt >= self.config.greedy_to_simple_when_leading_energy_threshold
        ):
            return RoundPolicyDecision(side, simple_rule_policy, "系列赛领先且损耗偏高，greedy 降级为保守策略")
        return RoundPolicyDecision(side, default_policy, "保持默认单局策略")

    def decide_round_policies(self, bo3_state: BO3State, my_default_policy: PolicyFn, opp_default_policy: PolicyFn) -> List[RoundPolicyDecision]:
        return [
            self.select_round_policy(bo3_state, "my", my_default_policy),
            self.select_round_policy(bo3_state, "opp", opp_default_policy),
        ]

    def should_reset_in_round(self, bo3_state: BO3State, state: RoundState, side: str) -> Optional[str]:
        fighter = self._get_round_fighter(state, side)
        side_state = self._get_side_state(bo3_state, side)
        if state.is_finished() or side_state.resets_left <= 0:
            return None
        if fighter.posture == PostureState.DOWNED:
            return "倒地后触发人工复位"
        return None

    def should_timeout_in_round(self, bo3_state: BO3State, state: RoundState, side: str) -> Optional[str]:
        fighter = self._get_round_fighter(state, side)
        side_state = self._get_side_state(bo3_state, side)
        if state.is_finished() or side_state.timeouts_left <= 0:
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
                decisions.append(ResourceDecision(side, ResourceType.RESET, "in_round", reset_reason))
                continue
            timeout_reason = self.should_timeout_in_round(bo3_state, state, side)
            if timeout_reason is not None:
                decisions.append(ResourceDecision(side, ResourceType.TIMEOUT, "in_round", timeout_reason))
        return decisions


class MonteCarloQ4ResourcePolicy:
    def __init__(self, config: Q4ResourcePolicyConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.rule_engine = RuleBasedQ4ResourcePolicy(config)
        self._in_rollout = False
        self._current_my_policy: Optional[PolicyFn] = None
        self._current_opp_policy: Optional[PolicyFn] = None

    @staticmethod
    def _get_side_state(bo3_state: BO3State, side: str):
        return bo3_state.my if side == "my" else bo3_state.opp

    @staticmethod
    def _get_other_side_state(bo3_state: BO3State, side: str):
        return bo3_state.opp if side == "my" else bo3_state.my

    @staticmethod
    def _unique_policies(policies: List[PolicyFn]) -> List[PolicyFn]:
        out: List[PolicyFn] = []
        seen = set()
        for p in policies:
            key = get_policy_name(p)
            if key not in seen:
                seen.add(key)
                out.append(p)
        return out

    def _fault_penalty_value(self, fault_level: FaultLevel) -> float:
        if fault_level == FaultLevel.HEALTHY:
            return 0.0
        if fault_level == FaultLevel.MINOR:
            return 1.0
        if fault_level == FaultLevel.MAJOR:
            return 2.0
        return 3.0

    def _score_final_bo3_state(self, bo3_state: BO3State, eval_side: str) -> float:
        me = self._get_side_state(bo3_state, eval_side)
        opp = self._get_other_side_state(bo3_state, eval_side)
        if bo3_state.winner == eval_side:
            base = self.config.score_series_win
        elif bo3_state.winner == "draw" or bo3_state.winner is None:
            base = self.config.score_series_draw
        else:
            base = self.config.score_series_loss

        round_margin = me.round_wins - opp.round_wins
        resource_left = me.resets_left + me.timeouts_left + me.repairs_left
        opp_resource_left = opp.resets_left + opp.timeouts_left + opp.repairs_left
        fault_gap = self._fault_penalty_value(me.fault_level) - self._fault_penalty_value(opp.fault_level)
        return (
            base
            + self.config.score_round_margin_weight * round_margin
            + self.config.score_resource_saving_weight * (resource_left - opp_resource_left)
            - self.config.score_fault_penalty_weight * fault_gap
        )

    def _policy_candidates(self, default_policy: PolicyFn) -> List[PolicyFn]:
        if default_policy is random_policy:
            return [random_policy]
        return self._unique_policies([default_policy, greedy_q3_policy, simple_rule_policy])

    def _rollout_remaining_series(
        self,
        bo3_state: BO3State,
        my_current_policy: PolicyFn,
        opp_current_policy: PolicyFn,
        my_base_policy: PolicyFn,
        opp_base_policy: PolicyFn,
        eval_side: str,
        seed: int,
    ) -> float:
        from simulators.bo3_simulator import simulate_round_in_bo3_context

        rng = random.Random(seed)
        cloned = bo3_state.clone()
        use_current_round_policies = cloned.current_round_state is not None and not cloned.is_finished()

        self._in_rollout = True
        try:
            while not cloned.is_finished():
                if cloned.current_round_state is None or cloned.phase.value in {"ready", "between_rounds"}:
                    cloned.start_next_round()
                    pre_actions = self.rule_engine.decide_pre_round_actions(cloned)
                    apply_resource_decisions_to_bo3_state(cloned, pre_actions)
                    my_policy = self.rule_engine.select_round_policy(cloned, "my", my_base_policy).policy
                    opp_policy = self.rule_engine.select_round_policy(cloned, "opp", opp_base_policy).policy
                else:
                    if use_current_round_policies:
                        my_policy = my_current_policy
                        opp_policy = opp_current_policy
                        use_current_round_policies = False
                    else:
                        my_policy = self.rule_engine.select_round_policy(cloned, "my", my_base_policy).policy
                        opp_policy = self.rule_engine.select_round_policy(cloned, "opp", opp_base_policy).policy

                sim_result, _ = simulate_round_in_bo3_context(
                    bo3_state=cloned,
                    my_policy=my_policy,
                    opp_policy=opp_policy,
                    mode=self.config.mc_rollout_mode,
                    seed=rng.randint(1, 10**9),
                    max_steps=self.config.mc_rollout_max_steps_per_round,
                )
                cloned.end_round_from_sim_result(sim_result)

            return self._score_final_bo3_state(cloned, eval_side)
        finally:
            self._in_rollout = False

    def decide_pre_round_actions(self, bo3_state: BO3State) -> List[ResourceDecision]:
        return self.rule_engine.decide_pre_round_actions(bo3_state)

    def _mc_select_round_policy_for_side(self, bo3_state: BO3State, side: str, default_policy: PolicyFn, other_default_policy: PolicyFn) -> RoundPolicyDecision:
        if not self.config.enable_mc_for_round_policy or self._in_rollout:
            return self.rule_engine.select_round_policy(bo3_state, side, default_policy)

        if default_policy is random_policy:
            return RoundPolicyDecision(side, random_policy, "基准随机策略，不做外层改写")

        baseline = self.rule_engine.select_round_policy(bo3_state, side, default_policy)
        candidates = self._policy_candidates(default_policy)
        if len(candidates) == 1:
            return baseline

        rng = random.Random(self.config.mc_seed_base + 1009 * bo3_state.current_round_index + (17 if side == "my" else 29))
        values = {}
        for policy in candidates:
            total = 0.0
            for _ in range(self.config.mc_rollouts_round_policy):
                cloned = bo3_state.clone()
                if side == "my":
                    total += self._rollout_remaining_series(
                        bo3_state=cloned,
                        my_current_policy=policy,
                        opp_current_policy=other_default_policy,
                        my_base_policy=policy,
                        opp_base_policy=other_default_policy,
                        eval_side=side,
                        seed=rng.randint(1, 10**9),
                    )
                else:
                    total += self._rollout_remaining_series(
                        bo3_state=cloned,
                        my_current_policy=other_default_policy,
                        opp_current_policy=policy,
                        my_base_policy=other_default_policy,
                        opp_base_policy=policy,
                        eval_side=side,
                        seed=rng.randint(1, 10**9),
                    )
            values[policy] = total / float(self.config.mc_rollouts_round_policy)

        baseline_value = values.get(baseline.policy, -10**9)
        best_policy = max(values, key=lambda p: values[p])
        best_value = values[best_policy]

        if best_policy is baseline.policy or best_value <= baseline_value + self.config.mc_min_improvement_round_policy:
            return RoundPolicyDecision(side, baseline.policy, f"Monte Carlo 评估后保持基线策略（baseline={baseline_value:.3f}, best={best_value:.3f}）")
        return RoundPolicyDecision(side, best_policy, f"Monte Carlo 评估后切换策略（baseline={baseline_value:.3f}, best={best_value:.3f}, choose={get_policy_name(best_policy)}）")

    def decide_round_policies(self, bo3_state: BO3State, my_default_policy: PolicyFn, opp_default_policy: PolicyFn) -> List[RoundPolicyDecision]:
        if self._in_rollout:
            return self.rule_engine.decide_round_policies(bo3_state, my_default_policy, opp_default_policy)

        my_decision = self._mc_select_round_policy_for_side(bo3_state, "my", my_default_policy, opp_default_policy)
        opp_decision = self._mc_select_round_policy_for_side(bo3_state, "opp", opp_default_policy, my_default_policy)
        self._current_my_policy = my_decision.policy
        self._current_opp_policy = opp_decision.policy
        return [my_decision, opp_decision]

    def _candidate_in_round_actions(self, bo3_state: BO3State, state: RoundState, side: str) -> List[Optional[ResourceDecision]]:
        candidates: List[Optional[ResourceDecision]] = [None]
        reset_reason = self.rule_engine.should_reset_in_round(bo3_state, state, side)
        if reset_reason is not None:
            candidates.append(ResourceDecision(side, ResourceType.RESET, "in_round", reset_reason))
            return candidates
        timeout_reason = self.rule_engine.should_timeout_in_round(bo3_state, state, side)
        if timeout_reason is not None:
            candidates.append(ResourceDecision(side, ResourceType.TIMEOUT, "in_round", timeout_reason))
        return candidates

    def _mc_select_in_round_action_for_side(self, bo3_state: BO3State, state: RoundState, side: str) -> Optional[ResourceDecision]:
        if not self.config.enable_mc_for_in_round_actions or self._in_rollout:
            rule_actions = self.rule_engine.decide_in_round_actions(bo3_state, state)
            for item in rule_actions:
                if item.side == side:
                    return item
            return None

        if self._current_my_policy is None or self._current_opp_policy is None:
            rule_actions = self.rule_engine.decide_in_round_actions(bo3_state, state)
            for item in rule_actions:
                if item.side == side:
                    return item
            return None

        candidates = self._candidate_in_round_actions(bo3_state, state, side)
        if len(candidates) == 1:
            return None

        rng = random.Random(self.config.mc_seed_base + 4001 * bo3_state.current_round_index + 97 * state.step_index + (7 if side == "my" else 13))
        scored_actions = []

        for action in candidates:
            total = 0.0
            for _ in range(self.config.mc_rollouts_in_round_action):
                cloned = bo3_state.clone()
                cloned.current_round_state = state.clone()

                if action is not None:
                    apply_resource_decisions_to_bo3_state(cloned, [action])

                total += self._rollout_remaining_series(
                    bo3_state=cloned,
                    my_current_policy=self._current_my_policy,
                    opp_current_policy=self._current_opp_policy,
                    my_base_policy=self._current_my_policy,
                    opp_base_policy=self._current_opp_policy,
                    eval_side=side,
                    seed=rng.randint(1, 10**9),
                )

            avg_value = total / float(self.config.mc_rollouts_in_round_action)
            scored_actions.append((action, avg_value))

        baseline_value = next((v for a, v in scored_actions if a is None), -10**9)
        best_action, best_value = max(scored_actions, key=lambda x: x[1])

        if best_action is None or best_value <= baseline_value + self.config.mc_min_improvement_in_round_action:
            return None
        return ResourceDecision(best_action.side, best_action.resource_type, best_action.phase, f"{best_action.reason}；Monte Carlo 评估优于不操作（baseline={baseline_value:.3f}, best={best_value:.3f}）")

    def decide_in_round_actions(self, bo3_state: BO3State, state: RoundState) -> List[ResourceDecision]:
        if self._in_rollout:
            return self.rule_engine.decide_in_round_actions(bo3_state, state)
        decisions: List[ResourceDecision] = []
        for side in ["my", "opp"]:
            action = self._mc_select_in_round_action_for_side(bo3_state, state, side)
            if action is not None:
                decisions.append(action)
        return decisions

    def decide_bundle(self, bo3_state: BO3State, my_default_policy: PolicyFn, opp_default_policy: PolicyFn, current_round_state: Optional[RoundState] = None) -> ResourcePolicyDecisionBundle:
        bundle = ResourcePolicyDecisionBundle()
        bundle.pre_round_actions = self.decide_pre_round_actions(bo3_state)
        bundle.round_policy_decisions = self.decide_round_policies(bo3_state, my_default_policy, opp_default_policy)
        if current_round_state is not None:
            bundle.in_round_actions = self.decide_in_round_actions(bo3_state, current_round_state)
        return bundle


_default_resource_policy_engine = MonteCarloQ4ResourcePolicy(DEFAULT_CONFIG)


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

    print("resource_policy_q4.py（Monte Carlo 版）自测开始")
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

    print("\nresource_policy_q4.py（Monte Carlo 版）自测完成")
