from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, TYPE_CHECKING

from models.bo3_state import BO3State, FaultLevel, ResourceType
from models.state import RoundState, PostureState
from simulators.round_simulator import (
    PolicyFn,
    simple_rule_policy,
    random_policy,
    get_policy_name,
)
from optimizers.policy_q3 import greedy_q3_policy


if TYPE_CHECKING:
    from simulators.bo3_simulator import ResourceUsageRecord


# =========================================================
# 一、数据结构
# =========================================================
@dataclass
class ResourceDecision:
    """
    单个资源动作决策。

    resource_type:
        - RESET
        - TIMEOUT
        - REPAIR
    """
    side: str
    resource_type: ResourceType
    phase: str                  # pre_round / in_round
    reason: str


@dataclass
class RoundPolicyDecision:
    """
    当前局的单局策略选择结果。
    """
    side: str
    policy: PolicyFn
    reason: str

    @property
    def policy_name(self) -> str:
        return get_policy_name(self.policy)


@dataclass
class ResourcePolicyDecisionBundle:
    """
    一次完整资源策略评估的输出：
    - pre_round 决策
    - in_round 决策
    - 当前局单局策略
    """
    pre_round_actions: List[ResourceDecision] = field(default_factory=list)
    in_round_actions: List[ResourceDecision] = field(default_factory=list)
    round_policy_decisions: List[RoundPolicyDecision] = field(default_factory=list)


@dataclass
class Q4ResourcePolicyConfig:
    """
    第一版规则型 Q4 资源策略配置。
    """
    # 赛前维修阈值
    repair_fault_levels: tuple[FaultLevel, ...] = (FaultLevel.MAJOR, FaultLevel.CRITICAL)
    repair_carry_stability_threshold: float = 18.0
    repair_carry_hp_threshold: float = 15.0
    repair_carry_energy_threshold: float = 22.0

    # 局内暂停阈值
    timeout_stability_threshold: float = 18.0
    timeout_energy_threshold: float = 15.0
    timeout_off_balance_stability_threshold: float = 25.0

    # 策略切换阈值
    heavy_damage_fault_levels: tuple[FaultLevel, ...] = (FaultLevel.MAJOR, FaultLevel.CRITICAL)
    heavy_damage_carry_stability_threshold: float = 16.0
    heavy_damage_carry_hp_threshold: float = 12.0
    heavy_damage_carry_energy_threshold: float = 20.0

    greedy_to_simple_when_leading_stability_threshold: float = 10.0
    greedy_to_simple_when_leading_energy_threshold: float = 14.0


DEFAULT_CONFIG = Q4ResourcePolicyConfig()


# =========================================================
# 二、规则型资源策略引擎
# =========================================================
class RuleBasedQ4ResourcePolicy:
    """
    Q4 第一版资源策略引擎。

    设计目标：
    1. 不重写 Q3 单局动作策略；
    2. 只负责 BO3 外层资源决策；
    3. 可被 bo3_simulator.py 直接调用。
    """

    def __init__(self, config: Q4ResourcePolicyConfig = DEFAULT_CONFIG) -> None:
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

    # -----------------------------------------------------
    # 局前资源决策
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 当前局单局策略选择
    # -----------------------------------------------------
    def select_round_policy(
        self,
        bo3_state: BO3State,
        side: str,
        default_policy: PolicyFn,
    ) -> RoundPolicyDecision:
        me = self._get_side_state(bo3_state, side)
        opp = self._get_other_side_state(bo3_state, side)

        # random 保持随机，不做升级/降级
        if default_policy is random_policy:
            return RoundPolicyDecision(
                side=side,
                policy=random_policy,
                reason="基准随机策略，不做外层改写",
            )

        heavy_damage = (
            me.fault_level in self.config.heavy_damage_fault_levels
            or me.carry_stability_debt >= self.config.heavy_damage_carry_stability_threshold
            or me.carry_hp_debt >= self.config.heavy_damage_carry_hp_threshold
            or me.carry_energy_debt >= self.config.heavy_damage_carry_energy_threshold
        )

        trailing = me.round_wins < opp.round_wins
        leading = me.round_wins > opp.round_wins

        if heavy_damage:
            return RoundPolicyDecision(
                side=side,
                policy=simple_rule_policy,
                reason="高故障/高跨局惩罚，切换为保守策略",
            )

        if trailing and default_policy is simple_rule_policy:
            return RoundPolicyDecision(
                side=side,
                policy=greedy_q3_policy,
                reason="系列赛落后，将 simple_rule 提升为 greedy",
            )

        if leading and default_policy is greedy_q3_policy and (
            me.carry_stability_debt >= self.config.greedy_to_simple_when_leading_stability_threshold
            or me.carry_energy_debt >= self.config.greedy_to_simple_when_leading_energy_threshold
        ):
            return RoundPolicyDecision(
                side=side,
                policy=simple_rule_policy,
                reason="系列赛领先且损耗偏高，greedy 降级为保守策略",
            )

        return RoundPolicyDecision(
            side=side,
            policy=default_policy,
            reason="保持默认单局策略",
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
    # 局内资源决策
    # -----------------------------------------------------
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
                # 已倒地时优先 reset；通常不再同一步叠加 timeout
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
# 三、对外暴露的默认策略对象与包装函数
# =========================================================
_default_resource_policy_engine = RuleBasedQ4ResourcePolicy(DEFAULT_CONFIG)


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
# 四、可选：把决策真正执行到 BO3State 上
# =========================================================
def apply_resource_decisions_to_bo3_state(
    bo3_state: BO3State,
    decisions: List[ResourceDecision],
) -> List[ResourceDecision]:
    """
    将资源决策执行到 BO3State 上。
    返回真正成功执行的动作列表。
    """
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
# 五、自测
# =========================================================
if __name__ == "__main__":
    from models.bo3_state import create_initial_bo3_state

    bo3 = create_initial_bo3_state()

    print("resource_policy_q4.py 自测开始")
    print("初始比分:", bo3.scoreline())
    print("我方资源:", bo3.my.to_dict())
    print("对方资源:", bo3.opp.to_dict())

    # 模拟跨局损伤
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

    print("\nresource_policy_q4.py 自测完成")
