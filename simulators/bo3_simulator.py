from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from models.bo3_state import (
    BO3State,
    create_initial_bo3_state,
)
from models.state import (
    RoundState,
)
from models.transition import (
    TransitionEngine,
)
from simulators.round_simulator import (
    PolicyFn,
    StepRecord,
    RoundSimulationResult,
    update_distance_simple,
    get_policy_name,
    simple_rule_policy,
    random_policy,
)
from optimizers.policy_q3 import greedy_q3_policy
from optimizers.resource_policy_q4 import (
    ResourceDecision,
    RoundPolicyDecision,
    decide_pre_round_resource_actions,
    decide_round_policies,
    decide_in_round_resource_actions,
    apply_resource_decisions_to_bo3_state,
)


# =========================================================
# 一、数据结构
# =========================================================
@dataclass
class ResourceUsageRecord:
    round_index: int
    step_index: int
    phase: str                 # pre_round / in_round
    side: str                  # my / opp
    resource_type: str         # reset / timeout / repair
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BO3RoundSummary:
    round_index: int
    my_policy: str
    opp_policy: str
    winner: Optional[str]
    win_reason: Optional[str]
    total_steps: int
    total_reward_my: float
    total_reward_opp: float
    my_hp_proxy: float
    my_stability: float
    my_energy: float
    opp_hp_proxy: float
    opp_stability: float
    opp_energy: float
    scoreline_after_round: str
    resource_usage: List[ResourceUsageRecord]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["resource_usage_count"] = len(self.resource_usage)
        data["resource_usage"] = [item.to_dict() for item in self.resource_usage]
        return data


@dataclass
class BO3SimulationResult:
    final_bo3_state: BO3State
    winner: Optional[str]
    win_reason: Optional[str]
    rounds_played: int
    round_summaries: List[BO3RoundSummary]
    all_resource_usage: List[ResourceUsageRecord]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "win_reason": self.win_reason,
            "rounds_played": self.rounds_played,
            "final_scoreline": self.final_bo3_state.scoreline(),
            "my_round_wins": self.final_bo3_state.my.round_wins,
            "opp_round_wins": self.final_bo3_state.opp.round_wins,
            "my_resets_left": self.final_bo3_state.my.resets_left,
            "my_timeouts_left": self.final_bo3_state.my.timeouts_left,
            "my_repairs_left": self.final_bo3_state.my.repairs_left,
            "opp_resets_left": self.final_bo3_state.opp.resets_left,
            "opp_timeouts_left": self.final_bo3_state.opp.timeouts_left,
            "opp_repairs_left": self.final_bo3_state.opp.repairs_left,
            "my_fault_level": self.final_bo3_state.my.fault_level.value,
            "opp_fault_level": self.final_bo3_state.opp.fault_level.value,
            "resource_usage_count": len(self.all_resource_usage),
        }


# =========================================================
# 二、内部工具函数
# =========================================================
def _record_applied_decisions(
    usage_list: List[ResourceUsageRecord],
    bo3_state: BO3State,
    step_index: int,
    decisions: List[ResourceDecision],
) -> None:
    for item in decisions:
        usage_list.append(
            ResourceUsageRecord(
                round_index=bo3_state.current_round_index,
                step_index=step_index,
                phase=item.phase,
                side=item.side,
                resource_type=item.resource_type.value,
                reason=item.reason,
            )
        )


def _select_policy_from_decisions(
    decisions: List[RoundPolicyDecision],
    side: str,
    fallback_policy: PolicyFn,
) -> PolicyFn:
    for item in decisions:
        if item.side == side:
            return item.policy
    return fallback_policy


# =========================================================
# 三、在 BO3 上下文中模拟单局
# =========================================================
def simulate_round_in_bo3_context(
    bo3_state: BO3State,
    my_policy: PolicyFn,
    opp_policy: PolicyFn,
    mode: str = "sample",
    seed: Optional[int] = None,
    max_steps: int = 500,
) -> Tuple[RoundSimulationResult, List[ResourceUsageRecord]]:
    """
    在 BO3 上下文中跑一局：
    - 起点来自 bo3_state.current_round_state
    - 中间资源动作由 resource_policy_q4.py 决策并执行
    - 结束后返回 RoundSimulationResult 与资源使用记录
    """
    if bo3_state.current_round_state is None:
        raise RuntimeError("当前没有正在进行的回合，请先调用 start_next_round().")

    rng = random.Random(seed)
    engine = TransitionEngine()

    state = bo3_state.current_round_state.clone()
    total_reward_my = 0.0
    total_reward_opp = 0.0
    step_records: List[StepRecord] = []
    resource_usage: List[ResourceUsageRecord] = []

    while not state.is_finished() and state.step_index < max_steps:
        my_decision = my_policy(state.clone(), "my", rng)
        opp_decision = opp_policy(state.clone(), "opp", rng)

        result = engine.step(
            state=state,
            my_decision=my_decision,
            opp_decision=opp_decision,
            mode=mode,
            seed=rng.randint(1, 10**9),
        )

        next_state = result.next_state

        if not next_state.is_finished():
            update_distance_simple(next_state, my_decision, opp_decision, rng)

        # 局内资源策略：决策 -> 执行 -> 回写状态
        if not next_state.is_finished():
            bo3_state.current_round_state = next_state.clone()
            decisions = decide_in_round_resource_actions(
                bo3_state=bo3_state,
                current_round_state=bo3_state.current_round_state,
            )
            applied = apply_resource_decisions_to_bo3_state(bo3_state, decisions)
            _record_applied_decisions(
                usage_list=resource_usage,
                bo3_state=bo3_state,
                step_index=bo3_state.current_round_state.step_index,
                decisions=applied,
            )
            next_state = bo3_state.current_round_state.clone()

        step_record = StepRecord(
            step_index=next_state.step_index,
            time_left_sec=next_state.time_left_sec,
            phase=next_state.phase.value,
            my_decision_type=my_decision.action_type,
            my_action_key=my_decision.action_key,
            opp_decision_type=opp_decision.action_type,
            opp_action_key=opp_decision.action_key,
            reward_my=result.reward_my,
            reward_opp=result.reward_opp,
            my_hp_proxy=next_state.my.hp_proxy,
            my_stability=next_state.my.stability,
            my_energy=next_state.my.energy,
            my_posture=next_state.my.posture.value,
            opp_hp_proxy=next_state.opp.hp_proxy,
            opp_stability=next_state.opp.stability,
            opp_energy=next_state.opp.energy,
            opp_posture=next_state.opp.posture.value,
            distance=next_state.distance.value,
            initiative=next_state.initiative.value,
            info_case=result.info.get("case"),
        )
        step_records.append(step_record)

        total_reward_my += result.reward_my
        total_reward_opp += result.reward_opp
        state = next_state

    if not state.is_finished():
        state.finish_by_score()

    bo3_state.current_round_state = state.clone()

    sim_result = RoundSimulationResult(
        final_state=state,
        winner=state.winner,
        win_reason=state.win_reason,
        total_steps=state.step_index,
        total_reward_my=total_reward_my,
        total_reward_opp=total_reward_opp,
        step_records=step_records,
        event_log=state.event_log,
    )
    return sim_result, resource_usage


# =========================================================
# 四、单次 BO3 仿真
# =========================================================
def simulate_bo3(
    my_base_policy: PolicyFn = greedy_q3_policy,
    opp_base_policy: PolicyFn = greedy_q3_policy,
    mode: str = "sample",
    seed: Optional[int] = None,
    max_steps_per_round: int = 500,
    initial_bo3_state: Optional[BO3State] = None,
) -> BO3SimulationResult:
    rng = random.Random(seed)
    bo3_state = initial_bo3_state.clone() if initial_bo3_state is not None else create_initial_bo3_state()

    round_summaries: List[BO3RoundSummary] = []
    all_resource_usage: List[ResourceUsageRecord] = []

    while not bo3_state.is_finished():
        bo3_state.start_next_round()

        # 1) 局前资源决策
        pre_round_decisions = decide_pre_round_resource_actions(bo3_state)
        applied_pre_round = apply_resource_decisions_to_bo3_state(bo3_state, pre_round_decisions)

        pre_round_usage: List[ResourceUsageRecord] = []
        _record_applied_decisions(
            usage_list=pre_round_usage,
            bo3_state=bo3_state,
            step_index=0,
            decisions=applied_pre_round,
        )

        # 2) 本局策略决策
        round_policy_decisions = decide_round_policies(
            bo3_state=bo3_state,
            my_default_policy=my_base_policy,
            opp_default_policy=opp_base_policy,
        )
        my_policy = _select_policy_from_decisions(round_policy_decisions, "my", my_base_policy)
        opp_policy = _select_policy_from_decisions(round_policy_decisions, "opp", opp_base_policy)

        # 3) 在 BO3 上下文中跑一局
        sim_result, in_round_usage = simulate_round_in_bo3_context(
            bo3_state=bo3_state,
            my_policy=my_policy,
            opp_policy=opp_policy,
            mode=mode,
            seed=rng.randint(1, 10**9),
            max_steps=max_steps_per_round,
        )

        # 4) 用单局结果更新 BO3State
        bo3_state.end_round_from_sim_result(sim_result)

        resource_usage = pre_round_usage + in_round_usage
        all_resource_usage.extend(resource_usage)

        summary = BO3RoundSummary(
            round_index=bo3_state.current_round_index,
            my_policy=get_policy_name(my_policy),
            opp_policy=get_policy_name(opp_policy),
            winner=sim_result.winner,
            win_reason=sim_result.win_reason,
            total_steps=sim_result.total_steps,
            total_reward_my=sim_result.total_reward_my,
            total_reward_opp=sim_result.total_reward_opp,
            my_hp_proxy=sim_result.final_state.my.hp_proxy,
            my_stability=sim_result.final_state.my.stability,
            my_energy=sim_result.final_state.my.energy,
            opp_hp_proxy=sim_result.final_state.opp.hp_proxy,
            opp_stability=sim_result.final_state.opp.stability,
            opp_energy=sim_result.final_state.opp.energy,
            scoreline_after_round=bo3_state.scoreline(),
            resource_usage=resource_usage,
        )
        round_summaries.append(summary)

    return BO3SimulationResult(
        final_bo3_state=bo3_state,
        winner=bo3_state.winner,
        win_reason=bo3_state.win_reason,
        rounds_played=bo3_state.current_round_index,
        round_summaries=round_summaries,
        all_resource_usage=all_resource_usage,
    )


# =========================================================
# 五、多次 BO3 仿真
# =========================================================
def simulate_many_bo3(
    n_runs: int = 100,
    my_base_policy: PolicyFn = greedy_q3_policy,
    opp_base_policy: PolicyFn = greedy_q3_policy,
    mode: str = "sample",
    seed: Optional[int] = None,
    max_steps_per_round: int = 500,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    win_count_my = 0
    win_count_opp = 0
    draw_count = 0

    reasons: Dict[str, int] = {}
    rounds_played_list: List[int] = []

    resource_counts: Dict[str, int] = {
        "reset": 0,
        "timeout": 0,
        "repair": 0,
    }

    scoreline_dist: Dict[str, int] = {}

    for _ in range(n_runs):
        result = simulate_bo3(
            my_base_policy=my_base_policy,
            opp_base_policy=opp_base_policy,
            mode=mode,
            seed=rng.randint(1, 10**9),
            max_steps_per_round=max_steps_per_round,
        )

        if result.winner == "my":
            win_count_my += 1
        elif result.winner == "opp":
            win_count_opp += 1
        else:
            draw_count += 1

        reason = result.win_reason or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1

        rounds_played_list.append(result.rounds_played)

        scoreline = result.final_bo3_state.scoreline()
        scoreline_dist[scoreline] = scoreline_dist.get(scoreline, 0) + 1

        for item in result.all_resource_usage:
            key = item.resource_type
            resource_counts[key] = resource_counts.get(key, 0) + 1

    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_runs": n_runs,
        "my_series_win_rate": win_count_my / n_runs if n_runs else 0.0,
        "opp_series_win_rate": win_count_opp / n_runs if n_runs else 0.0,
        "draw_series_rate": draw_count / n_runs if n_runs else 0.0,
        "series_win_reason_distribution": reasons,
        "avg_rounds_played": _avg(rounds_played_list),
        "resource_usage_totals": resource_counts,
        "final_scoreline_distribution": scoreline_dist,
    }


# =========================================================
# 六、自测
# =========================================================
if __name__ == "__main__":
    print("bo3_simulator.py 自测开始")

    single = simulate_bo3(
        my_base_policy=greedy_q3_policy,
        opp_base_policy=simple_rule_policy,
        mode="sample",
        seed=42,
    )

    print("\n[单次 BO3]")
    print("winner:", single.winner)
    print("reason:", single.win_reason)
    print("rounds played:", single.rounds_played)
    print("final scoreline:", single.final_bo3_state.scoreline())
    print("resource usage count:", len(single.all_resource_usage))
    print("my fault:", single.final_bo3_state.my.fault_level.value)
    print("opp fault:", single.final_bo3_state.opp.fault_level.value)

    for summary in single.round_summaries:
        print(
            f"R{summary.round_index}: "
            f"{summary.my_policy} vs {summary.opp_policy} "
            f"| winner={summary.winner} "
            f"| reason={summary.win_reason} "
            f"| steps={summary.total_steps} "
            f"| scoreline={summary.scoreline_after_round} "
            f"| resources={len(summary.resource_usage)}"
        )

    many = simulate_many_bo3(
        n_runs=30,
        my_base_policy=greedy_q3_policy,
        opp_base_policy=simple_rule_policy,
        mode="sample",
        seed=123,
    )

    print("\n[多次 BO3]")
    print("my series win rate:", round(many["my_series_win_rate"], 3))
    print("opp series win rate:", round(many["opp_series_win_rate"], 3))
    print("draw series rate:", round(many["draw_series_rate"], 3))
    print("avg rounds played:", round(many["avg_rounds_played"], 3))
    print("series reasons:", many["series_win_reason_distribution"])
    print("resource usage totals:", many["resource_usage_totals"])
    print("scoreline dist:", many["final_scoreline_distribution"])

    print("\nbo3_simulator.py 自测完成")
