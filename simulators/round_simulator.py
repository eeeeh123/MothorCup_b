from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

from config.rules import get_modeling_rule_subset
from models.state import (
    RoundState,
    DistanceState,
    InitiativeState,
    PostureState,
    create_initial_round_state,
)
from models.transition import (
    TransitionEngine,
    ActionDecision,
    TransitionResult,
)

from optimizers.policy_q3 import greedy_q3_policy

RULE = get_modeling_rule_subset()

# 策略函数接口：
# 输入：当前状态、side("my"/"opp")、随机数生成器
# 输出：ActionDecision
PolicyFn = Callable[[RoundState, str, random.Random], ActionDecision]

def get_policy_name(policy: PolicyFn) -> str:
    """
    返回策略函数名，便于打印。
    """
    if hasattr(policy, "__name__"):
        return policy.__name__
    return policy.__class__.__name__

# =========================================================
# 一、仿真输出数据结构
# =========================================================
@dataclass
class StepRecord:
    step_index: int
    time_left_sec: float
    phase: str

    my_decision_type: str
    my_action_key: Optional[str]

    opp_decision_type: str
    opp_action_key: Optional[str]

    reward_my: float
    reward_opp: float

    my_hp_proxy: float
    my_stability: float
    my_energy: float
    my_posture: str

    opp_hp_proxy: float
    opp_stability: float
    opp_energy: float
    opp_posture: str

    distance: str
    initiative: str
    info_case: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoundSimulationResult:
    final_state: RoundState
    winner: Optional[str]
    win_reason: Optional[str]
    total_steps: int
    total_reward_my: float
    total_reward_opp: float
    step_records: List[StepRecord]
    event_log: List[str]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "win_reason": self.win_reason,
            "total_steps": self.total_steps,
            "total_reward_my": self.total_reward_my,
            "total_reward_opp": self.total_reward_opp,
            "final_time_left_sec": self.final_state.time_left_sec,
            "final_phase": self.final_state.phase.value,
            "my_hp_proxy": self.final_state.my.hp_proxy,
            "my_stability": self.final_state.my.stability,
            "my_energy": self.final_state.my.energy,
            "opp_hp_proxy": self.final_state.opp.hp_proxy,
            "opp_stability": self.final_state.opp.stability,
            "opp_energy": self.final_state.opp.energy,
        }


# =========================================================
# 二、默认策略：随机策略
# =========================================================
def random_policy(
    state: RoundState,
    side: str,
    rng: random.Random,
) -> ActionDecision:
    """
    最简单的随机策略。
    用于验证仿真链条是否通。
    """
    # 这批动作 key 基于当前已写好的项目
    attack_keys = ["A01", "A02", "A03", "A05", "A06", "A08", "A09", "A12"]
    defend_keys = ["D01", "D02", "D04", "D08", "D09", "D12", "D14", "D15", "D22"]

    fighter = state.my if side == "my" else state.opp

    # 若正在恢复/倒地，优先 recover
    if fighter.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
        return ActionDecision(action_type="recover", action_key=None)

    # 能量太低则保守
    if fighter.energy < 20:
        return rng.choice(
            [
                ActionDecision("defend", rng.choice(defend_keys)),
                ActionDecision("recover", None),
                ActionDecision("hold", None),
            ]
        )

    pool = [
        ActionDecision("attack", rng.choice(attack_keys)),
        ActionDecision("defend", rng.choice(defend_keys)),
        ActionDecision("hold", None),
    ]
    return rng.choice(pool)


# =========================================================
# 三、默认策略：简单规则策略
# =========================================================
def simple_rule_policy(
    state: RoundState,
    side: str,
    rng: random.Random,
) -> ActionDecision:
    """
    一个比随机策略更合理的第一版规则策略。
    目标不是最优，而是让 Q3 更像比赛过程。
    """
    me = state.my if side == "my" else state.opp
    opp = state.opp if side == "my" else state.my

    # 1. 倒地或恢复中：优先 recover
    if me.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
        return ActionDecision("recover", None)

    # 2. 自身稳定性太低：优先防守或恢复
    if me.stability < 30:
        return rng.choice([
            ActionDecision("defend", "D14"),  # 重心补偿
            ActionDecision("defend", "D15"),  # 卸力缓冲
            ActionDecision("recover", None),
        ])

    # 3. 能量过低：保守
    if me.energy < 18:
        return rng.choice([
            ActionDecision("defend", "D11"),  # 护头防御
            ActionDecision("defend", "D22"),  # 挡→撤→绕
            ActionDecision("hold", None),
        ])

    # 4. 若对方失衡，优先追击
    if opp.posture == PostureState.OFF_BALANCE:
        if state.distance in [DistanceState.NEAR, DistanceState.CLINCH]:
            return rng.choice([
                ActionDecision("attack", "A09"),  # 膝撞
                ActionDecision("attack", "A12"),  # 冲撞
                ActionDecision("attack", "A03"),  # 组合拳
            ])
        return rng.choice([
            ActionDecision("attack", "A06"),  # 侧踢
            ActionDecision("attack", "A05"),  # 前踢
            ActionDecision("attack", "A08"),  # 低扫腿
        ])

    # 5. 根据距离选动作
    if state.distance == DistanceState.FAR:
        return rng.choice([
            ActionDecision("attack", "A05"),  # 前踢
            ActionDecision("attack", "A06"),  # 侧踢
            ActionDecision("hold", None),
        ])

    if state.distance == DistanceState.MID:
        return rng.choice([
            ActionDecision("attack", "A05"),
            ActionDecision("attack", "A06"),
            ActionDecision("attack", "A08"),
            ActionDecision("defend", "D09"),  # 转身闪避
        ])

    if state.distance == DistanceState.NEAR:
        return rng.choice([
            ActionDecision("attack", "A01"),  # 直拳
            ActionDecision("attack", "A03"),  # 组合拳
            ActionDecision("attack", "A09"),  # 膝撞
            ActionDecision("defend", "D03"),  # 肘挡
        ])

    # clinch
    return rng.choice([
        ActionDecision("attack", "A09"),
        ActionDecision("attack", "A12"),
        ActionDecision("defend", "D12"),      # 沉身防御
        ActionDecision("defend", "D15"),      # 卸力缓冲
    ])


# =========================================================
# 四、距离更新（简单版）
# =========================================================
def update_distance_simple(
    state: RoundState,
    my_decision: ActionDecision,
    opp_decision: ActionDecision,
    rng: random.Random,
) -> None:
    """
    第一版简单距离更新规则。
    后续可在 transition.py 或独立 motion 模块中细化。
    """
    current = state.distance

    # 若双方都 hold / defend，则距离更可能拉开一点
    if my_decision.action_type in ["hold", "defend"] and opp_decision.action_type in ["hold", "defend"]:
        if current == DistanceState.CLINCH:
            state.set_distance(DistanceState.NEAR)
        elif current == DistanceState.NEAR:
            state.set_distance(DistanceState.MID)
        return

    # 若出现冲撞/膝撞等近身动作，更可能进入近距离或缠斗
    close_attack_keys = {"A09", "A12"}  # 膝撞、冲撞
    if my_decision.action_key in close_attack_keys or opp_decision.action_key in close_attack_keys:
        if current == DistanceState.FAR:
            state.set_distance(DistanceState.MID)
        elif current == DistanceState.MID:
            state.set_distance(DistanceState.NEAR)
        else:
            state.set_distance(DistanceState.CLINCH if rng.random() < 0.35 else DistanceState.NEAR)
        return

    # 若双方都在中远距离用腿法，更可能保持 mid
    mid_attack_keys = {"A05", "A06", "A08"}  # 前踢、侧踢、低扫腿
    if my_decision.action_key in mid_attack_keys or opp_decision.action_key in mid_attack_keys:
        if current == DistanceState.CLINCH:
            state.set_distance(DistanceState.NEAR)
        else:
            state.set_distance(DistanceState.MID)
        return

    # 否则做一个温和随机游走
    transitions = {
        DistanceState.FAR: [DistanceState.FAR, DistanceState.MID],
        DistanceState.MID: [DistanceState.FAR, DistanceState.MID, DistanceState.NEAR],
        DistanceState.NEAR: [DistanceState.MID, DistanceState.NEAR, DistanceState.CLINCH],
        DistanceState.CLINCH: [DistanceState.NEAR, DistanceState.CLINCH],
    }
    state.set_distance(rng.choice(transitions[current]))


# =========================================================
# 五、单回合仿真
# =========================================================
def simulate_round(
    my_policy: PolicyFn = simple_rule_policy,
    opp_policy: PolicyFn = simple_rule_policy,
    mode: str = "expected",         # expected / sample
    seed: Optional[int] = None,
    max_steps: int = 500,
    initial_state: Optional[RoundState] = None,
) -> RoundSimulationResult:
    """
    模拟一个完整单回合。

    参数：
    - my_policy / opp_policy: 双方策略函数
    - mode:
        - expected: 期望模式，适合先看整体趋势
        - sample: 采样模式，适合蒙特卡洛
    - max_steps: 防止死循环；若达到上限仍未结束，则启用扩展 tie-break 兜底
    """
    rng = random.Random(seed)
    engine = TransitionEngine()

    state = initial_state.clone() if initial_state is not None else create_initial_round_state()

    total_reward_my = 0.0
    total_reward_opp = 0.0
    step_records: List[StepRecord] = []

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

        # 简单距离更新（基于动作倾向）
        if not next_state.is_finished():
            update_distance_simple(next_state, my_decision, opp_decision, rng)

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
        state.log("达到 max_steps，触发仿真兜底扩展 tie-break。")
        state.finish_by_extended_tiebreak()

    return RoundSimulationResult(
        final_state=state,
        winner=state.winner,
        win_reason=state.win_reason,
        total_steps=state.step_index,
        total_reward_my=total_reward_my,
        total_reward_opp=total_reward_opp,
        step_records=step_records,
        event_log=state.event_log,
    )


# =========================================================
# 六、多次仿真（蒙特卡洛入口）
# =========================================================
def simulate_many_rounds(
    n_runs: int = 100,
    my_policy: PolicyFn = simple_rule_policy,
    opp_policy: PolicyFn = simple_rule_policy,
    mode: str = "sample",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    多次重复模拟，用于粗略胜率统计。
    """
    rng = random.Random(seed)

    win_count_my = 0
    win_count_opp = 0
    draw_count = 0

    reasons: Dict[str, int] = {}
    reward_my_list: List[float] = []
    reward_opp_list: List[float] = []
    step_count_list: List[int] = []

    for _ in range(n_runs):
        result = simulate_round(
            my_policy=my_policy,
            opp_policy=opp_policy,
            mode=mode,
            seed=rng.randint(1, 10**9),
        )

        if result.winner == "my":
            win_count_my += 1
        elif result.winner == "opp":
            win_count_opp += 1
        else:
            draw_count += 1

        reason = result.win_reason or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1

        reward_my_list.append(result.total_reward_my)
        reward_opp_list.append(result.total_reward_opp)
        step_count_list.append(result.total_steps)

    def _avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_runs": n_runs,
        "my_win_rate": win_count_my / n_runs if n_runs else 0.0,
        "opp_win_rate": win_count_opp / n_runs if n_runs else 0.0,
        "draw_rate": draw_count / n_runs if n_runs else 0.0,
        "win_reason_distribution": reasons,
        "avg_reward_my": _avg(reward_my_list),
        "avg_reward_opp": _avg(reward_opp_list),
        "avg_step_count": _avg(step_count_list),
    }


# =========================================================
# 七、自测
# =========================================================
if __name__ == "__main__":
    print("round_simulator.py 自测开始")

    my_policy_for_test = greedy_q3_policy
    opp_policy_for_test = greedy_q3_policy

    # 1) 单局 expected 模式
    single = simulate_round(
        my_policy=my_policy_for_test,
        opp_policy=opp_policy_for_test,
        mode="expected",
        seed=42,
    )

    print("\n[单局 expected 模式]")
    print("my policy:", get_policy_name(my_policy_for_test))
    print("opp policy:", get_policy_name(opp_policy_for_test))
    print("winner:", single.winner)
    print("reason:", single.win_reason)
    print("steps:", single.total_steps)
    print("my total reward:", round(single.total_reward_my, 3))
    print("opp total reward:", round(single.total_reward_opp, 3))
    print("final my hp/stability/energy:",
          round(single.final_state.my.hp_proxy, 3),
          round(single.final_state.my.stability, 3),
          round(single.final_state.my.energy, 3))
    print("final opp hp/stability/energy:",
          round(single.final_state.opp.hp_proxy, 3),
          round(single.final_state.opp.stability, 3),
          round(single.final_state.opp.energy, 3))

    # 2) 多局 sample 模式
    many = simulate_many_rounds(
        n_runs=200,
        my_policy=my_policy_for_test,
        opp_policy=opp_policy_for_test,
        mode="sample",
        seed=123,
    )

    print("\n[多局 sample 模式]")
    print("my policy:", get_policy_name(my_policy_for_test))
    print("opp policy:", get_policy_name(opp_policy_for_test))
    print("my win rate:", round(many["my_win_rate"], 3))
    print("opp win rate:", round(many["opp_win_rate"], 3))
    print("draw rate:", round(many["draw_rate"], 3))
    print("avg reward my:", round(many["avg_reward_my"], 3))
    print("avg reward opp:", round(many["avg_reward_opp"], 3))
    print("avg step count:", round(many["avg_step_count"], 3))
    print("win reasons:", many["win_reason_distribution"])

    print("\nround_simulator.py 自测完成")