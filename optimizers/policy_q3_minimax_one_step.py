from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models.state import (
    RoundState,
    DistanceState,
    InitiativeState,
    PostureState,
)
from models.transition import (
    TransitionEngine,
    ActionDecision,
)

# =========================================================
# 复合惩罚先验（来自 action_library）
# 若上游尚未配置，则自动回退为空映射，保证脚本可运行
# =========================================================
try:
    from config.action_library import COMPOSITE_PENALTY_MAP
except Exception:
    COMPOSITE_PENALTY_MAP = {}


# =========================================================
# 一、配置
# =========================================================
@dataclass
class MinimaxOneStepConfig:
    """
    纯 minimax 一步版 Q3 配置。

    设计目标：
    1. 不再对对手动作取平均，而是取最坏对手动作（min）；
    2. 仍然只做“一步 expected 前瞻 + 启发式状态价值”；
    3. 复用当前 Q3 的候选动作池和风险先验，控制工程改动规模。
    """

    gamma: float = 0.90

    # 下一状态价值函数权重
    hp_weight: float = 1.20
    stability_weight: float = 0.90
    energy_weight: float = 0.50
    score_weight: float = 1.00
    initiative_bonus: float = 6.0

    # 姿态奖励/惩罚
    opp_off_balance_bonus: float = 8.0
    opp_downed_bonus: float = 18.0
    my_off_balance_penalty: float = 8.0
    my_downed_penalty: float = 18.0

    # 终局奖励
    terminal_win_bonus: float = 1000.0
    terminal_lose_penalty: float = 1000.0
    terminal_draw_bonus: float = 0.0

    # 动作池规模
    max_attack_candidates: int = 6
    max_defense_candidates: int = 5
    max_opp_candidates: int = 5

    # 动作先验惩罚
    action_penalty_weight: float = 6.0

    # 若多个动作的最坏情形价值很接近，则允许随机打散
    near_best_tolerance: float = 0.8


DEFAULT_CONFIG = MinimaxOneStepConfig()


# =========================================================
# 二、策略引擎
# =========================================================
class MinimaxOneStepPolicyEngine:
    """
    纯 minimax 一步版 Q3。

    与当前 greedy_q3_policy 的区别：
    - 当前 greedy：对对手动作池求平均（expect）
    - 这里：对对手动作池取最坏情形（min）

    数学形式近似为：
        V(s) ≈ max_a min_b [ r(s,a,b) + gamma * Phi(s') - penalty(a) ]
    其中 Phi(s') 为启发式状态价值函数。
    """

    def __init__(self, config: MinimaxOneStepConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.engine = TransitionEngine()

    # -----------------------------------------------------
    # 复合惩罚
    # -----------------------------------------------------
    @staticmethod
    def get_action_composite_penalty(action_key: Optional[str]) -> float:
        if not action_key:
            return 0.0
        try:
            return float(COMPOSITE_PENALTY_MAP.get(action_key, 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def normalize_composite_penalty(raw_penalty: float) -> float:
        if raw_penalty <= 0:
            return 0.0
        return max(0.0, min(1.0, (raw_penalty - 1.0) / 4.0))

    def action_penalty_value(self, action: ActionDecision) -> float:
        if action.action_type != "attack":
            return 0.0

        raw_penalty = self.get_action_composite_penalty(action.action_key)
        norm_penalty = self.normalize_composite_penalty(raw_penalty)
        return self.config.action_penalty_weight * norm_penalty

    # -----------------------------------------------------
    # 候选动作池（复用当前 Q3 风格）
    # -----------------------------------------------------
    def _attack_candidates(self, state: RoundState, side: str) -> List[str]:
        me = state.my if side == "my" else state.opp

        if me.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
            return []

        if state.distance == DistanceState.FAR:
            pool = ["A05", "A06"]  # 前踢、侧踢
        elif state.distance == DistanceState.MID:
            pool = ["A05", "A06", "A08", "A10", "A11"]  # 前踢、侧踢、低扫腿、拳腿组合、五连踢
        elif state.distance == DistanceState.NEAR:
            pool = ["A01", "A02", "A03", "A08", "A09", "A12"]  # 直拳、勾拳、组合拳、低扫腿、膝撞、冲撞
        else:  # clinch
            pool = ["A09", "A12", "A03"]  # 膝撞、冲撞、组合拳

        if me.energy < 25:
            pool = [a for a in pool if a not in {"A06", "A11", "A12"}]

        if me.stability < 35:
            pool = [a for a in pool if a not in {"A06", "A11", "A12"}]

        opp = state.opp if side == "my" else state.my
        if opp.posture == PostureState.OFF_BALANCE:
            bonus_pool = ["A03", "A09", "A12"]
            pool = bonus_pool + [x for x in pool if x not in bonus_pool]

        uniq: List[str] = []
        for x in pool:
            if x not in uniq:
                uniq.append(x)
        return uniq[: self.config.max_attack_candidates]

    def _defense_candidates(self, state: RoundState, side: str) -> List[str]:
        me = state.my if side == "my" else state.opp

        if me.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
            return []

        if state.distance == DistanceState.FAR:
            pool = ["D08", "D22"]  # 后撤、挡撤绕
        elif state.distance == DistanceState.MID:
            pool = ["D09", "D14", "D15", "D22", "D11"]  # 转身闪避、重心补偿、卸力缓冲、挡撤绕、护头
        elif state.distance == DistanceState.NEAR:
            pool = ["D03", "D11", "D12", "D15", "D20"]  # 肘挡、护头、沉身、卸力、闪挡反
        else:  # clinch
            pool = ["D12", "D15", "D14", "D08"]  # 沉身、卸力、重心补偿、后撤

        if me.stability < 35:
            priority = ["D14", "D15", "D12"]
            pool = priority + [x for x in pool if x not in priority]

        uniq: List[str] = []
        for x in pool:
            if x not in uniq:
                uniq.append(x)
        return uniq[: self.config.max_defense_candidates]

    def build_candidate_actions(self, state: RoundState, side: str) -> List[ActionDecision]:
        me = state.my if side == "my" else state.opp

        if me.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
            return [ActionDecision("recover", None)]

        candidates: List[ActionDecision] = []

        for atk in self._attack_candidates(state, side):
            candidates.append(ActionDecision("attack", atk))

        for dfd in self._defense_candidates(state, side):
            candidates.append(ActionDecision("defend", dfd))

        candidates.append(ActionDecision("hold", None))
        if me.energy < 30 or me.stability < 40:
            candidates.append(ActionDecision("recover", None))

        uniq: List[ActionDecision] = []
        seen = set()
        for c in candidates:
            key = (c.action_type, c.action_key)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        return uniq

    def build_opponent_pool(self, state: RoundState, side: str) -> List[ActionDecision]:
        other = "opp" if side == "my" else "my"
        pool = self.build_candidate_actions(state, other)
        return pool[: self.config.max_opp_candidates]

    # -----------------------------------------------------
    # 启发式状态价值（复用当前 Q3 风格）
    # -----------------------------------------------------
    def state_value(self, state: RoundState, side: str) -> float:
        if side == "my":
            me = state.my
            opp = state.opp
            my_score = state.my_score_proxy
            opp_score = state.opp_score_proxy
            initiative_bonus = (
                self.config.initiative_bonus
                if state.initiative == InitiativeState.MY
                else -self.config.initiative_bonus if state.initiative == InitiativeState.OPP else 0.0
            )
        else:
            me = state.opp
            opp = state.my
            my_score = state.opp_score_proxy
            opp_score = state.my_score_proxy
            initiative_bonus = (
                self.config.initiative_bonus
                if state.initiative == InitiativeState.OPP
                else -self.config.initiative_bonus if state.initiative == InitiativeState.MY else 0.0
            )

        value = 0.0
        value += self.config.hp_weight * (me.hp_proxy - opp.hp_proxy)
        value += self.config.stability_weight * (me.stability - opp.stability)
        value += self.config.energy_weight * (me.energy - opp.energy)
        value += self.config.score_weight * (my_score - opp_score)
        value += initiative_bonus

        if opp.posture == PostureState.OFF_BALANCE:
            value += self.config.opp_off_balance_bonus
        if opp.posture == PostureState.DOWNED:
            value += self.config.opp_downed_bonus
        if me.posture == PostureState.OFF_BALANCE:
            value -= self.config.my_off_balance_penalty
        if me.posture == PostureState.DOWNED:
            value -= self.config.my_downed_penalty

        if state.is_finished():
            if state.winner == side:
                value += self.config.terminal_win_bonus
            elif state.winner is None or state.winner == "draw":
                value += self.config.terminal_draw_bonus
            else:
                value -= self.config.terminal_lose_penalty

        return value

    # -----------------------------------------------------
    # 联合动作仿真
    # -----------------------------------------------------
    def _simulate_pair(
        self,
        state: RoundState,
        side: str,
        my_action: ActionDecision,
        opp_action: ActionDecision,
    ):
        if side == "my":
            return self.engine.step(
                state=state,
                my_decision=my_action,
                opp_decision=opp_action,
                mode="expected",
            )
        else:
            return self.engine.step(
                state=state,
                my_decision=opp_action,
                opp_decision=my_action,
                mode="expected",
            )

    # -----------------------------------------------------
    # 纯 minimax 一步评估
    # -----------------------------------------------------
    def evaluate_action_minimax(
        self,
        state: RoundState,
        side: str,
        action: ActionDecision,
        opp_pool: List[ActionDecision],
    ) -> Tuple[float, Optional[ActionDecision], List[Tuple[ActionDecision, float]]]:
        """
        对固定动作 a，返回：
        - 最坏情形价值 min_b Q(s,a,b)
        - 对应最坏对手动作
        - 所有对手动作下的明细列表
        """
        if not opp_pool:
            return -1e9, None, []

        action_penalty = self.action_penalty_value(action)
        q_details: List[Tuple[ActionDecision, float]] = []

        for opp_action in opp_pool:
            result = self._simulate_pair(state, side, action, opp_action)

            # 严格零和内核下，直接取本方单步收益
            immediate_reward = result.reward_my if side == "my" else result.reward_opp
            future_value = self.state_value(result.next_state, side)

            q = immediate_reward + self.config.gamma * future_value - action_penalty
            q_details.append((opp_action, q))

        worst_opp_action, worst_q = min(q_details, key=lambda x: x[1])
        return worst_q, worst_opp_action, q_details

    # -----------------------------------------------------
    # 主接口
    # -----------------------------------------------------
    def choose_action(
        self,
        state: RoundState,
        side: str,
        rng: Optional[random.Random] = None,
    ) -> ActionDecision:
        rng = rng or random.Random()

        candidates = self.build_candidate_actions(state, side)
        opp_pool = self.build_opponent_pool(state, side)

        if not candidates:
            return ActionDecision("hold", None)

        scored: List[Tuple[ActionDecision, float, Optional[ActionDecision]]] = []

        for action in candidates:
            worst_q, worst_opp_action, _ = self.evaluate_action_minimax(
                state=state,
                side=side,
                action=action,
                opp_pool=opp_pool,
            )
            scored.append((action, worst_q, worst_opp_action))

        # minimax: 选“最坏情况下仍最好”的动作
        scored.sort(key=lambda x: x[1], reverse=True)

        best_action, best_value, _ = scored[0]
        near_best = [item for item in scored if item[1] >= best_value - self.config.near_best_tolerance]

        chosen = rng.choice(near_best)[0] if len(near_best) > 1 else best_action
        return chosen

    def debug_rank_actions(
        self,
        state: RoundState,
        side: str,
    ) -> List[Dict[str, object]]:
        """
        便于自测：输出所有候选动作的 minimax 排名和最坏响应。
        """
        candidates = self.build_candidate_actions(state, side)
        opp_pool = self.build_opponent_pool(state, side)

        rows: List[Dict[str, object]] = []
        for action in candidates:
            worst_q, worst_opp_action, q_details = self.evaluate_action_minimax(
                state=state,
                side=side,
                action=action,
                opp_pool=opp_pool,
            )
            rows.append(
                {
                    "action_type": action.action_type,
                    "action_key": action.action_key,
                    "worst_q": worst_q,
                    "worst_opp_type": None if worst_opp_action is None else worst_opp_action.action_type,
                    "worst_opp_key": None if worst_opp_action is None else worst_opp_action.action_key,
                    "opp_pool_size": len(q_details),
                }
            )

        rows.sort(key=lambda x: float(x["worst_q"]), reverse=True)
        return rows


# =========================================================
# 三、对外暴露
# =========================================================
_default_minimax_one_step_engine = MinimaxOneStepPolicyEngine(DEFAULT_CONFIG)


def minimax_one_step_q3_policy(
    state: RoundState,
    side: str,
    rng: random.Random,
) -> ActionDecision:
    """
    纯 minimax 一步版 Q3 策略：
    对每个我方候选动作，取“最坏对手动作”作为评估标准，再选最优动作。
    """
    return _default_minimax_one_step_engine.choose_action(state, side, rng)


# 便于 run_q3 / round_simulator 直接替换时使用的别名
pure_minimax_q3_policy = minimax_one_step_q3_policy


# =========================================================
# 四、自测
# =========================================================
if __name__ == "__main__":
    from models.state import create_initial_round_state

    rng = random.Random(42)
    state = create_initial_round_state()

    print("policy_q3_minimax_one_step.py 自测开始")

    action_my = minimax_one_step_q3_policy(state, "my", rng)
    action_opp = minimax_one_step_q3_policy(state, "opp", rng)

    print("我方 minimax 动作:", action_my.action_type, action_my.action_key)
    print("对方 minimax 动作:", action_opp.action_type, action_opp.action_key)

    print("\n[我方候选动作 minimax 排名 Top 5]")
    rows = _default_minimax_one_step_engine.debug_rank_actions(state, "my")[:5]
    for i, row in enumerate(rows, start=1):
        print(
            f"Rank {i}: {row['action_type']} {row['action_key']} | "
            f"worst_q={row['worst_q']:.3f} | "
            f"worst_opp={row['worst_opp_type']} {row['worst_opp_key']}"
        )

    print("\npolicy_q3_minimax_one_step.py 自测完成")
