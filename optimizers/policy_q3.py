from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
# 一、配置
# =========================================================
@dataclass
class PolicyQ3Config:
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

    # 动作池规模控制
    max_attack_candidates: int = 6
    max_defense_candidates: int = 5

    # 对手动作池评估方式
    max_opp_candidates: int = 5


DEFAULT_CONFIG = PolicyQ3Config()


# =========================================================
# 二、策略引擎
# =========================================================
class Q3PolicyEngine:
    """
    Q3 策略引擎（第一版）

    核心思想：
    - 枚举当前可选动作
    - 枚举对手候选动作池
    - 调用 transition.step(..., mode='expected') 做一步前瞻
    - 计算 即时奖励 + gamma * 下一状态价值
    - 选择价值最高动作
    """

    def __init__(self, config: PolicyQ3Config = DEFAULT_CONFIG) -> None:
        self.config = config
        self.engine = TransitionEngine()

    # =====================================================
    # 三、候选动作生成
    # =====================================================
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

        # 体力低时删去高成本动作
        if me.energy < 25:
            pool = [a for a in pool if a not in {"A06", "A11", "A12"}]

        # 稳定性低时删去高风险动作
        if me.stability < 35:
            pool = [a for a in pool if a not in {"A06", "A11", "A12"}]

        # 对方失衡时倾向可追击动作
        opp = state.opp if side == "my" else state.my
        if opp.posture == PostureState.OFF_BALANCE:
            bonus_pool = ["A03", "A09", "A12"]
            pool = bonus_pool + [x for x in pool if x not in bonus_pool]

        # 去重并截断
        uniq = []
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

        # 稳定性低时优先平衡类
        if me.stability < 35:
            priority = ["D14", "D15", "D12"]
            pool = priority + [x for x in pool if x not in priority]

        uniq = []
        for x in pool:
            if x not in uniq:
                uniq.append(x)

        return uniq[: self.config.max_defense_candidates]

    def build_candidate_actions(self, state: RoundState, side: str) -> List[ActionDecision]:
        me = state.my if side == "my" else state.opp

        # 倒地/恢复中：只允许 recover
        if me.posture in [PostureState.DOWNED, PostureState.RECOVERING]:
            return [ActionDecision("recover", None)]

        candidates: List[ActionDecision] = []

        # attack
        for atk in self._attack_candidates(state, side):
            candidates.append(ActionDecision("attack", atk))

        # defend
        for dfd in self._defense_candidates(state, side):
            candidates.append(ActionDecision("defend", dfd))

        # hold / recover
        candidates.append(ActionDecision("hold", None))
        if me.energy < 30 or me.stability < 40:
            candidates.append(ActionDecision("recover", None))

        # 去重
        uniq: List[ActionDecision] = []
        seen = set()
        for c in candidates:
            key = (c.action_type, c.action_key)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        return uniq

    def build_opponent_pool(self, state: RoundState, side: str) -> List[ActionDecision]:
        """
        为当前 side 评估动作时，构造对手的候选动作池。
        """
        other = "opp" if side == "my" else "my"
        pool = self.build_candidate_actions(state, other)
        return pool[: self.config.max_opp_candidates]

    # =====================================================
    # 四、状态价值函数
    # =====================================================
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

        # 姿态奖励/惩罚
        if opp.posture == PostureState.OFF_BALANCE:
            value += self.config.opp_off_balance_bonus
        if opp.posture == PostureState.DOWNED:
            value += self.config.opp_downed_bonus
        if me.posture == PostureState.OFF_BALANCE:
            value -= self.config.my_off_balance_penalty
        if me.posture == PostureState.DOWNED:
            value -= self.config.my_downed_penalty

        # 终局特殊值
        if state.is_finished():
            if state.winner == side:
                value += self.config.terminal_win_bonus
            elif state.winner is None or state.winner == "draw":
                value += self.config.terminal_draw_bonus
            else:
                value -= self.config.terminal_lose_penalty

        return value

    # =====================================================
    # 五、动作价值评估
    # =====================================================
    def _simulate_pair(
        self,
        state: RoundState,
        side: str,
        my_action: ActionDecision,
        opp_action: ActionDecision,
    ):
        """
        统一从 side 视角调用 TransitionEngine.step
        """
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

    def evaluate_action(
        self,
        state: RoundState,
        side: str,
        action: ActionDecision,
        opp_pool: List[ActionDecision],
    ) -> float:
        """
        评估某个动作的期望价值：
        Q(s,a) = E[R + gamma * V(s')]
        """
        if not opp_pool:
            return -1e9

        total = 0.0

        for opp_action in opp_pool:
            result = self._simulate_pair(state, side, action, opp_action)

            if side == "my":
                immediate_reward = result.reward_my - 0.35 * result.reward_opp
            else:
                immediate_reward = result.reward_opp - 0.35 * result.reward_my

            future_value = self.state_value(result.next_state, side)
            q = immediate_reward + self.config.gamma * future_value
            total += q

        return total / len(opp_pool)

    # =====================================================
    # 六、主策略接口
    # =====================================================
    def choose_action(
        self,
        state: RoundState,
        side: str,
        rng: Optional[random.Random] = None,
    ) -> ActionDecision:
        rng = rng or random.Random()

        candidates = self.build_candidate_actions(state, side)
        opp_pool = self.build_opponent_pool(state, side)

        # 安全兜底
        if not candidates:
            return ActionDecision("hold", None)

        scored: List[Tuple[ActionDecision, float]] = []
        for action in candidates:
            score = self.evaluate_action(state, side, action, opp_pool)
            scored.append((action, score))

        # 从高到低排序
        scored.sort(key=lambda x: x[1], reverse=True)

        # 取最优；若非常接近，则允许少量随机打散，避免策略过于机械
        best_action, best_score = scored[0]
        near_best = [item for item in scored if item[1] >= best_score - 1.0]

        chosen = rng.choice(near_best)[0] if len(near_best) > 1 else best_action
        return chosen


# =========================================================
# 七、对外暴露的策略函数
# =========================================================
_default_policy_engine = Q3PolicyEngine()


def greedy_q3_policy(
    state: RoundState,
    side: str,
    rng: random.Random,
) -> ActionDecision:
    """
    Q3 第一版主策略：
    一步前瞻 + expected 模式评估。
    """
    return _default_policy_engine.choose_action(state, side, rng)


# =========================================================
# 八、可选：蒙特卡洛 rollout 策略（第二版增强入口）
# =========================================================
def make_rollout_policy(
    rollout_steps: int = 3,
    n_rollouts_per_action: int = 8,
    gamma: float = 0.92,
):
    """
    返回一个“轻量 rollout Monte Carlo”策略函数。
    这不是当前主策略，但你后面想增强时可以直接用。

    思路：
    - 对每个候选动作先走一步
    - 然后随机/贪心混合 rollout 若干步
    - 取平均总回报
    """
    engine = TransitionEngine()
    base_engine = Q3PolicyEngine(DEFAULT_CONFIG)

    def _random_from_candidates(state: RoundState, side: str, rng: random.Random) -> ActionDecision:
        cands = base_engine.build_candidate_actions(state, side)
        return rng.choice(cands) if cands else ActionDecision("hold", None)

    def _policy(state: RoundState, side: str, rng: random.Random) -> ActionDecision:
        candidates = base_engine.build_candidate_actions(state, side)
        opp_pool = base_engine.build_opponent_pool(state, side)

        if not candidates:
            return ActionDecision("hold", None)

        best_action = candidates[0]
        best_value = -1e18

        for action in candidates:
            total_value = 0.0

            for _ in range(n_rollouts_per_action):
                # 先对对手当前反应做一个随机抽样
                opp_action = rng.choice(opp_pool) if opp_pool else ActionDecision("hold", None)

                if side == "my":
                    result = engine.step(
                        state=state.clone(),
                        my_decision=action,
                        opp_decision=opp_action,
                        mode="sample",
                        seed=rng.randint(1, 10**9),
                    )
                else:
                    result = engine.step(
                        state=state.clone(),
                        my_decision=opp_action,
                        opp_decision=action,
                        mode="sample",
                        seed=rng.randint(1, 10**9),
                    )

                rollout_state = result.next_state.clone()
                g = (result.reward_my if side == "my" else result.reward_opp) - 0.35 * (
                    result.reward_opp if side == "my" else result.reward_my
                )

                discount = gamma

                # 后续 rollout
                for _step in range(rollout_steps):
                    if rollout_state.is_finished():
                        break

                    my_roll = _random_from_candidates(rollout_state, "my", rng)
                    opp_roll = _random_from_candidates(rollout_state, "opp", rng)

                    r2 = engine.step(
                        state=rollout_state,
                        my_decision=my_roll,
                        opp_decision=opp_roll,
                        mode="sample",
                        seed=rng.randint(1, 10**9),
                    )
                    rollout_state = r2.next_state

                    step_reward = (r2.reward_my if side == "my" else r2.reward_opp) - 0.35 * (
                        r2.reward_opp if side == "my" else r2.reward_my
                    )
                    g += discount * step_reward
                    discount *= gamma

                g += discount * base_engine.state_value(rollout_state, side)
                total_value += g

            avg_value = total_value / max(1, n_rollouts_per_action)

            if avg_value > best_value:
                best_value = avg_value
                best_action = action

        return best_action

    return _policy


# =========================================================
# 九、自测
# =========================================================
if __name__ == "__main__":
    from models.state import create_initial_round_state

    rng = random.Random(42)
    state = create_initial_round_state()

    print("policy_q3.py 自测开始")

    action_my = greedy_q3_policy(state, "my", rng)
    action_opp = greedy_q3_policy(state, "opp", rng)

    print("我方策略动作:", action_my.action_type, action_my.action_key)
    print("对方策略动作:", action_opp.action_type, action_opp.action_key)

    rollout_policy = make_rollout_policy(rollout_steps=2, n_rollouts_per_action=4)
    action_my_rollout = rollout_policy(state, "my", rng)
    print("我方 rollout 策略动作:", action_my_rollout.action_type, action_my_rollout.action_key)

    print("policy_q3.py 自测完成")