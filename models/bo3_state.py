from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from config.rules import get_modeling_rule_subset
from config.pm01_params import get_modeling_pm01_subset
from models.state import (
    RoundState,
    FighterState,
    PostureState,
    create_initial_round_state,
)


RULE = get_modeling_rule_subset()
PM01 = get_modeling_pm01_subset()


# =========================================================
# 一、工具函数
# =========================================================
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return int(default)


def _clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def _pick_rule_value(candidates: List[str], default: Any) -> Any:
    for key in candidates:
        if key in RULE:
            return RULE[key]
    return default


def _default_hp() -> float:
    return _safe_float(PM01.get("hp_proxy_default", 100.0), 100.0)


def _default_stability() -> float:
    return _safe_float(PM01.get("stability_default", 100.0), 100.0)


def _default_energy() -> float:
    return _safe_float(PM01.get("energy_default", 100.0), 100.0)


def _default_reset_limit() -> int:
    return _safe_int(
        _pick_rule_value(
            [
                "manual_resets_per_match",
                "reset_limit_bo3",
                "reset_limit",
            ],
            2,
        ),
        2,
    )


def _default_timeout_limit() -> int:
    return _safe_int(
        _pick_rule_value(
            [
                "tactical_timeouts_per_match",
                "timeout_limit_bo3",
                "timeout_limit",
            ],
            1,
        ),
        1,
    )


def _default_repair_limit() -> int:
    return _safe_int(
        _pick_rule_value(
            [
                "emergency_repairs_per_match",
                "repair_limit_bo3",
                "repair_limit",
            ],
            1,
        ),
        1,
    )


# =========================================================
# 二、枚举定义
# =========================================================
class SeriesPhase(str, Enum):
    READY = "ready"               # 系列赛已创建，尚未开始第一局
    BETWEEN_ROUNDS = "between_rounds"
    IN_ROUND = "in_round"
    FINISHED = "finished"


class FaultLevel(str, Enum):
    HEALTHY = "healthy"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class ResourceType(str, Enum):
    RESET = "reset"
    TIMEOUT = "timeout"
    REPAIR = "repair"


# =========================================================
# 三、单侧 BO3 外层状态
# =========================================================
@dataclass
class SideBO3State:
    name: str

    # 系列赛比分
    round_wins: int = 0

    # 资源
    resets_left: int = field(default_factory=_default_reset_limit)
    timeouts_left: int = field(default_factory=_default_timeout_limit)
    repairs_left: int = field(default_factory=_default_repair_limit)

    # 跨局延续代价（用于下一局初始状态修正）
    carry_hp_debt: float = 0.0
    carry_stability_debt: float = 0.0
    carry_energy_debt: float = 0.0

    # 故障状态
    accumulated_fault_score: float = 0.0
    fault_level: FaultLevel = FaultLevel.HEALTHY

    # 最近一局摘要
    last_round_reward: float = 0.0
    last_round_win_reason: Optional[str] = None

    def clone(self) -> "SideBO3State":
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "round_wins": self.round_wins,
            "resets_left": self.resets_left,
            "timeouts_left": self.timeouts_left,
            "repairs_left": self.repairs_left,
            "carry_hp_debt": self.carry_hp_debt,
            "carry_stability_debt": self.carry_stability_debt,
            "carry_energy_debt": self.carry_energy_debt,
            "accumulated_fault_score": self.accumulated_fault_score,
            "fault_level": self.fault_level.value,
            "last_round_reward": self.last_round_reward,
            "last_round_win_reason": self.last_round_win_reason,
        }

    def add_win(self) -> None:
        self.round_wins += 1

    def can_use(self, resource_type: ResourceType) -> bool:
        if resource_type == ResourceType.RESET:
            return self.resets_left > 0
        if resource_type == ResourceType.TIMEOUT:
            return self.timeouts_left > 0
        if resource_type == ResourceType.REPAIR:
            return self.repairs_left > 0
        return False

    def consume(self, resource_type: ResourceType) -> bool:
        if not self.can_use(resource_type):
            return False

        if resource_type == ResourceType.RESET:
            self.resets_left -= 1
        elif resource_type == ResourceType.TIMEOUT:
            self.timeouts_left -= 1
        elif resource_type == ResourceType.REPAIR:
            self.repairs_left -= 1
        return True


# =========================================================
# 四、BO3 系列赛外层状态
# =========================================================
@dataclass
class BO3State:
    """
    Q4 外层状态：
    - 负责 BO3 比分
    - 负责人工复位 / 战术暂停 / 紧急维修资源
    - 负责将上一局结果折算成下一局初始惩罚
    - 内层单局状态由 RoundState 承担
    """
    best_of: int = 3
    phase: SeriesPhase = SeriesPhase.READY

    current_round_index: int = 0
    my: SideBO3State = field(default_factory=lambda: SideBO3State(name="my"))
    opp: SideBO3State = field(default_factory=lambda: SideBO3State(name="opp"))

    current_round_state: Optional[RoundState] = None

    winner: Optional[str] = None      # "my" / "opp" / "draw" / None
    win_reason: Optional[str] = None

    series_log: List[str] = field(default_factory=list)

    # -------------------------
    # 基础方法
    # -------------------------
    def clone(self) -> "BO3State":
        return deepcopy(self)

    def log(self, message: str) -> None:
        self.series_log.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_of": self.best_of,
            "phase": self.phase.value,
            "current_round_index": self.current_round_index,
            "winner": self.winner,
            "win_reason": self.win_reason,
            "my": self.my.to_dict(),
            "opp": self.opp.to_dict(),
            "current_round_state": None if self.current_round_state is None else self.current_round_state.to_dict(),
            "series_log": list(self.series_log),
        }

    def wins_needed(self) -> int:
        return self.best_of // 2 + 1

    def is_finished(self) -> bool:
        return self.phase == SeriesPhase.FINISHED

    def scoreline(self) -> str:
        return f"{self.my.round_wins}:{self.opp.round_wins}"

    # -------------------------
    # 故障等级映射
    # -------------------------
    @staticmethod
    def fault_level_from_score(score: float) -> FaultLevel:
        if score < 12.0:
            return FaultLevel.HEALTHY
        if score < 28.0:
            return FaultLevel.MINOR
        if score < 48.0:
            return FaultLevel.MAJOR
        return FaultLevel.CRITICAL

    def _refresh_fault_level(self, side_state: SideBO3State) -> None:
        side_state.accumulated_fault_score = _clamp(side_state.accumulated_fault_score, 0.0, 100.0)
        side_state.fault_level = self.fault_level_from_score(side_state.accumulated_fault_score)

    # -------------------------
    # 跨局延续折算
    # -------------------------
    def _estimate_carryover_from_round(self, fighter: FighterState, side_state: SideBO3State) -> None:
        """
        将某一局结束时的单侧状态，折算成下一局初始惩罚。
        这不是官方精确物理量，而是 Q4 外层状态机的资源规划代理量。
        """
        hp_loss = max(0.0, _default_hp() - fighter.hp_proxy)
        stability_loss = max(0.0, _default_stability() - fighter.stability)
        energy_loss = max(0.0, _default_energy() - fighter.energy)

        # 折算为下一局初始惩罚（保守比例）
        side_state.carry_hp_debt = _clamp(0.10 * hp_loss, 0.0, 25.0)
        side_state.carry_stability_debt = _clamp(0.16 * stability_loss, 0.0, 30.0)
        side_state.carry_energy_debt = _clamp(0.20 * energy_loss, 0.0, 35.0)

        # 故障累计分：更偏重稳定性损失与倒地/停机风险
        fault_increment = (
            0.08 * hp_loss
            + 0.18 * stability_loss
            + 0.06 * energy_loss
        )

        if fighter.posture == PostureState.DOWNED:
            fault_increment += 10.0
        elif fighter.posture == PostureState.OFF_BALANCE:
            fault_increment += 4.0

        if not fighter.is_active:
            fault_increment += 20.0

        side_state.accumulated_fault_score += fault_increment
        self._refresh_fault_level(side_state)

    def _apply_carryover_to_fighter(self, fighter: FighterState, side_state: SideBO3State) -> None:
        """
        将 BO3 外层跨局惩罚施加到下一局初始 FighterState。
        """
        fighter.apply_hp_change(-side_state.carry_hp_debt)
        fighter.apply_stability_change(-side_state.carry_stability_debt)
        fighter.apply_energy_change(-side_state.carry_energy_debt)

        # 根据故障等级施加额外惩罚
        if side_state.fault_level == FaultLevel.MINOR:
            fighter.apply_stability_change(-4.0)
            fighter.apply_energy_change(-2.0)
        elif side_state.fault_level == FaultLevel.MAJOR:
            fighter.apply_stability_change(-10.0)
            fighter.apply_energy_change(-6.0)
        elif side_state.fault_level == FaultLevel.CRITICAL:
            fighter.apply_stability_change(-18.0)
            fighter.apply_energy_change(-10.0)

    # -------------------------
    # 系列赛流程
    # -------------------------
    def start_next_round(self) -> RoundState:
        """
        开始下一局。
        若系列赛已结束则抛错。
        """
        if self.is_finished():
            raise RuntimeError("系列赛已结束，不能继续开始新回合。")

        if self.current_round_index >= self.best_of:
            self.finish_series_by_scoreboard("round_cap_reached")
            raise RuntimeError("已达到 best_of 上限。")

        round_state = create_initial_round_state()

        # 将跨局延续代价施加到下一局初始状态
        self._apply_carryover_to_fighter(round_state.my, self.my)
        self._apply_carryover_to_fighter(round_state.opp, self.opp)

        self.current_round_state = round_state
        self.current_round_index += 1
        self.phase = SeriesPhase.IN_ROUND

        self.log(
            f"开始第 {self.current_round_index} 局："
            f"score={self.scoreline()}, "
            f"my_fault={self.my.fault_level.value}, "
            f"opp_fault={self.opp.fault_level.value}"
        )
        return round_state

    def end_round(
        self,
        final_round_state: RoundState,
        winner: Optional[str],
        win_reason: Optional[str],
        reward_my: float = 0.0,
        reward_opp: float = 0.0,
    ) -> None:
        """
        用一局的最终结果更新 BO3 外层状态。
        """
        if self.is_finished():
            return

        self.current_round_state = final_round_state
        self.my.last_round_reward = reward_my
        self.opp.last_round_reward = reward_opp
        self.my.last_round_win_reason = win_reason
        self.opp.last_round_win_reason = win_reason

        if winner == "my":
            self.my.add_win()
        elif winner == "opp":
            self.opp.add_win()

        # 把单局末状态转成跨局延续惩罚
        self._estimate_carryover_from_round(final_round_state.my, self.my)
        self._estimate_carryover_from_round(final_round_state.opp, self.opp)

        self.log(
            f"第 {self.current_round_index} 局结束：winner={winner}, "
            f"reason={win_reason}, score={self.scoreline()}"
        )

        if self.my.round_wins >= self.wins_needed():
            self.phase = SeriesPhase.FINISHED
            self.winner = "my"
            self.win_reason = "bo3_score"
            self.log("系列赛结束：my 获胜。")
            return

        if self.opp.round_wins >= self.wins_needed():
            self.phase = SeriesPhase.FINISHED
            self.winner = "opp"
            self.win_reason = "bo3_score"
            self.log("系列赛结束：opp 获胜。")
            return

        if self.current_round_index >= self.best_of:
            self.finish_series_by_scoreboard("bo3_completed")
            return

        self.phase = SeriesPhase.BETWEEN_ROUNDS

    def end_round_from_sim_result(self, sim_result: Any) -> None:
        """
        兼容 round_simulator 返回对象的鸭子类型接口：
        需要有 final_state / winner / win_reason / total_reward_my / total_reward_opp
        """
        self.end_round(
            final_round_state=sim_result.final_state,
            winner=sim_result.winner,
            win_reason=sim_result.win_reason,
            reward_my=getattr(sim_result, "total_reward_my", 0.0),
            reward_opp=getattr(sim_result, "total_reward_opp", 0.0),
        )

    def finish_series_by_scoreboard(self, reason: str = "bo3_scoreboard") -> None:
        self.phase = SeriesPhase.FINISHED
        if self.my.round_wins > self.opp.round_wins:
            self.winner = "my"
        elif self.opp.round_wins > self.my.round_wins:
            self.winner = "opp"
        else:
            self.winner = "draw"
        self.win_reason = reason
        self.log(f"系列赛按比分结束：winner={self.winner}, reason={reason}")

    # -------------------------
    # 资源动作：暂停 / 复位 / 维修
    # -------------------------
    def _get_side_state(self, side: str) -> SideBO3State:
        if side == "my":
            return self.my
        if side == "opp":
            return self.opp
        raise ValueError(f"未知 side: {side}")

    def _get_round_fighter(self, side: str) -> FighterState:
        if self.current_round_state is None:
            raise RuntimeError("当前没有正在进行的回合。")
        return self.current_round_state.my if side == "my" else self.current_round_state.opp

    def can_use_resource(self, side: str, resource_type: ResourceType) -> bool:
        return self._get_side_state(side).can_use(resource_type)

    def use_timeout(self, side: str) -> bool:
        """
        战术暂停：
        - 消耗一个 timeout
        - 当前局内给予适度稳定性/能量恢复
        - 清空一部分对方控制累计
        """
        side_state = self._get_side_state(side)
        if not side_state.consume(ResourceType.TIMEOUT):
            return False

        if self.current_round_state is not None and not self.current_round_state.is_finished():
            fighter = self._get_round_fighter(side)
            fighter.apply_stability_change(+8.0)
            fighter.apply_energy_change(+5.0)

            # 暂停期间视为打断节奏
            fighter.reset_control_time()
            if side == "my":
                self.current_round_state.opp.reset_control_time()
            else:
                self.current_round_state.my.reset_control_time()

        self.log(f"{side} 使用战术暂停，剩余={side_state.timeouts_left}")
        return True

    def use_manual_reset(self, side: str) -> bool:
        """
        人工复位：
        - 主要用于倒地/严重失衡时恢复姿态
        - 消耗一个 reset
        """
        side_state = self._get_side_state(side)
        if not side_state.consume(ResourceType.RESET):
            return False

        if self.current_round_state is not None and not self.current_round_state.is_finished():
            fighter = self._get_round_fighter(side)

            fighter.set_posture(PostureState.RECOVERING)
            fighter.recovery_time_left_sec = 0.0
            fighter.apply_stability_change(+18.0)
            fighter.apply_energy_change(-2.0)
            fighter.is_active = True

        # 复位会略微增加后续故障累积
        side_state.accumulated_fault_score += 2.0
        self._refresh_fault_level(side_state)

        self.log(f"{side} 使用人工复位，剩余={side_state.resets_left}")
        return True

    def use_emergency_repair(self, side: str) -> bool:
        """
        紧急维修：
        - 消耗一个 repair
        - 降低跨局惩罚和故障分
        - 若在局内使用，也可提供少量即时恢复
        """
        side_state = self._get_side_state(side)
        if not side_state.consume(ResourceType.REPAIR):
            return False

        side_state.carry_hp_debt = max(0.0, side_state.carry_hp_debt - 6.0)
        side_state.carry_stability_debt = max(0.0, side_state.carry_stability_debt - 10.0)
        side_state.carry_energy_debt = max(0.0, side_state.carry_energy_debt - 8.0)
        side_state.accumulated_fault_score = max(0.0, side_state.accumulated_fault_score - 14.0)
        self._refresh_fault_level(side_state)

        if self.current_round_state is not None and not self.current_round_state.is_finished():
            fighter = self._get_round_fighter(side)
            fighter.apply_stability_change(+10.0)
            fighter.apply_energy_change(+6.0)
            fighter.is_active = True

        self.log(f"{side} 使用紧急维修，剩余={side_state.repairs_left}")
        return True


# =========================================================
# 五、初始化函数
# =========================================================
def create_initial_bo3_state() -> BO3State:
    state = BO3State(
        best_of=3,
        phase=SeriesPhase.READY,
        current_round_index=0,
        my=SideBO3State(name="my"),
        opp=SideBO3State(name="opp"),
        current_round_state=None,
        winner=None,
        win_reason=None,
        series_log=[],
    )
    state.log(
        f"初始化 BO3 状态：best_of=3, "
        f"reset_limit={state.my.resets_left}, "
        f"timeout_limit={state.my.timeouts_left}, "
        f"repair_limit={state.my.repairs_left}"
    )
    return state


# =========================================================
# 六、自测
# =========================================================
if __name__ == "__main__":
    bo3 = create_initial_bo3_state()

    print("bo3_state.py 自测开始")
    print("初始 phase:", bo3.phase.value)
    print("初始比分:", bo3.scoreline())
    print("我方资源:", bo3.my.to_dict())

    round_state = bo3.start_next_round()
    print("开始第几局:", bo3.current_round_index)
    print("当前 phase:", bo3.phase.value)
    print("当前局初始我方能量:", round_state.my.energy)

    # 模拟一局结束
    round_state.my.hp_proxy = 62.0
    round_state.my.stability = 48.0
    round_state.my.energy = 72.0

    round_state.opp.hp_proxy = 0.0
    round_state.opp.stability = 8.0
    round_state.opp.energy = 85.0
    round_state.finish_round("my", "ko_like")

    bo3.end_round(
        final_round_state=round_state,
        winner=round_state.winner,
        win_reason=round_state.win_reason,
        reward_my=120.0,
        reward_opp=20.0,
    )

    print("一局后比分:", bo3.scoreline())
    print("一局后 phase:", bo3.phase.value)
    print("我方跨局状态:", bo3.my.to_dict())
    print("对方跨局状态:", bo3.opp.to_dict())

    bo3.start_next_round()
    print("第二局开始后，我方初始能量:", bo3.current_round_state.my.energy)
    print("第二局开始后，对方初始稳定性:", bo3.current_round_state.opp.stability)

    print("bo3_state.py 自测完成")