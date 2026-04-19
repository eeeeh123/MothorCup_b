from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from config.rules import get_modeling_rule_subset
from config.pm01_params import get_modeling_pm01_subset


RULE = get_modeling_rule_subset()
PM01 = get_modeling_pm01_subset()


# =========================================================
# 一、枚举定义
# =========================================================
class RoundPhase(str, Enum):
    NORMAL = "normal"       # 常规时间
    OVERTIME = "overtime"   # 加时赛
    FINISHED = "finished"   # 回合结束


class DistanceState(str, Enum):
    FAR = "far"
    MID = "mid"
    NEAR = "near"
    CLINCH = "clinch"       # 贴身/缠斗


class PostureState(str, Enum):
    STANDING = "standing"         # 正常站立
    OFF_BALANCE = "off_balance"   # 失衡但未倒
    DOWNED = "downed"             # 倒地
    RECOVERING = "recovering"     # 起身/恢复过程


class InitiativeState(str, Enum):
    MY = "my"
    OPP = "opp"
    NEUTRAL = "neutral"


class WinReason(str, Enum):
    KO = "ko"
    TKO = "tko"
    CONTROL = "effective_control"
    SCORE = "proxy_score"
    FOUL = "foul_limit"
    TIMEOUT = "time_expired"
    UNKNOWN = "unknown"
    DOUBLE_KO = "double_ko"

    TIEBREAK_STATE = "tiebreak_state"
    TIEBREAK_INITIATIVE = "tiebreak_initiative"

# =========================================================
# 二、单个机器人状态
# =========================================================
@dataclass
class FighterState:
    """
    描述某一时刻单个机器人的状态。
    """

    name: str

    # 核心状态
    hp_proxy: float = PM01["hp_proxy_default"]            # 代理状态值，不是官方血量
    stability: float = PM01["stability_default"]          # 稳定性
    energy: float = PM01["energy_default"]                # 体力/能量余量

    # 姿态状态
    posture: PostureState = PostureState.STANDING
    downed_time_sec: float = 0.0                          # 倒地持续时间
    recovery_time_left_sec: float = 0.0                   # 起身/恢复剩余时间

    # 控制与节奏
    control_time_sec: float = 0.0                         # 当前累计有效压制时间
    combo_window_sec: float = 0.0                         # 连续追击窗口
    last_action_code: Optional[str] = None                # 上一个动作编号
    last_action_name: Optional[str] = None                # 上一个动作名称

    # 规则相关
    foul_points: int = 0                                  # 累计违规分
    is_active: bool = True                                # 是否仍可正常行动

    def clone(self) -> "FighterState":
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hp_proxy": self.hp_proxy,
            "stability": self.stability,
            "energy": self.energy,
            "posture": self.posture.value,
            "downed_time_sec": self.downed_time_sec,
            "recovery_time_left_sec": self.recovery_time_left_sec,
            "control_time_sec": self.control_time_sec,
            "combo_window_sec": self.combo_window_sec,
            "last_action_code": self.last_action_code,
            "last_action_name": self.last_action_name,
            "foul_points": self.foul_points,
            "is_active": self.is_active,
        }

    # -------------------------
    # 基础更新方法
    # -------------------------
    def apply_hp_change(self, delta: float) -> None:
        self.hp_proxy = max(0.0, self.hp_proxy + delta)

    def apply_stability_change(self, delta: float) -> None:
        self.stability = max(0.0, min(100.0, self.stability + delta))

    def apply_energy_change(self, delta: float) -> None:
        self.energy = max(0.0, min(100.0, self.energy + delta))

    def add_control_time(self, dt: float) -> None:
        self.control_time_sec = max(0.0, self.control_time_sec + dt)

    def reset_control_time(self) -> None:
        self.control_time_sec = 0.0

    def add_combo_window(self, dt: float) -> None:
        self.combo_window_sec = max(0.0, self.combo_window_sec + dt)

    def decay_combo_window(self, dt: float) -> None:
        self.combo_window_sec = max(0.0, self.combo_window_sec - dt)

    def add_foul_points(self, points: int) -> None:
        self.foul_points = max(0, self.foul_points + int(points))

    def set_last_action(self, code: Optional[str], name: Optional[str]) -> None:
        self.last_action_code = code
        self.last_action_name = name

    # -------------------------
    # 姿态相关方法
    # -------------------------
    def set_posture(self, posture: PostureState) -> None:
        self.posture = posture
        if posture != PostureState.DOWNED:
            self.downed_time_sec = 0.0

    def update_downed_timer(self, dt: float) -> None:
        if self.posture == PostureState.DOWNED:
            self.downed_time_sec += dt
        else:
            self.downed_time_sec = 0.0

    def update_recovery_timer(self, dt: float) -> None:
        if self.recovery_time_left_sec > 0:
            self.recovery_time_left_sec = max(0.0, self.recovery_time_left_sec - dt)
            if self.recovery_time_left_sec == 0 and self.posture == PostureState.RECOVERING:
                self.posture = PostureState.STANDING

    def start_recovery(self, recovery_time_sec: float) -> None:
        self.posture = PostureState.RECOVERING
        self.recovery_time_left_sec = max(0.0, recovery_time_sec)

    def needs_reset_by_rule(self) -> bool:
        """
        是否已经达到规则上的“10秒未自主站起”阈值。
        Q3 不一定直接使用人工复位，但这个判定留给 Q4 很重要。
        """
        return self.posture == PostureState.DOWNED and self.downed_time_sec >= RULE["reset_stand_timeout_sec"]

    def is_ko_like(self) -> bool:
        """
        代理意义下的 KO/TKO 风险触发条件。
        这里只做保守判断，真正逻辑可在 transition 或 simulator 中再细化。
        """
        return self.hp_proxy <= 0 or self.stability <= 0 or not self.is_active


# =========================================================
# 三、单回合比赛状态
# =========================================================
@dataclass
class RoundState:
    """
    Q3 单回合状态。
    注意：
    1. 当前主要服务 Q3。
    2. Q4 会在外层再套一个 BO3 / 资源状态。
    """

    phase: RoundPhase = RoundPhase.NORMAL
    time_left_sec: float = float(RULE["round_time_sec"])

    my: FighterState = field(default_factory=lambda: FighterState(name="my"))
    opp: FighterState = field(default_factory=lambda: FighterState(name="opp"))

    distance: DistanceState = DistanceState.MID
    initiative: InitiativeState = InitiativeState.NEUTRAL

    # 代理得分（因为官方完整得分细则尚未全部公开，可先用代理分）
    my_score_proxy: float = 0.0
    opp_score_proxy: float = 0.0

    # 过程信息
    step_index: int = 0
    event_log: List[str] = field(default_factory=list)

    # 结果信息
    winner: Optional[str] = None          # "my" / "opp" / "draw" / None
    win_reason: Optional[str] = None

    def clone(self) -> "RoundState":
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "time_left_sec": self.time_left_sec,
            "distance": self.distance.value,
            "initiative": self.initiative.value,
            "my_score_proxy": self.my_score_proxy,
            "opp_score_proxy": self.opp_score_proxy,
            "step_index": self.step_index,
            "winner": self.winner,
            "win_reason": self.win_reason,
            "my": self.my.to_dict(),
            "opp": self.opp.to_dict(),
        }

    # -------------------------
    # 日志与基础操作
    # -------------------------
    def log(self, message: str) -> None:
        self.event_log.append(message)

    def next_step(self) -> None:
        self.step_index += 1

    def set_distance(self, distance: DistanceState) -> None:
        self.distance = distance

    def set_initiative(self, initiative: InitiativeState) -> None:
        self.initiative = initiative

    def add_score(self, side: str, value: float) -> None:
        if side == "my":
            self.my_score_proxy += value
        elif side == "opp":
            self.opp_score_proxy += value
        else:
            raise ValueError(f"未知 side: {side}")

    # -------------------------
    # 时间推进
    # -------------------------
    def advance_time(self, dt: float) -> None:
        """
        推进比赛时间。
        - 若常规时间结束，则进入加时
        - 若加时也结束，则先按代理得分判定
        - 若代理得分相同，则进入扩展 tie-break
        - 保留跨阶段时间溢出
        """
        if self.phase == RoundPhase.FINISHED:
            return

        dt = max(0.0, float(dt))

        # 双方倒地/恢复状态推进
        self.my.update_downed_timer(dt)
        self.opp.update_downed_timer(dt)
        self.my.update_recovery_timer(dt)
        self.opp.update_recovery_timer(dt)

        # combo window 衰减
        self.my.decay_combo_window(dt)
        self.opp.decay_combo_window(dt)

        if self.phase == RoundPhase.NORMAL:
            if dt < self.time_left_sec:
                self.time_left_sec -= dt
                return

            overflow = dt - self.time_left_sec
            self.phase = RoundPhase.OVERTIME
            self.time_left_sec = max(0.0, float(RULE["overtime_sec"]) - overflow)
            self.log("常规时间结束，进入加时赛。")

            if self.time_left_sec <= 0:
                self.finish_by_score()
            return

        if self.phase == RoundPhase.OVERTIME:
            self.time_left_sec = max(0.0, self.time_left_sec - dt)
            if self.time_left_sec <= 0:
                self.finish_by_score()

    # -------------------------
    # 控制/压制判定
    # -------------------------
    def register_control(self, side: str, dt: float) -> None:
        """
        登记一段有效控制时间。
        若累计达到规则阈值，则该 side 直接获胜。
        """
        if self.phase == RoundPhase.FINISHED:
            return

        if side == "my":
            self.my.add_control_time(dt)
            self.opp.reset_control_time()
            if self.my.control_time_sec >= RULE["effective_control_sec"]:
                self.finish_round("my", WinReason.CONTROL.value)
        elif side == "opp":
            self.opp.add_control_time(dt)
            self.my.reset_control_time()
            if self.opp.control_time_sec >= RULE["effective_control_sec"]:
                self.finish_round("opp", WinReason.CONTROL.value)
        else:
            raise ValueError(f"未知 side: {side}")

    def reset_both_control(self) -> None:
        self.my.reset_control_time()
        self.opp.reset_control_time()

    # -------------------------
    # 犯规与终局判定
    # -------------------------
    def apply_foul(self, side: str, foul_points: int) -> None:
        """
        对某一方累计犯规分。
        达到规则上限则判负。
        """
        if side == "my":
            self.my.add_foul_points(foul_points)
            if self.my.foul_points >= RULE["tko_penalty_cap"]:
                self.finish_round("opp", WinReason.FOUL.value)
        elif side == "opp":
            self.opp.add_foul_points(foul_points)
            if self.opp.foul_points >= RULE["tko_penalty_cap"]:
                self.finish_round("my", WinReason.FOUL.value)
        else:
            raise ValueError(f"未知 side: {side}")

    def check_ko_like_terminal(self) -> Optional[str]:
        my_down = self.my.is_ko_like()
        opp_down = self.opp.is_ko_like()

        if my_down and opp_down:
            return "draw"
        if my_down and not opp_down:
            return "opp"
        if opp_down and not my_down:
            return "my"
        return None

    def finish_round(self, winner: str, reason: str) -> None:
        self.phase = RoundPhase.FINISHED
        self.winner = winner
        self.win_reason = reason
        self.log(f"回合结束：winner={winner}, reason={reason}")

    def finish_by_score(self) -> None:
        """
        时间耗尽后，先按代理得分判定。
        若代理得分相同，则进入扩展 tie-break。
        """
        if self.phase == RoundPhase.FINISHED:
            return

        if self.my_score_proxy > self.opp_score_proxy:
            self.phase = RoundPhase.FINISHED
            self.winner = "my"
            self.win_reason = WinReason.SCORE.value
            self.log("时间耗尽，按代理得分判定 my 获胜。")
            return

        if self.opp_score_proxy > self.my_score_proxy:
            self.phase = RoundPhase.FINISHED
            self.winner = "opp"
            self.win_reason = WinReason.SCORE.value
            self.log("时间耗尽，按代理得分判定 opp 获胜。")
            return

        self.log("时间耗尽，代理得分相同，启用扩展 tie-break。")
        self.finish_by_extended_tiebreak()

    def is_finished(self) -> bool:
        return self.phase == RoundPhase.FINISHED

    def _posture_rank(self, posture: PostureState) -> int:
        rank_map = {
            PostureState.STANDING: 3,
            PostureState.OFF_BALANCE: 2,
            PostureState.RECOVERING: 1,
            PostureState.DOWNED: 0,
        }
        return rank_map.get(posture, 0)

    def _fighter_state_key(self, fighter: FighterState) -> float:
        """
        扩展 tie-break 的状态综合指标。
        权重含义：
        - hp_proxy：最重要
        - stability：次重要
        - energy：再次之
        """
        return (
            1.00 * fighter.hp_proxy
            + 0.60 * fighter.stability
            + 0.20 * fighter.energy
        )

    def finish_by_extended_tiebreak(self) -> None:
        """
        加时结束后若代理得分仍相同，则使用扩展 tie-break。
        判定顺序：
        1) 状态综合指标
        2) 姿态等级
        3) 主动权
        4) 仍无法区分则保留 draw
        """
        if self.phase == RoundPhase.FINISHED:
            return

        self.phase = RoundPhase.FINISHED
        eps = 1e-9

        my_key = self._fighter_state_key(self.my)
        opp_key = self._fighter_state_key(self.opp)

        if my_key > opp_key + eps:
            self.winner = "my"
            self.win_reason = WinReason.TIEBREAK_STATE.value
            self.log("加时后同分，按扩展状态指标判定 my 获胜。")
            return

        if opp_key > my_key + eps:
            self.winner = "opp"
            self.win_reason = WinReason.TIEBREAK_STATE.value
            self.log("加时后同分，按扩展状态指标判定 opp 获胜。")
            return

        my_posture_rank = self._posture_rank(self.my.posture)
        opp_posture_rank = self._posture_rank(self.opp.posture)

        if my_posture_rank > opp_posture_rank:
            self.winner = "my"
            self.win_reason = WinReason.TIEBREAK_STATE.value
            self.log("加时后同分，按姿态等级判定 my 获胜。")
            return

        if opp_posture_rank > my_posture_rank:
            self.winner = "opp"
            self.win_reason = WinReason.TIEBREAK_STATE.value
            self.log("加时后同分，按姿态等级判定 opp 获胜。")
            return

        if self.initiative == InitiativeState.MY:
            self.winner = "my"
            self.win_reason = WinReason.TIEBREAK_INITIATIVE.value
            self.log("加时后同分，按主动权判定 my 获胜。")
            return

        if self.initiative == InitiativeState.OPP:
            self.winner = "opp"
            self.win_reason = WinReason.TIEBREAK_INITIATIVE.value
            self.log("加时后同分，按主动权判定 opp 获胜。")
            return

        self.winner = "draw"
        self.win_reason = WinReason.SCORE.value
        self.log("加时后同分，扩展 tie-break 仍无法区分，保留 draw。")
        
# =========================================================
# 四、状态初始化与辅助函数
# =========================================================
def create_initial_fighter(name: str) -> FighterState:
    """
    创建单个机器人的默认初始状态。
    """
    return FighterState(
        name=name,
        hp_proxy=PM01["hp_proxy_default"],
        stability=PM01["stability_default"],
        energy=PM01["energy_default"],
        posture=PostureState.STANDING,
        downed_time_sec=0.0,
        recovery_time_left_sec=0.0,
        control_time_sec=0.0,
        combo_window_sec=0.0,
        foul_points=0,
        is_active=True,
    )


def create_initial_round_state() -> RoundState:
    """
    创建 Q3 单回合初始状态。
    """
    return RoundState(
        phase=RoundPhase.NORMAL,
        time_left_sec=float(RULE["round_time_sec"]),
        my=create_initial_fighter("my"),
        opp=create_initial_fighter("opp"),
        distance=DistanceState.MID,
        initiative=InitiativeState.NEUTRAL,
        my_score_proxy=0.0,
        opp_score_proxy=0.0,
        step_index=0,
        event_log=[],
        winner=None,
        win_reason=None,
    )


# =========================================================
# 五、自测
# =========================================================
if __name__ == "__main__":
    state = create_initial_round_state()

    print("state.py 自测开始")
    print("初始阶段:", state.phase.value)
    print("初始剩余时间:", state.time_left_sec)
    print("初始距离:", state.distance.value)
    print("我方初始体力:", state.my.energy)
    print("对方初始稳定性:", state.opp.stability)

    state.add_score("my", 8.0)
    state.register_control("my", 1.5)
    state.advance_time(10.0)

    print("推进后剩余时间:", state.time_left_sec)
    print("我方代理得分:", state.my_score_proxy)
    print("我方累计控制时间:", state.my.control_time_sec)
    print("是否结束:", state.is_finished())
    print("state.py 自测完成")