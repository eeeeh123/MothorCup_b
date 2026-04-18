from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass(frozen=True)
class RuleParams:
    """
    众擎机器人竞技赛规则参数（Q3/Q4 直接使用）

    说明：
    1. 本文件分为“官网明确值”和“建模默认值”两层。
    2. 官网明确值尽量保持原意，不做人为改写。
    3. 建模默认值用于状态转移、资源约束和仿真流程。
    """

    # =========================
    # 一、官网明确规则值
    # =========================
    round_time_sec: int = 300                 # 单回合净比赛时间 5 分钟
    overtime_sec: int = 120                  # 加时赛 2 分钟
    bo3_target_wins: int = 2                 # BO3 三局两胜
    effective_control_sec: int = 3           # 有效压制需持续超过 3 秒（建模阈值取 3）
    reset_limit_per_match: int = 2           # 每场比赛最多人工复位 2 次
    reset_stand_timeout_sec: int = 10        # 倒地后 10 秒内未自主站起，可申请人工复位
    reset_process_max_sec: int = 30          # 单次人工复位不超过 30 秒
    tactical_timeout_limit_per_match: int = 2  # 每场比赛最多 2 次战术暂停
    tactical_timeout_max_sec: int = 60       # 单次战术暂停不超过 60 秒
    timeout_gap_net_sec: int = 60            # 两次暂停之间至少间隔 1 分钟净比赛时间
    repair_limit_per_match: int = 1          # 每场比赛最多 1 次紧急故障维修
    repair_max_sec: int = 300                # 紧急故障维修不超过 5 分钟
    battery_swap_max_sec: int = 120          # 换电不超过 2 分钟
    irreversible_fault_max_sec: int = 1800   # 30 分钟内无法修复视为不可逆故障
    spare_robot_replace_limit: int = 1       # 每队仅可更换 1 次备用机器人
    overtime_timeout_allowed: bool = False   # 加时赛期间不得申请战术暂停

    # =========================
    # 二、与胜负/终止条件相关
    # =========================
    tko_penalty_cap: int = 15                # 违规累计扣满 15 分，判负
    dangerous_zone_attack_penalty: int = 10  # 攻击头部/动力系统/电池等危险部位
    destructive_attack_penalty: int = 5      # 尖锐坚硬部件/破坏性攻击
    attack_downed_or_paused_penalty: int = 8 # 攻击已倒地/暂停状态机器人

    # =========================
    # 三、比赛流程默认值（建模时使用）
    # =========================
    prep_time_before_round_sec: int = 180    # 上场准备阶段 3 分钟（流程建模可用）
    between_round_rest_sec: int = 600        # 回合间休息 10 分钟
    countdown_before_start_sec: int = 5      # 开始前 5 秒倒计时

    # =========================
    # 四、Q3/Q4 建模建议默认值（非官网硬规则）
    # =========================
    # 这些值是为了让状态机/仿真器更容易落地，不代表官方新增规则
    max_score_margin_proxy: float = 100.0
    state_time_step_sec: float = 1.0
    control_check_step_sec: float = 0.5
    allow_early_overtime_win_by_first_score: bool = True
    allow_early_overtime_win_by_first_control: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


RULE_PARAMS = RuleParams()


def get_rule_params() -> Dict[str, Any]:
    return RULE_PARAMS.to_dict()


def get_official_rule_subset() -> Dict[str, Any]:
    """
    只返回官网明确写出的核心规则值。
    """
    d = RULE_PARAMS.to_dict()
    keys = [
        "round_time_sec",
        "overtime_sec",
        "bo3_target_wins",
        "effective_control_sec",
        "reset_limit_per_match",
        "reset_stand_timeout_sec",
        "reset_process_max_sec",
        "tactical_timeout_limit_per_match",
        "tactical_timeout_max_sec",
        "timeout_gap_net_sec",
        "repair_limit_per_match",
        "repair_max_sec",
        "battery_swap_max_sec",
        "irreversible_fault_max_sec",
        "spare_robot_replace_limit",
        "overtime_timeout_allowed",
        "tko_penalty_cap",
        "dangerous_zone_attack_penalty",
        "destructive_attack_penalty",
        "attack_downed_or_paused_penalty",
    ]
    return {k: d[k] for k in keys}


def get_modeling_rule_subset() -> Dict[str, Any]:
    """
    返回更适合 Q3/Q4 状态机直接使用的参数集合。
    """
    d = RULE_PARAMS.to_dict()
    keys = [
        "round_time_sec",
        "overtime_sec",
        "bo3_target_wins",
        "effective_control_sec",
        "reset_limit_per_match",
        "reset_stand_timeout_sec",
        "reset_process_max_sec",
        "tactical_timeout_limit_per_match",
        "tactical_timeout_max_sec",
        "timeout_gap_net_sec",
        "repair_limit_per_match",
        "repair_max_sec",
        "battery_swap_max_sec",
        "irreversible_fault_max_sec",
        "spare_robot_replace_limit",
        "overtime_timeout_allowed",
        "tko_penalty_cap",
        "prep_time_before_round_sec",
        "between_round_rest_sec",
        "countdown_before_start_sec",
        "state_time_step_sec",
        "control_check_step_sec",
        "allow_early_overtime_win_by_first_score",
        "allow_early_overtime_win_by_first_control",
    ]
    return {k: d[k] for k in keys}


if __name__ == "__main__":
    print("rule_params.py 自测开始")
    params = get_rule_params()
    print(f"单回合净比赛时间: {params['round_time_sec']} 秒")
    print(f"加时赛时间: {params['overtime_sec']} 秒")
    print(f"BO3 目标胜局: {params['bo3_target_wins']}")
    print(f"人工复位上限: {params['reset_limit_per_match']}")
    print(f"战术暂停上限: {params['tactical_timeout_limit_per_match']}")
    print(f"紧急维修上限: {params['repair_limit_per_match']}")
    print("rule_params.py 自测完成")