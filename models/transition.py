from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from config.rules import get_modeling_rule_subset
from models.state import (
    RoundState,
    FighterState,
    DistanceState,
    InitiativeState,
    PostureState,
    WinReason,
)
from models.matchup_matrix import (
    prepare_attack_records,
    prepare_defense_records,
    build_matchup_matrix,
)


RULE = get_modeling_rule_subset()


# =========================================================
# 一、基础数据结构
# =========================================================
@dataclass
class ActionDecision:
    """
    单步决策。
    action_type:
        - attack
        - defend
        - hold
        - recover
    action_key:
        可传攻击/防守动作 code 或 name
    """
    action_type: str
    action_key: Optional[str] = None


@dataclass
class TransitionResult:
    next_state: RoundState
    reward_my: float
    reward_opp: float
    info: Dict[str, Any]


# =========================================================
# 二、工具函数
# =========================================================
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _normalize(value: float, low: float, high: float) -> float:
    if high - low <= 1e-12:
        return 0.0
    return _clamp((value - low) / (high - low), 0.0, 1.0)


# =========================================================
# 三、Transition Engine
# =========================================================
class TransitionEngine:
    """
    Q3 单步状态转移引擎。

    支持两种模式：
    - mode='expected'：按期望值更新，适合 DP / MDP
    - mode='sample'：按概率采样，适合蒙特卡洛仿真
    """

    def __init__(self) -> None:
        self.attack_records = prepare_attack_records()
        self.defense_records = prepare_defense_records()

        self.attack_map = self._build_attack_map(self.attack_records)
        self.defense_map = self._build_defense_map(self.defense_records)

        matchup_df = build_matchup_matrix(self.attack_records, self.defense_records)
        self.matchup_lookup = self._build_matchup_lookup(matchup_df)

    # -------------------------
    # 数据索引
    # -------------------------
    @staticmethod
    def _build_attack_map(records):
        m = {}
        for r in records:
            m[r["code"]] = r
            m[r["name"]] = r
        return m

    @staticmethod
    def _build_defense_map(records):
        m = {}
        for r in records:
            m[r["code"]] = r
            m[r["name"]] = r
        return m

    @staticmethod
    def _build_matchup_lookup(df):
        lookup = {}
        for _, row in df.iterrows():
            key = (row["attack_name"], row["defense_name"])
            lookup[key] = {
                "score": _safe_float(row.get("matchup_score", 0.0)),
                "reason": row.get("reason", ""),
            }
        return lookup

    # -------------------------
    # 动作查询
    # -------------------------
    def get_attack(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        if key is None:
            return None
        return self.attack_map.get(key)

    def get_defense(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        if key is None:
            return None
        return self.defense_map.get(key)

    def get_matchup_score(self, attack_name: str, defense_name: str) -> float:
        item = self.matchup_lookup.get((attack_name, defense_name))
        if item is None:
            return 0.0
        return _safe_float(item["score"], 0.0)

    # =====================================================
    # 四、连续变量读取（兼容当前/后续字段）
    # =====================================================
    def _attack_feature_pack(self, attack: Dict[str, Any]) -> Dict[str, float]:
        """
        攻击动作底层/上层特征统一读取。
        优先读连续变量；若没有，再退回到已有评分字段。
        """
        force = _safe_float(
            attack.get("F", attack.get("avg_impact_force", 0.0))
        )
        ek = _safe_float(
            attack.get("Ek", attack.get("kinetic_energy", 0.0))
        )
        pre = _safe_float(
            attack.get("Tpre", attack.get("time_cost", 0.0))
        )
        reach = _safe_float(
            attack.get("Lreach", 0.0)
        )
        sweep = _safe_float(
            attack.get("theta_sweep", attack.get("θsweep", 0.0))
        )
        kbreak = _safe_float(
            attack.get("Kbreak", attack.get("balance_break_level", 0.0))
        )
        combo = _safe_float(
            attack.get("Ccombo", attack.get("combo_potential", 0.0))
        )
        vuln = _safe_float(
            attack.get("Vvul", attack.get("counter_risk", 0.0))
        )
        pfall = _safe_float(
            attack.get("Pfall", attack.get("p_fall", attack.get("balance_risk", 0.0)))
        )
        trec = _safe_float(
            attack.get("Trec", attack.get("recovery_time", 0.0))
        )
        cstam = _safe_float(
            attack.get("Cstam", attack.get("stamina_cost_raw", attack.get("energy_cost", 0.0)))
        )

        # 归一化（按当前已知量级，后续可统一改）
        return {
            "force_n": _normalize(force, 0.0, 7500.0),
            "ek_n": _normalize(ek, 0.0, 800.0),
            "pre_n": _normalize(pre, 0.10, 0.80),
            "reach_n": _normalize(reach, 0.20, 1.00),
            "sweep_n": _normalize(sweep, 0.0, 3.20),
            "kbreak_n": _normalize(kbreak, 0.0, 1.0 if kbreak <= 1.0 else 10.0),
            "combo_n": _normalize(combo, 0.0, 1.0 if combo <= 1.0 else 10.0),
            "vuln_n": _normalize(vuln, 0.0, 1.0 if vuln <= 1.0 else 10.0),
            "pfall_n": _normalize(pfall, 0.0, 0.50 if pfall <= 1.0 else 10.0),
            "trec_n": _normalize(trec, 0.0, 2.5),
            "cstam_n": _normalize(cstam, 0.0, 1000.0),
            "pre_raw": pre,
            "trec_raw": trec,
            "cstam_raw": cstam,
            "force_raw": force,
            "ek_raw": ek,
            "reach_raw": reach,
        }

    def _defense_feature_pack(self, defense: Dict[str, Any]) -> Dict[str, float]:
        direct = _safe_float(defense.get("direct_defense_level", defense.get("防护量级", 0.0)))
        stability = _safe_float(defense.get("stability_recovery", 0.0))
        counter = _safe_float(defense.get("counter_attack_potential", defense.get("反击潜力加成", 0.0)))
        timing = _safe_float(defense.get("timing_difficulty", 0.0))
        risk = _safe_float(defense.get("risk_if_failed", defense.get("失误惩罚", 0.0)))
        energy = _safe_float(defense.get("energy_cost", defense.get("能耗", 0.0)))
        mobility = _safe_float(defense.get("mobility_loss", 0.0))
        recovery = _safe_float(defense.get("recovery_time", 0.0))

        return {
            "direct_n": _normalize(direct, 0.0, 10.0 if direct > 5 else 5.0),
            "stability_n": _normalize(stability, 0.0, 10.0 if stability > 5 else 5.0),
            "counter_n": _normalize(counter, 0.0, 10.0 if counter > 5 else 5.0),
            "timing_n": _normalize(timing, 0.0, 10.0 if timing > 5 else 5.0),
            "risk_n": _normalize(risk, 0.0, 10.0 if risk > 5 else 5.0),
            "energy_n": _normalize(energy, 0.0, 10.0 if energy > 5 else 5.0),
            "mobility_n": _normalize(mobility, 0.0, 10.0 if mobility > 5 else 5.0),
            "recovery_n": _normalize(recovery, 0.0, 3.0),
            "recovery_raw": recovery,
        }

    # =====================================================
    # 五、上下文修正项
    # =====================================================
    def _distance_bonus(self, attack: Dict[str, Any], state: RoundState, side: str) -> float:
        """
        攻击动作与当前距离是否匹配。
        """
        attack_range = attack.get("attack_range", "near-mid")
        current_distance = state.distance.value

        table = {
            ("near", "clinch"): 1.0,
            ("near", "near"): 0.9,
            ("near-mid", "near"): 1.0,
            ("near-mid", "mid"): 0.9,
            ("mid", "mid"): 1.0,
            ("mid", "near"): 0.6,
            ("far", "far"): 1.0,
            ("mid-far", "far"): 1.0,
            ("mid-far", "mid"): 0.8,
        }
        return table.get((attack_range, current_distance), 0.4)

    def _initiative_bonus(self, state: RoundState, side: str) -> float:
        if side == "my":
            if state.initiative == InitiativeState.MY:
                return 0.15
            if state.initiative == InitiativeState.OPP:
                return -0.10
        elif side == "opp":
            if state.initiative == InitiativeState.OPP:
                return 0.15
            if state.initiative == InitiativeState.MY:
                return -0.10
        return 0.0

    # =====================================================
    # 六、质量分 / 概率 / 伤害
    # =====================================================
    def _attack_quality(self, attack: Dict[str, Any], state: RoundState, side: str) -> float:
        a = self._attack_feature_pack(attack)
        dist_b = self._distance_bonus(attack, state, side)
        init_b = self._initiative_bonus(state, side)

        q = (
            0.24 * a["force_n"]
            + 0.20 * a["ek_n"]
            + 0.14 * a["kbreak_n"]
            + 0.10 * a["sweep_n"]
            + 0.10 * a["combo_n"]
            - 0.10 * a["pre_n"]
            - 0.07 * a["vuln_n"]
            - 0.05 * a["pfall_n"]
            + 0.10 * dist_b
            + init_b
        )
        return q

    def _defense_quality(self, attack: Dict[str, Any], defense: Dict[str, Any]) -> float:
        d = self._defense_feature_pack(defense)
        matchup = self.get_matchup_score(attack["name"], defense["name"])
        matchup_n = _normalize(matchup, 0.0, 100.0)

        q = (
            0.32 * matchup_n
            + 0.24 * d["direct_n"]
            + 0.16 * d["stability_n"]
            + 0.10 * d["counter_n"]
            - 0.08 * d["timing_n"]
            - 0.06 * d["risk_n"]
            - 0.04 * d["energy_n"]
        )
        return q

    def _hit_probability_attack_vs_defense(
        self,
        attack: Dict[str, Any],
        defense: Dict[str, Any],
        state: RoundState,
        attacker_side: str,
    ) -> float:
        q_atk = self._attack_quality(attack, state, attacker_side)
        q_def = self._defense_quality(attack, defense)

        # 攻击质量高于防守质量，则命中概率上升
        x = 4.5 * (q_atk - q_def)
        return _clamp(_sigmoid(x), 0.05, 0.95)

    def _hit_probability_attack_vs_attack(
        self,
        attack_a: Dict[str, Any],
        attack_b: Dict[str, Any],
        state: RoundState,
    ) -> Tuple[float, float]:
        qa = self._attack_quality(attack_a, state, "my")
        qb = self._attack_quality(attack_b, state, "opp")
        diff = qa - qb

        p_my = _clamp(_sigmoid(3.5 * diff), 0.10, 0.90)
        p_opp = _clamp(_sigmoid(-3.5 * diff), 0.10, 0.90)
        return p_my, p_opp

    def _damage_and_stability_effect(self, attack: Dict[str, Any]) -> Dict[str, float]:
        a = self._attack_feature_pack(attack)

        damage = (
            14.0 * a["force_n"]
            + 12.0 * a["ek_n"]
            + 8.0 * a["kbreak_n"]
        )

        stability_loss = (
            16.0 * a["kbreak_n"]
            + 10.0 * a["force_n"]
            + 6.0 * a["sweep_n"]
        )

        base_exposure = (
            8.0 * a["vuln_n"] + 8.0 * a["pfall_n"]
        )

        energy_cost = (
            6.0 * a["cstam_n"]
            + 2.0 * a["pre_n"]
            + 2.0 * a["trec_n"]
        )

        return {
            "damage": damage,
            "stability_loss": stability_loss,
            "base_exposure": base_exposure,
            "energy_cost": energy_cost,
        }

    # =====================================================
    # 七、时间推进
    # =====================================================
    def _estimate_action_duration(
        self,
        decision: ActionDecision,
        action_record: Optional[Dict[str, Any]],
        defense_record: Optional[Dict[str, Any]],
    ) -> float:
        if decision.action_type == "attack" and action_record is not None:
            a = self._attack_feature_pack(action_record)
            # 一个动作步长：前摇 + 一部分硬直
            return max(0.5, a["pre_raw"] + 0.35 * max(a["trec_raw"], 0.2))

        if decision.action_type == "defend" and defense_record is not None:
            d = self._defense_feature_pack(defense_record)
            if d["recovery_raw"] > 0:
                return max(0.4, 0.20 + d["recovery_raw"])
            return max(0.4, 0.25 + 0.20 * _safe_float(defense_record.get("timing_difficulty", 2.0)))

        if decision.action_type == "recover":
            return 0.8

        return 0.5

    # =====================================================
    # 八、即时奖励
    # =====================================================
    def _reward_from_effect(
        self,
        damage_to_opp: float,
        stability_loss_to_opp: float,
        energy_cost_self: float,
        self_exposure: float,
        control_gain: float,
        foul_penalty: float = 0.0,
    ) -> float:
        return (
            1.0 * damage_to_opp
            + 0.6 * stability_loss_to_opp
            + 8.0 * control_gain
            - 0.8 * energy_cost_self
            - 1.0 * self_exposure
            - foul_penalty
        )

    # =====================================================
    # 九、决策解析
    # =====================================================
    def _resolve_single_side_decision(
        self,
        decision: ActionDecision,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        attack_record = None
        defense_record = None

        if decision.action_type == "attack":
            attack_record = self.get_attack(decision.action_key)
        elif decision.action_type == "defend":
            defense_record = self.get_defense(decision.action_key)

        return attack_record, defense_record

    # =====================================================
    # 十、主转移函数
    # =====================================================
    def step(
        self,
        state: RoundState,
        my_decision: ActionDecision,
        opp_decision: ActionDecision,
        mode: str = "expected",   # expected / sample
        seed: Optional[int] = None,
    ) -> TransitionResult:
        """
        单步转移。

        mode='expected':
            用期望值更新，适合 DP / MDP

        mode='sample':
            用概率采样更新，适合 Monte Carlo
        """
        if state.is_finished():
            return TransitionResult(
                next_state=state.clone(),
                reward_my=0.0,
                reward_opp=0.0,
                info={"message": "state already finished"},
            )

        rng = random.Random(seed) if seed is not None else random

        next_state = state.clone()
        next_state.next_step()

        my_attack, my_defense = self._resolve_single_side_decision(my_decision)
        opp_attack, opp_defense = self._resolve_single_side_decision(opp_decision)

        reward_my = 0.0
        reward_opp = 0.0
        info: Dict[str, Any] = {
            "my_decision": my_decision.action_type,
            "opp_decision": opp_decision.action_type,
        }

        # -------------------------------------------------
        # Case 1: 我攻 / 对防
        # -------------------------------------------------
        if my_attack is not None and opp_defense is not None:
            p_hit = self._hit_probability_attack_vs_defense(my_attack, opp_defense, next_state, "my")
            eff = self._damage_and_stability_effect(my_attack)

            if mode == "sample":
                hit = rng.random() < p_hit
                hit_factor = 1.0 if hit else 0.0
                applied_hit = hit_factor
                self_exposure = (0.0 if hit else 1.0) * eff["base_exposure"]
            else:
                hit = None
                hit_factor = p_hit
                applied_hit = p_hit
                self_exposure = (1.0 - p_hit) * eff["base_exposure"]

            next_state.opp.apply_hp_change(-eff["damage"] * applied_hit)
            next_state.opp.apply_stability_change(-eff["stability_loss"] * applied_hit)
            next_state.my.apply_stability_change(-self_exposure)
            next_state.my.apply_energy_change(-eff["energy_cost"])

            # 防守成功会减少对方压制连续性
            if hit_factor > 0.6 and next_state.distance in [DistanceState.NEAR, DistanceState.CLINCH]:
                next_state.register_control("my", min(1.0, RULE["control_check_step_sec"]))
                control_gain = 1.0
            else:
                next_state.reset_both_control()
                control_gain = 0.0

            next_state.set_initiative(
                InitiativeState.MY if hit_factor >= 0.5 else InitiativeState.OPP
            )

            next_state.my.set_last_action(my_attack["code"], my_attack["name"])
            next_state.opp.set_last_action(opp_defense["code"], opp_defense["name"])

            reward_my += self._reward_from_effect(
                damage_to_opp=eff["damage"] * hit_factor,
                stability_loss_to_opp=eff["stability_loss"] * hit_factor,
                energy_cost_self=eff["energy_cost"],
                self_exposure=self_exposure,
                control_gain=control_gain,
            )
            reward_opp += 0.2 * (1.0 - hit_factor)

            info.update({
                "case": "my_attack_vs_opp_defense",
                "p_hit_my": p_hit,
                "hit_my": hit,
                "my_attack_name": my_attack["name"],
                "opp_defense_name": opp_defense["name"],
            })

            dt = max(
                self._estimate_action_duration(my_decision, my_attack, None),
                self._estimate_action_duration(opp_decision, None, opp_defense),
            )
            next_state.advance_time(dt)

        # -------------------------------------------------
        # Case 2: 我防 / 对攻
        # -------------------------------------------------
        elif my_defense is not None and opp_attack is not None:
            p_hit_opp = self._hit_probability_attack_vs_defense(opp_attack, my_defense, next_state, "opp")
            eff = self._damage_and_stability_effect(opp_attack)

            if mode == "sample":
                hit = rng.random() < p_hit_opp
                hit_factor = 1.0 if hit else 0.0
                applied_hit = hit_factor
                self_exposure = (0.0 if hit else 1.0) * eff["base_exposure"]
            else:
                hit = None
                hit_factor = p_hit_opp
                applied_hit = p_hit_opp
                self_exposure = (1.0 - p_hit_opp) * eff["base_exposure"]

            next_state.my.apply_hp_change(-eff["damage"] * applied_hit)
            next_state.my.apply_stability_change(-eff["stability_loss"] * applied_hit)
            next_state.opp.apply_stability_change(-self_exposure)
            next_state.opp.apply_energy_change(-eff["energy_cost"])

            if hit_factor > 0.6 and next_state.distance in [DistanceState.NEAR, DistanceState.CLINCH]:
                next_state.register_control("opp", min(1.0, RULE["control_check_step_sec"]))
                control_gain_opp = 1.0
            else:
                next_state.reset_both_control()
                control_gain_opp = 0.0

            next_state.set_initiative(
                InitiativeState.OPP if hit_factor >= 0.5 else InitiativeState.MY
            )

            next_state.my.set_last_action(my_defense["code"], my_defense["name"])
            next_state.opp.set_last_action(opp_attack["code"], opp_attack["name"])

            reward_opp += self._reward_from_effect(
                damage_to_opp=eff["damage"] * hit_factor,
                stability_loss_to_opp=eff["stability_loss"] * hit_factor,
                energy_cost_self=eff["energy_cost"],
                self_exposure=self_exposure,
                control_gain=control_gain_opp,
            )
            reward_my += 0.2 * (1.0 - hit_factor)

            info.update({
                "case": "my_defense_vs_opp_attack",
                "p_hit_opp": p_hit_opp,
                "hit_opp": hit,
                "my_defense_name": my_defense["name"],
                "opp_attack_name": opp_attack["name"],
            })

            dt = max(
                self._estimate_action_duration(my_decision, None, my_defense),
                self._estimate_action_duration(opp_decision, opp_attack, None),
            )
            next_state.advance_time(dt)

        # -------------------------------------------------
        # Case 3: 双攻
        # -------------------------------------------------
        elif my_attack is not None and opp_attack is not None:
            p_hit_my, p_hit_opp = self._hit_probability_attack_vs_attack(
                my_attack, opp_attack, next_state
            )

            eff_my = self._damage_and_stability_effect(my_attack)
            eff_opp = self._damage_and_stability_effect(opp_attack)

            if mode == "sample":
                hit_my = rng.random() < p_hit_my
                hit_opp = rng.random() < p_hit_opp

                hit_factor_my = 1.0 if hit_my else 0.0
                hit_factor_opp = 1.0 if hit_opp else 0.0

                self_exposure_my = (0.0 if hit_my else 1.0) * eff_my["base_exposure"]
                self_exposure_opp = (0.0 if hit_opp else 1.0) * eff_opp["base_exposure"]
            else:
                hit_my = None
                hit_opp = None

                hit_factor_my = p_hit_my
                hit_factor_opp = p_hit_opp

                self_exposure_my = (1.0 - p_hit_my) * eff_my["base_exposure"]
                self_exposure_opp = (1.0 - p_hit_opp) * eff_opp["base_exposure"]

            # 对对手造成的伤害与稳定性损失
            next_state.opp.apply_hp_change(-eff_my["damage"] * hit_factor_my)
            next_state.opp.apply_stability_change(-eff_my["stability_loss"] * hit_factor_my)

            next_state.my.apply_hp_change(-eff_opp["damage"] * hit_factor_opp)
            next_state.my.apply_stability_change(-eff_opp["stability_loss"] * hit_factor_opp)

            # 自身能量消耗
            next_state.my.apply_energy_change(-eff_my["energy_cost"])
            next_state.opp.apply_energy_change(-eff_opp["energy_cost"])

            # 双攻的额外暴露代价
            next_state.my.apply_stability_change(
                -(self_exposure_my + 0.5 * self_exposure_opp)
            )
            next_state.opp.apply_stability_change(
                -(self_exposure_opp + 0.5 * self_exposure_my)
            )

            # 主动权更新
            if hit_factor_my > hit_factor_opp:
                next_state.set_initiative(InitiativeState.MY)
            elif hit_factor_opp > hit_factor_my:
                next_state.set_initiative(InitiativeState.OPP)
            else:
                next_state.set_initiative(InitiativeState.NEUTRAL)

            next_state.reset_both_control()

            next_state.my.set_last_action(my_attack["code"], my_attack["name"])
            next_state.opp.set_last_action(opp_attack["code"], opp_attack["name"])

            reward_my += self._reward_from_effect(
                damage_to_opp=eff_my["damage"] * hit_factor_my,
                stability_loss_to_opp=eff_my["stability_loss"] * hit_factor_my,
                energy_cost_self=eff_my["energy_cost"],
                self_exposure=self_exposure_my,
                control_gain=0.0,
            )
            reward_opp += self._reward_from_effect(
                damage_to_opp=eff_opp["damage"] * hit_factor_opp,
                stability_loss_to_opp=eff_opp["stability_loss"] * hit_factor_opp,
                energy_cost_self=eff_opp["energy_cost"],
                self_exposure=self_exposure_opp,
                control_gain=0.0,
            )

            info.update({
                "case": "my_attack_vs_opp_attack",
                "p_hit_my": p_hit_my,
                "p_hit_opp": p_hit_opp,
                "hit_my": hit_my,
                "hit_opp": hit_opp,
                "my_attack_name": my_attack["name"],
                "opp_attack_name": opp_attack["name"],
            })

            dt = max(
                self._estimate_action_duration(my_decision, my_attack, None),
                self._estimate_action_duration(opp_decision, opp_attack, None),
            )
            next_state.advance_time(dt)

        # -------------------------------------------------
        # Case 4: 双防 / hold / recover
        # -------------------------------------------------
        else:
            # 保守恢复逻辑
            recover_my = 2.5 if my_decision.action_type in ["defend", "recover", "hold"] else 0.0
            recover_opp = 2.5 if opp_decision.action_type in ["defend", "recover", "hold"] else 0.0

            next_state.my.apply_stability_change(+recover_my)
            next_state.opp.apply_stability_change(+recover_opp)

            next_state.my.apply_energy_change(+0.8 if my_decision.action_type == "recover" else +0.2)
            next_state.opp.apply_energy_change(+0.8 if opp_decision.action_type == "recover" else +0.2)

            next_state.reset_both_control()
            next_state.set_initiative(InitiativeState.NEUTRAL)

            info.update({
                "case": "neutral_or_recovery",
                "my_action_type": my_decision.action_type,
                "opp_action_type": opp_decision.action_type,
            })

            dt = max(
                self._estimate_action_duration(my_decision, None, my_defense),
                self._estimate_action_duration(opp_decision, None, opp_defense),
                0.5,
            )
            next_state.advance_time(dt)

        # -------------------------------------------------
        # 十一、姿态与终局检查
        # -------------------------------------------------
        # 若稳定性极低，则进入失衡/倒地
        self._update_posture_by_state(next_state.my)
        self._update_posture_by_state(next_state.opp)

        winner = next_state.check_ko_like_terminal()
        if winner == "draw":
            next_state.finish_round("draw", WinReason.DOUBLE_KO.value)
        elif winner is not None:
            next_state.finish_round(winner, "ko_like")

        return TransitionResult(
            next_state=next_state,
            reward_my=reward_my,
            reward_opp=reward_opp,
            info=info,
        )

    # =====================================================
    # 十二、姿态更新
    # =====================================================
    def _update_posture_by_state(self, fighter: FighterState) -> None:
        if fighter.posture == PostureState.RECOVERING:
            return

        if fighter.stability <= 10:
            fighter.set_posture(PostureState.DOWNED)
        elif fighter.stability <= 35:
            fighter.set_posture(PostureState.OFF_BALANCE)
        else:
            fighter.set_posture(PostureState.STANDING)


# =========================================================
# 十三、自测
# =========================================================
if __name__ == "__main__":
    from models.state import create_initial_round_state

    engine = TransitionEngine()
    state = create_initial_round_state()

    my_decision = ActionDecision(action_type="attack", action_key="A06")   # 侧踢
    opp_decision = ActionDecision(action_type="defend", action_key="D09")  # 转身闪避

    result = engine.step(state, my_decision, opp_decision, mode="expected")

    print("transition.py 自测开始")
    print("case:", result.info.get("case"))
    print("我方奖励:", round(result.reward_my, 3))
    print("对方奖励:", round(result.reward_opp, 3))
    print("剩余时间:", round(result.next_state.time_left_sec, 3))
    print("我方能量:", round(result.next_state.my.energy, 3))
    print("对方稳定性:", round(result.next_state.opp.stability, 3))
    print("当前主动权:", result.next_state.initiative.value)
    print("transition.py 自测完成")