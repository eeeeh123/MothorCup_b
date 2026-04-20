from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from config.rules import get_modeling_rule_subset
from models.state import RoundState, FighterState, DistanceState, InitiativeState, PostureState, WinReason
from models.matchup_matrix import prepare_attack_records, prepare_defense_records, build_matchup_matrix

RULE = get_modeling_rule_subset()

@dataclass
class ActionDecision:
    action_type: str
    action_key: Optional[str] = None

@dataclass
class TransitionResult:
    next_state: RoundState
    reward_my: float
    reward_opp: float
    info: Dict[str, Any]

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

def _positive_delta(before: float, after: float) -> float:
    return max(0.0, before - after)

class TransitionEngine:
    def __init__(self) -> None:
        self.attack_records = prepare_attack_records()
        self.defense_records = prepare_defense_records()
        self.attack_map = self._build_attack_map(self.attack_records)
        self.defense_map = self._build_defense_map(self.defense_records)
        matchup_df = build_matchup_matrix(self.attack_records, self.defense_records)
        self.matchup_lookup = self._build_matchup_lookup(matchup_df)

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
            lookup[(row["attack_name"], row["defense_name"])] = {"score": _safe_float(row.get("matchup_score", 0.0)), "reason": row.get("reason", "")}
        return lookup

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
        return _safe_float(item["score"], 0.0) if item is not None else 0.0

    def _attack_feature_pack(self, attack: Dict[str, Any]) -> Dict[str, float]:
        force = _safe_float(attack.get("F", attack.get("avg_impact_force", 0.0)))
        ek = _safe_float(attack.get("Ek", attack.get("kinetic_energy", 0.0)))
        pre = _safe_float(attack.get("Tpre", attack.get("time_cost", 0.0)))
        reach = _safe_float(attack.get("Lreach", 0.0))
        sweep = _safe_float(attack.get("theta_sweep", attack.get("θsweep", 0.0)))
        kbreak = _safe_float(attack.get("Kbreak", attack.get("balance_break_level", 0.0)))
        combo = _safe_float(attack.get("Ccombo", attack.get("combo_potential", 0.0)))
        vuln = _safe_float(attack.get("Vvul", attack.get("counter_risk", 0.0)))
        pfall = _safe_float(attack.get("Pfall", attack.get("p_fall", attack.get("balance_risk", 0.0))))
        trec = _safe_float(attack.get("Trec", attack.get("recovery_time", 0.0)))
        cstam = _safe_float(attack.get("Cstam", attack.get("stamina_cost_raw", attack.get("energy_cost", 0.0))))
        return {"force_n": _normalize(force, 0.0, 7500.0), "ek_n": _normalize(ek, 0.0, 800.0), "pre_n": _normalize(pre, 0.10, 0.80), "reach_n": _normalize(reach, 0.20, 1.00), "sweep_n": _normalize(sweep, 0.0, 3.20), "kbreak_n": _normalize(kbreak, 0.0, 1.0 if kbreak <= 1.0 else 10.0), "combo_n": _normalize(combo, 0.0, 1.0 if combo <= 1.0 else 10.0), "vuln_n": _normalize(vuln, 0.0, 1.0 if vuln <= 1.0 else 10.0), "pfall_n": _normalize(pfall, 0.0, 0.50 if pfall <= 1.0 else 10.0), "trec_n": _normalize(trec, 0.0, 2.5), "cstam_n": _normalize(cstam, 0.0, 1000.0), "pre_raw": pre, "trec_raw": trec}

    def _defense_feature_pack(self, defense: Dict[str, Any]) -> Dict[str, float]:
        direct = _safe_float(defense.get("direct_defense_level", defense.get("防护量级", 0.0)))
        stability = _safe_float(defense.get("stability_recovery", 0.0))
        counter = _safe_float(defense.get("counter_attack_potential", defense.get("反击潜力加成", 0.0)))
        timing = _safe_float(defense.get("timing_difficulty", 0.0))
        risk = _safe_float(defense.get("risk_if_failed", defense.get("失误惩罚", 0.0)))
        energy = _safe_float(defense.get("energy_cost", defense.get("能耗", 0.0)))
        mobility = _safe_float(defense.get("mobility_loss", 0.0))
        recovery = _safe_float(defense.get("recovery_time", 0.0))
        return {"direct_n": _normalize(direct, 0.0, 5.0), "stability_n": _normalize(stability, 0.0, 5.0), "counter_n": _normalize(counter, 0.0, 5.0), "timing_n": _normalize(timing, 0.0, 5.0), "risk_n": _normalize(risk, 0.0, 5.0), "energy_n": _normalize(energy, 0.0, 5.0), "mobility_n": _normalize(mobility, 0.0, 5.0), "recovery_n": _normalize(recovery, 0.0, 3.0), "recovery_raw": recovery}

    def _distance_bonus(self, attack: Dict[str, Any], state: RoundState, side: str) -> float:
        attack_range = attack.get("attack_range", "near-mid")
        current_distance = state.distance.value
        table = {("near", "clinch"): 1.0, ("near", "near"): 0.9, ("near-mid", "near"): 1.0, ("near-mid", "mid"): 0.9, ("mid", "mid"): 0.92, ("mid-far", "mid"): 0.86, ("mid-far", "far"): 0.95, ("far", "far"): 1.0}
        return table.get((attack_range, current_distance), 0.45)

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

    def _attack_quality(self, attack: Dict[str, Any], state: RoundState, side: str) -> float:
        a = self._attack_feature_pack(attack)
        return 0.22 * a["force_n"] + 0.18 * a["ek_n"] + 0.14 * a["kbreak_n"] + 0.08 * a["sweep_n"] + 0.08 * a["combo_n"] - 0.12 * a["pre_n"] - 0.12 * a["vuln_n"] - 0.10 * a["pfall_n"] + 0.10 * self._distance_bonus(attack, state, side) + self._initiative_bonus(state, side)

    def _defense_quality(self, attack: Dict[str, Any], defense: Dict[str, Any]) -> float:
        d = self._defense_feature_pack(defense)
        matchup_n = _normalize(self.get_matchup_score(attack["name"], defense["name"]), 0.0, 100.0)
        return 0.32 * matchup_n + 0.24 * d["direct_n"] + 0.16 * d["stability_n"] + 0.10 * d["counter_n"] - 0.08 * d["timing_n"] - 0.06 * d["risk_n"] - 0.04 * d["energy_n"]

    def _hit_probability_attack_vs_defense(self, attack: Dict[str, Any], defense: Dict[str, Any], state: RoundState, attacker_side: str) -> float:
        return _clamp(_sigmoid(4.1 * (self._attack_quality(attack, state, attacker_side) - self._defense_quality(attack, defense))), 0.05, 0.93)

    def _hit_probability_attack_vs_attack(self, attack_a: Dict[str, Any], attack_b: Dict[str, Any], state: RoundState) -> Tuple[float, float]:
        diff = self._attack_quality(attack_a, state, "my") - self._attack_quality(attack_b, state, "opp")
        return _clamp(_sigmoid(3.0 * diff), 0.10, 0.88), _clamp(_sigmoid(-3.0 * diff), 0.10, 0.88)

    def _damage_and_stability_effect(self, attack: Dict[str, Any]) -> Dict[str, float]:
        a = self._attack_feature_pack(attack)
        return {"damage": 12.0 * a["force_n"] + 10.0 * a["ek_n"] + 7.0 * a["kbreak_n"], "stability_loss": 13.0 * a["kbreak_n"] + 8.0 * a["force_n"] + 5.0 * a["sweep_n"], "base_exposure": 12.0 * a["vuln_n"] + 12.0 * a["pfall_n"], "energy_cost": 8.0 * a["cstam_n"] + 3.0 * a["pre_n"] + 3.0 * a["trec_n"]}

    def _estimate_action_duration(self, decision: ActionDecision, action_record: Optional[Dict[str, Any]], defense_record: Optional[Dict[str, Any]]) -> float:
        if decision.action_type == "attack" and action_record is not None:
            a = self._attack_feature_pack(action_record)
            return max(0.5, a["pre_raw"] + 0.35 * max(a["trec_raw"], 0.2))
        if decision.action_type == "defend" and defense_record is not None:
            d = self._defense_feature_pack(defense_record)
            if d["recovery_raw"] > 0:
                return max(0.4, 0.20 + d["recovery_raw"])
            return max(0.4, 0.25 + 0.20 * _safe_float(defense_record.get("timing_difficulty", 2.0)))
        if decision.action_type == "recover":
            return 0.8
        return 0.5

    @staticmethod
    def _fighter_snapshot(f: FighterState) -> Dict[str, Any]:
        return {"hp": f.hp_proxy, "stability": f.stability, "energy": f.energy, "posture": f.posture}

    @staticmethod
    def _posture_rank(posture: PostureState) -> int:
        if posture == PostureState.STANDING:
            return 0
        if posture == PostureState.OFF_BALANCE:
            return 1
        if posture in (PostureState.DOWNED, PostureState.RECOVERING):
            return 2
        return 0

    def _initiative_advantage(self, next_state: RoundState) -> float:
        if next_state.initiative == InitiativeState.MY:
            return 1.0
        if next_state.initiative == InitiativeState.OPP:
            return -1.0
        return 0.0

    def _terminal_bonus(self, next_state: RoundState) -> float:
        if not next_state.is_finished():
            return 0.0
        if next_state.winner == "my":
            return 80.0
        if next_state.winner == "opp":
            return -80.0
        return 0.0

    def _zero_sum_stage_payoff(self, state: RoundState, next_state: RoundState, before_my: Dict[str, Any], before_opp: Dict[str, Any]) -> float:
        after_my = self._fighter_snapshot(next_state.my); after_opp = self._fighter_snapshot(next_state.opp)
        my_hp_loss = _positive_delta(before_my["hp"], after_my["hp"]); opp_hp_loss = _positive_delta(before_opp["hp"], after_opp["hp"])
        my_stability_loss = _positive_delta(before_my["stability"], after_my["stability"]); opp_stability_loss = _positive_delta(before_opp["stability"], after_opp["stability"])
        my_energy_loss = _positive_delta(before_my["energy"], after_my["energy"]); opp_energy_loss = _positive_delta(before_opp["energy"], after_opp["energy"])
        my_posture_worsen = self._posture_rank(after_my["posture"]) - self._posture_rank(before_my["posture"]); opp_posture_worsen = self._posture_rank(after_opp["posture"]) - self._posture_rank(before_opp["posture"])
        return 1.00 * (opp_hp_loss - my_hp_loss) + 0.70 * (opp_stability_loss - my_stability_loss) + 0.18 * (opp_energy_loss - my_energy_loss) + 6.00 * (opp_posture_worsen - my_posture_worsen) + 4.00 * self._initiative_advantage(next_state) + self._terminal_bonus(next_state)

    def _resolve_single_side_decision(self, decision: ActionDecision) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        attack_record = self.get_attack(decision.action_key) if decision.action_type == "attack" else None
        defense_record = self.get_defense(decision.action_key) if decision.action_type == "defend" else None
        return attack_record, defense_record

    def _decision_distance_bias(self, decision: ActionDecision) -> float:
        if decision.action_type == "attack":
            atk = self.get_attack(decision.action_key)
            return _safe_float(atk.get("distance_bias", 0.0), 0.0) if atk else 0.0
        if decision.action_type == "defend":
            dfd = self.get_defense(decision.action_key)
            return _safe_float(dfd.get("distance_bias", 0.0), 0.0) if dfd else 0.0
        if decision.action_type == "hold":
            return -0.10
        if decision.action_type == "recover":
            return -0.20
        return 0.0

    @staticmethod
    def _distance_order(distance: DistanceState) -> int:
        return {DistanceState.FAR: 0, DistanceState.MID: 1, DistanceState.NEAR: 2, DistanceState.CLINCH: 3}[distance]

    @staticmethod
    def _distance_from_index(idx: int) -> DistanceState:
        return {0: DistanceState.FAR, 1: DistanceState.MID, 2: DistanceState.NEAR, 3: DistanceState.CLINCH}[max(0, min(3, idx))]

    def _distance_transition_probs(self, current: DistanceState, aggregate_bias: float) -> Dict[DistanceState, float]:
        if current == DistanceState.FAR:
            base = {DistanceState.FAR: 0.55, DistanceState.MID: 0.35, DistanceState.NEAR: 0.10, DistanceState.CLINCH: 0.00}
        elif current == DistanceState.MID:
            base = {DistanceState.FAR: 0.20, DistanceState.MID: 0.50, DistanceState.NEAR: 0.25, DistanceState.CLINCH: 0.05}
        elif current == DistanceState.NEAR:
            base = {DistanceState.FAR: 0.05, DistanceState.MID: 0.25, DistanceState.NEAR: 0.50, DistanceState.CLINCH: 0.20}
        else:
            base = {DistanceState.FAR: 0.00, DistanceState.MID: 0.10, DistanceState.NEAR: 0.35, DistanceState.CLINCH: 0.55}
        shift = _clamp(aggregate_bias / 3.0, -0.35, 0.35)
        probs = dict(base)
        if shift > 0:
            for d in list(probs.keys()):
                idx = self._distance_order(d)
                if idx < 3 and probs[d] > 0:
                    moved = min(probs[d] * shift, probs[d]); probs[d] -= moved; probs[self._distance_from_index(idx + 1)] += moved
        elif shift < 0:
            s = abs(shift)
            for d in list(probs.keys()):
                idx = self._distance_order(d)
                if idx > 0 and probs[d] > 0:
                    moved = min(probs[d] * s, probs[d]); probs[d] -= moved; probs[self._distance_from_index(idx - 1)] += moved
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()} if total > 1e-12 else base

    def _update_distance_by_joint_actions(self, state: RoundState, my_decision: ActionDecision, opp_decision: ActionDecision, info: Dict[str, Any], rng: random.Random, mode: str) -> None:
        my_bias = self._decision_distance_bias(my_decision)
        opp_bias = self._decision_distance_bias(opp_decision)
        my_hit = _safe_float(info.get("my_hit_factor", 0.0), 0.0)
        opp_hit = _safe_float(info.get("opp_hit_factor", 0.0), 0.0)
        if my_decision.action_type == "attack":
            my_bias += 0.35 * my_hit * max(0.0, my_bias)
        if opp_decision.action_type == "attack":
            opp_bias += 0.35 * opp_hit * max(0.0, opp_bias)
        if my_decision.action_type == "defend" and my_bias < 0:
            my_bias -= 0.20 * (1.0 - opp_hit)
        if opp_decision.action_type == "defend" and opp_bias < 0:
            opp_bias -= 0.20 * (1.0 - my_hit)
        aggregate_bias = my_bias + opp_bias
        if state.initiative == InitiativeState.MY:
            aggregate_bias += 0.12
        elif state.initiative == InitiativeState.OPP:
            aggregate_bias -= 0.12
        probs = self._distance_transition_probs(state.distance, aggregate_bias)
        info["distance_transition_probs"] = {k.value: round(v, 4) for k, v in probs.items()}
        info["aggregate_distance_bias"] = round(aggregate_bias, 4)
        if mode == "sample":
            r = rng.random(); cum = 0.0; chosen = list(probs.keys())[-1]
            for d, p in probs.items():
                cum += p
                if r <= cum:
                    chosen = d; break
            state.set_distance(chosen)
        else:
            state.set_distance(max(probs.items(), key=lambda x: x[1])[0])

    def step(self, state: RoundState, my_decision: ActionDecision, opp_decision: ActionDecision, mode: str = "expected", seed: Optional[int] = None) -> TransitionResult:
        if state.is_finished():
            return TransitionResult(next_state=state.clone(), reward_my=0.0, reward_opp=0.0, info={"message": "state already finished"})
        rng = random.Random(seed) if seed is not None else random
        next_state = state.clone(); next_state.next_step()
        before_my = self._fighter_snapshot(state.my); before_opp = self._fighter_snapshot(state.opp)
        my_attack, my_defense = self._resolve_single_side_decision(my_decision)
        opp_attack, opp_defense = self._resolve_single_side_decision(opp_decision)
        info: Dict[str, Any] = {"my_decision": my_decision.action_type, "opp_decision": opp_decision.action_type, "my_hit_factor": 0.0, "opp_hit_factor": 0.0}

        if my_attack is not None and opp_defense is not None:
            p_hit = self._hit_probability_attack_vs_defense(my_attack, opp_defense, next_state, "my"); eff = self._damage_and_stability_effect(my_attack)
            if mode == "sample":
                hit = rng.random() < p_hit; hit_factor = 1.0 if hit else 0.0; self_exposure = (0.0 if hit else 1.0) * eff["base_exposure"]
            else:
                hit = None; hit_factor = p_hit; self_exposure = (1.0 - p_hit) * eff["base_exposure"]
            info["my_hit_factor"] = hit_factor
            next_state.opp.apply_hp_change(-eff["damage"] * hit_factor); next_state.opp.apply_stability_change(-eff["stability_loss"] * hit_factor)
            next_state.my.apply_stability_change(-self_exposure); next_state.my.apply_energy_change(-eff["energy_cost"])
            if hit_factor > 0.6 and next_state.distance in [DistanceState.NEAR, DistanceState.CLINCH]:
                next_state.register_control("my", min(1.0, RULE["control_check_step_sec"]))
            else:
                next_state.reset_both_control()
            next_state.set_initiative(InitiativeState.MY if hit_factor >= 0.5 else InitiativeState.OPP)
            next_state.my.set_last_action(my_attack["code"], my_attack["name"]); next_state.opp.set_last_action(opp_defense["code"], opp_defense["name"])
            info.update({"case": "my_attack_vs_opp_defense", "p_hit_my": p_hit, "hit_my": hit})
            dt = max(self._estimate_action_duration(my_decision, my_attack, None), self._estimate_action_duration(opp_decision, None, opp_defense)); next_state.advance_time(dt)

        elif my_defense is not None and opp_attack is not None:
            p_hit_opp = self._hit_probability_attack_vs_defense(opp_attack, my_defense, next_state, "opp"); eff = self._damage_and_stability_effect(opp_attack)
            if mode == "sample":
                hit = rng.random() < p_hit_opp; hit_factor = 1.0 if hit else 0.0; self_exposure = (0.0 if hit else 1.0) * eff["base_exposure"]
            else:
                hit = None; hit_factor = p_hit_opp; self_exposure = (1.0 - p_hit_opp) * eff["base_exposure"]
            info["opp_hit_factor"] = hit_factor
            next_state.my.apply_hp_change(-eff["damage"] * hit_factor); next_state.my.apply_stability_change(-eff["stability_loss"] * hit_factor)
            next_state.opp.apply_stability_change(-self_exposure); next_state.opp.apply_energy_change(-eff["energy_cost"])
            if hit_factor > 0.6 and next_state.distance in [DistanceState.NEAR, DistanceState.CLINCH]:
                next_state.register_control("opp", min(1.0, RULE["control_check_step_sec"]))
            else:
                next_state.reset_both_control()
            next_state.set_initiative(InitiativeState.OPP if hit_factor >= 0.5 else InitiativeState.MY)
            next_state.my.set_last_action(my_defense["code"], my_defense["name"]); next_state.opp.set_last_action(opp_attack["code"], opp_attack["name"])
            info.update({"case": "my_defense_vs_opp_attack", "p_hit_opp": p_hit_opp, "hit_opp": hit})
            dt = max(self._estimate_action_duration(my_decision, None, my_defense), self._estimate_action_duration(opp_decision, opp_attack, None)); next_state.advance_time(dt)

        elif my_attack is not None and opp_attack is not None:
            p_hit_my, p_hit_opp = self._hit_probability_attack_vs_attack(my_attack, opp_attack, next_state)
            eff_my = self._damage_and_stability_effect(my_attack); eff_opp = self._damage_and_stability_effect(opp_attack)
            if mode == "sample":
                hit_my = rng.random() < p_hit_my; hit_opp = rng.random() < p_hit_opp
                hit_factor_my = 1.0 if hit_my else 0.0; hit_factor_opp = 1.0 if hit_opp else 0.0
                self_exposure_my = (0.0 if hit_my else 1.0) * eff_my["base_exposure"]; self_exposure_opp = (0.0 if hit_opp else 1.0) * eff_opp["base_exposure"]
            else:
                hit_my = None; hit_opp = None; hit_factor_my = p_hit_my; hit_factor_opp = p_hit_opp
                self_exposure_my = (1.0 - p_hit_my) * eff_my["base_exposure"]; self_exposure_opp = (1.0 - p_hit_opp) * eff_opp["base_exposure"]
            info["my_hit_factor"] = hit_factor_my; info["opp_hit_factor"] = hit_factor_opp
            next_state.opp.apply_hp_change(-eff_my["damage"] * hit_factor_my); next_state.opp.apply_stability_change(-eff_my["stability_loss"] * hit_factor_my)
            next_state.my.apply_hp_change(-eff_opp["damage"] * hit_factor_opp); next_state.my.apply_stability_change(-eff_opp["stability_loss"] * hit_factor_opp)
            next_state.my.apply_energy_change(-eff_my["energy_cost"]); next_state.opp.apply_energy_change(-eff_opp["energy_cost"])
            next_state.my.apply_stability_change(-(self_exposure_my + 0.5 * self_exposure_opp)); next_state.opp.apply_stability_change(-(self_exposure_opp + 0.5 * self_exposure_my))
            next_state.set_initiative(InitiativeState.MY if hit_factor_my > hit_factor_opp else InitiativeState.OPP if hit_factor_opp > hit_factor_my else InitiativeState.NEUTRAL)
            next_state.reset_both_control()
            next_state.my.set_last_action(my_attack["code"], my_attack["name"]); next_state.opp.set_last_action(opp_attack["code"], opp_attack["name"])
            info.update({"case": "my_attack_vs_opp_attack", "p_hit_my": p_hit_my, "p_hit_opp": p_hit_opp, "hit_my": hit_my, "hit_opp": hit_opp})
            dt = max(self._estimate_action_duration(my_decision, my_attack, None), self._estimate_action_duration(opp_decision, opp_attack, None)); next_state.advance_time(dt)

        elif my_attack is not None and opp_decision.action_type in ["hold", "recover"]:
            eff = self._damage_and_stability_effect(my_attack)
            base_p_hit = 0.80 if opp_decision.action_type == "hold" else 0.88
            p_hit = _clamp(base_p_hit + 0.06 * (self._attack_quality(my_attack, next_state, "my") - 0.5), 0.62, 0.94)
            if mode == "sample":
                hit = rng.random() < p_hit; hit_factor = 1.0 if hit else 0.0; self_exposure = (0.0 if hit else 1.0) * eff["base_exposure"]
            else:
                hit = None; hit_factor = p_hit; self_exposure = (1.0 - p_hit) * eff["base_exposure"]
            info["my_hit_factor"] = hit_factor
            vuln_mul = 1.04 if opp_decision.action_type == "recover" else 1.00
            next_state.opp.apply_hp_change(-eff["damage"] * hit_factor * vuln_mul); next_state.opp.apply_stability_change(-eff["stability_loss"] * hit_factor * vuln_mul)
            next_state.my.apply_stability_change(-self_exposure); next_state.my.apply_energy_change(-eff["energy_cost"])
            if opp_decision.action_type == "recover":
                recover_gain = 0.0 if (mode == "sample" and hit) else (1.0 - p_hit) * 0.9 if mode == "expected" else 0.9
                next_state.opp.apply_energy_change(+recover_gain); next_state.opp.apply_stability_change(+0.35 * recover_gain)
            next_state.set_initiative(InitiativeState.MY if hit_factor >= 0.5 else InitiativeState.NEUTRAL)
            next_state.reset_both_control()
            next_state.my.set_last_action(my_attack["code"], my_attack["name"]); next_state.opp.set_last_action(None, opp_decision.action_type)
            info.update({"case": "my_attack_vs_opp_hold_or_recover", "p_hit_my": p_hit, "hit_my": hit})
            dt = max(self._estimate_action_duration(my_decision, my_attack, None), self._estimate_action_duration(opp_decision, None, None)); next_state.advance_time(dt)

        elif my_decision.action_type in ["hold", "recover"] and opp_attack is not None:
            eff = self._damage_and_stability_effect(opp_attack)
            base_p_hit_opp = 0.80 if my_decision.action_type == "hold" else 0.88
            p_hit_opp = _clamp(base_p_hit_opp + 0.06 * (self._attack_quality(opp_attack, next_state, "opp") - 0.5), 0.62, 0.94)
            if mode == "sample":
                hit = rng.random() < p_hit_opp; hit_factor = 1.0 if hit else 0.0; self_exposure_opp = (0.0 if hit else 1.0) * eff["base_exposure"]
            else:
                hit = None; hit_factor = p_hit_opp; self_exposure_opp = (1.0 - p_hit_opp) * eff["base_exposure"]
            info["opp_hit_factor"] = hit_factor
            vuln_mul = 1.04 if my_decision.action_type == "recover" else 1.00
            next_state.my.apply_hp_change(-eff["damage"] * hit_factor * vuln_mul); next_state.my.apply_stability_change(-eff["stability_loss"] * hit_factor * vuln_mul)
            next_state.opp.apply_stability_change(-self_exposure_opp); next_state.opp.apply_energy_change(-eff["energy_cost"])
            if my_decision.action_type == "recover":
                recover_gain = 0.0 if (mode == "sample" and hit) else (1.0 - p_hit_opp) * 0.9 if mode == "expected" else 0.9
                next_state.my.apply_energy_change(+recover_gain); next_state.my.apply_stability_change(+0.35 * recover_gain)
            next_state.set_initiative(InitiativeState.OPP if hit_factor >= 0.5 else InitiativeState.NEUTRAL)
            next_state.reset_both_control()
            next_state.my.set_last_action(None, my_decision.action_type); next_state.opp.set_last_action(opp_attack["code"], opp_attack["name"])
            info.update({"case": "my_hold_or_recover_vs_opp_attack", "p_hit_opp": p_hit_opp, "hit_opp": hit})
            dt = max(self._estimate_action_duration(my_decision, None, None), self._estimate_action_duration(opp_decision, opp_attack, None)); next_state.advance_time(dt)

        else:
            my_type = my_decision.action_type; opp_type = opp_decision.action_type
            recover_my = 1.0 if my_type == "defend" else 0.4 if my_type == "hold" else 1.2 if my_type == "recover" else 0.0
            recover_opp = 1.0 if opp_type == "defend" else 0.4 if opp_type == "hold" else 1.2 if opp_type == "recover" else 0.0
            energy_gain_my = 0.08 if my_type in ["defend", "hold"] else 0.40 if my_type == "recover" else 0.0
            energy_gain_opp = 0.08 if opp_type in ["defend", "hold"] else 0.40 if opp_type == "recover" else 0.0
            next_state.my.apply_stability_change(+recover_my); next_state.opp.apply_stability_change(+recover_opp)
            next_state.my.apply_energy_change(+energy_gain_my); next_state.opp.apply_energy_change(+energy_gain_opp)
            next_state.reset_both_control(); next_state.set_initiative(InitiativeState.NEUTRAL)
            next_state.my.set_last_action(None, my_type); next_state.opp.set_last_action(None, opp_type)
            info.update({"case": "low_interaction_or_recovery", "my_action_type": my_type, "opp_action_type": opp_type})
            dt = max(self._estimate_action_duration(my_decision, None, my_defense), self._estimate_action_duration(opp_decision, None, opp_defense), 0.5); next_state.advance_time(dt)

        self._update_distance_by_joint_actions(next_state, my_decision, opp_decision, info, rng, mode)
        static_power_equiv = 0.15 * max(dt, 0.5)
        next_state.my.apply_energy_change(-static_power_equiv); next_state.opp.apply_energy_change(-static_power_equiv)
        self._update_posture_by_state(next_state.my); self._update_posture_by_state(next_state.opp)
        winner = next_state.check_ko_like_terminal()
        if winner == "draw":
            next_state.finish_round("draw", WinReason.DOUBLE_KO.value)
        elif winner is not None:
            next_state.finish_round(winner, "ko_like")
        reward_my = self._zero_sum_stage_payoff(state, next_state, before_my, before_opp)
        reward_opp = -reward_my
        info["stage_payoff_my"] = reward_my; info["stage_payoff_opp"] = reward_opp; info["distance_after_step"] = next_state.distance.value
        return TransitionResult(next_state=next_state, reward_my=reward_my, reward_opp=reward_opp, info=info)

    def _update_posture_by_state(self, fighter: FighterState) -> None:
        if fighter.posture == PostureState.RECOVERING:
            return
        if fighter.stability <= 10:
            fighter.set_posture(PostureState.DOWNED)
        elif fighter.stability <= 35:
            fighter.set_posture(PostureState.OFF_BALANCE)
        else:
            fighter.set_posture(PostureState.STANDING)
