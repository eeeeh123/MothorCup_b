from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

DEFENSE_ACTION_LIBRARY: Dict[str, Dict] = {
    "D01": {"code": "D01", "name": "格挡", "direct_defense_level": 3.8, "stability_recovery": 1.2, "counter_attack_potential": 1.0, "timing_difficulty": 2.6, "risk_if_failed": 2.5, "energy_cost": 1.0, "mobility_loss": 1.0, "recovery_time": 0.18, "effective_ranges": ["near", "mid"], "distance_bias": 0.0},
    "D02": {"code": "D02", "name": "拍挡", "direct_defense_level": 3.4, "stability_recovery": 1.0, "counter_attack_potential": 1.6, "timing_difficulty": 3.0, "risk_if_failed": 2.8, "energy_cost": 1.0, "mobility_loss": 0.8, "recovery_time": 0.16, "effective_ranges": ["near"], "distance_bias": 0.0},
    "D03": {"code": "D03", "name": "肘挡", "direct_defense_level": 4.3, "stability_recovery": 1.5, "counter_attack_potential": 1.7, "timing_difficulty": 3.0, "risk_if_failed": 2.6, "energy_cost": 1.2, "mobility_loss": 1.2, "recovery_time": 0.22, "effective_ranges": ["near", "clinch"], "distance_bias": 0.1},
    "D04": {"code": "D04", "name": "架挡", "direct_defense_level": 4.0, "stability_recovery": 1.4, "counter_attack_potential": 0.8, "timing_difficulty": 2.7, "risk_if_failed": 2.4, "energy_cost": 1.1, "mobility_loss": 1.1, "recovery_time": 0.20, "effective_ranges": ["near", "mid"], "distance_bias": 0.0},
    "D05": {"code": "D05", "name": "外拨", "direct_defense_level": 3.5, "stability_recovery": 1.0, "counter_attack_potential": 1.5, "timing_difficulty": 3.1, "risk_if_failed": 2.9, "energy_cost": 1.0, "mobility_loss": 0.9, "recovery_time": 0.16, "effective_ranges": ["near", "mid"], "distance_bias": -0.1},
    "D06": {"code": "D06", "name": "格挡反击", "direct_defense_level": 3.8, "stability_recovery": 1.1, "counter_attack_potential": 2.4, "timing_difficulty": 3.5, "risk_if_failed": 3.2, "energy_cost": 1.5, "mobility_loss": 1.0, "recovery_time": 0.24, "effective_ranges": ["near"], "distance_bias": 0.0},
    "D07": {"code": "D07", "name": "内闪", "direct_defense_level": 2.8, "stability_recovery": 0.8, "counter_attack_potential": 2.0, "timing_difficulty": 3.4, "risk_if_failed": 3.6, "energy_cost": 1.2, "mobility_loss": 0.6, "recovery_time": 0.18, "effective_ranges": ["near"], "distance_bias": 0.2},
    "D08": {"code": "D08", "name": "后撤", "direct_defense_level": 3.0, "stability_recovery": 0.9, "counter_attack_potential": 0.5, "timing_difficulty": 2.2, "risk_if_failed": 2.3, "energy_cost": 1.1, "mobility_loss": 0.4, "recovery_time": 0.14, "effective_ranges": ["near", "mid", "far"], "distance_bias": -1.3},
    "D09": {"code": "D09", "name": "转身闪避", "direct_defense_level": 3.3, "stability_recovery": 1.0, "counter_attack_potential": 1.2, "timing_difficulty": 2.8, "risk_if_failed": 2.7, "energy_cost": 1.2, "mobility_loss": 0.6, "recovery_time": 0.16, "effective_ranges": ["mid", "far"], "distance_bias": -0.8},
    "D10": {"code": "D10", "name": "侧移", "direct_defense_level": 3.1, "stability_recovery": 0.9, "counter_attack_potential": 1.1, "timing_difficulty": 2.7, "risk_if_failed": 2.6, "energy_cost": 1.0, "mobility_loss": 0.5, "recovery_time": 0.15, "effective_ranges": ["mid", "far"], "distance_bias": -0.6},
    "D11": {"code": "D11", "name": "护头", "direct_defense_level": 3.9, "stability_recovery": 1.4, "counter_attack_potential": 0.7, "timing_difficulty": 2.4, "risk_if_failed": 2.3, "energy_cost": 0.9, "mobility_loss": 1.0, "recovery_time": 0.20, "effective_ranges": ["near", "mid"], "distance_bias": 0.0},
    "D12": {"code": "D12", "name": "沉身", "direct_defense_level": 3.6, "stability_recovery": 1.8, "counter_attack_potential": 1.2, "timing_difficulty": 2.5, "risk_if_failed": 2.7, "energy_cost": 1.0, "mobility_loss": 0.8, "recovery_time": 0.18, "effective_ranges": ["near", "clinch"], "distance_bias": 0.1},
    "D13": {"code": "D13", "name": "压臂", "direct_defense_level": 3.7, "stability_recovery": 1.5, "counter_attack_potential": 1.4, "timing_difficulty": 3.1, "risk_if_failed": 3.0, "energy_cost": 1.2, "mobility_loss": 0.9, "recovery_time": 0.22, "effective_ranges": ["near", "clinch"], "distance_bias": 0.2},
    "D14": {"code": "D14", "name": "重心补偿", "direct_defense_level": 2.9, "stability_recovery": 2.6, "counter_attack_potential": 0.4, "timing_difficulty": 1.8, "risk_if_failed": 1.8, "energy_cost": 0.8, "mobility_loss": 0.7, "recovery_time": 0.18, "effective_ranges": ["near", "mid", "far", "clinch"], "distance_bias": 0.0},
    "D15": {"code": "D15", "name": "卸力缓冲", "direct_defense_level": 3.4, "stability_recovery": 2.2, "counter_attack_potential": 0.8, "timing_difficulty": 2.1, "risk_if_failed": 2.1, "energy_cost": 0.9, "mobility_loss": 0.8, "recovery_time": 0.19, "effective_ranges": ["near", "mid", "far"], "distance_bias": -0.2},
    "D16": {"code": "D16", "name": "接触缠斗", "direct_defense_level": 3.2, "stability_recovery": 1.6, "counter_attack_potential": 1.1, "timing_difficulty": 3.2, "risk_if_failed": 3.1, "energy_cost": 1.4, "mobility_loss": 1.3, "recovery_time": 0.26, "effective_ranges": ["near", "clinch"], "distance_bias": 0.9},
    "D17": {"code": "D17", "name": "拍肩压近", "direct_defense_level": 2.8, "stability_recovery": 1.2, "counter_attack_potential": 1.5, "timing_difficulty": 3.0, "risk_if_failed": 3.2, "energy_cost": 1.3, "mobility_loss": 0.9, "recovery_time": 0.24, "effective_ranges": ["mid", "near"], "distance_bias": 0.7},
    "D18": {"code": "D18", "name": "卡位转角", "direct_defense_level": 3.1, "stability_recovery": 1.1, "counter_attack_potential": 1.3, "timing_difficulty": 2.9, "risk_if_failed": 2.8, "energy_cost": 1.1, "mobility_loss": 0.7, "recovery_time": 0.18, "effective_ranges": ["mid", "far"], "distance_bias": -0.7},
    "D19": {"code": "D19", "name": "抱架顶住", "direct_defense_level": 4.1, "stability_recovery": 1.7, "counter_attack_potential": 0.6, "timing_difficulty": 2.6, "risk_if_failed": 2.6, "energy_cost": 1.3, "mobility_loss": 1.4, "recovery_time": 0.24, "effective_ranges": ["near", "clinch"], "distance_bias": 0.3},
    "D20": {"code": "D20", "name": "闪挡反", "direct_defense_level": 3.5, "stability_recovery": 1.2, "counter_attack_potential": 2.1, "timing_difficulty": 3.4, "risk_if_failed": 3.2, "energy_cost": 1.5, "mobility_loss": 0.8, "recovery_time": 0.22, "effective_ranges": ["near"], "distance_bias": 0.0},
    "D21": {"code": "D21", "name": "侧身引导", "direct_defense_level": 3.0, "stability_recovery": 1.1, "counter_attack_potential": 1.7, "timing_difficulty": 3.0, "risk_if_failed": 2.9, "energy_cost": 1.2, "mobility_loss": 0.6, "recovery_time": 0.18, "effective_ranges": ["mid"], "distance_bias": -0.4},
    "D22": {"code": "D22", "name": "挡撤绕", "direct_defense_level": 3.7, "stability_recovery": 1.3, "counter_attack_potential": 1.0, "timing_difficulty": 2.5, "risk_if_failed": 2.4, "energy_cost": 1.1, "mobility_loss": 0.4, "recovery_time": 0.16, "effective_ranges": ["mid", "far"], "distance_bias": -1.6},
}

def _attach_legacy_fields(item: Dict) -> Dict:
    x = deepcopy(item)
    x["防护量级"] = x["direct_defense_level"]
    x["反击潜力加成"] = x["counter_attack_potential"]
    x["失误惩罚"] = x["risk_if_failed"]
    x["能耗"] = x["energy_cost"]
    return x

def get_defense_action(code_or_name: str) -> Optional[Dict]:
    for _, item in DEFENSE_ACTION_LIBRARY.items():
        if code_or_name in (item["code"], item["name"]):
            return _attach_legacy_fields(item)
    return None

def get_defense_action_library() -> Dict[str, Dict]:
    return {k: _attach_legacy_fields(v) for k, v in DEFENSE_ACTION_LIBRARY.items()}

def iter_defense_actions() -> List[Dict]:
    return [_attach_legacy_fields(v) for _, v in DEFENSE_ACTION_LIBRARY.items()]
