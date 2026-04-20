from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

ATTACK_ACTION_LIBRARY: Dict[str, Dict] = {
    "A01": {"code": "A01", "name": "直拳", "attack_range": "near", "F": 1134.0, "Ek": 54.0, "Tpre": 0.18, "Lreach": 0.34, "theta_sweep": 0.25, "Kbreak": 0.26, "Ccombo": 0.30, "Vvul": 0.16, "Pfall": 0.02, "Trec": 0.28, "Cstam": 150.0, "accuracy_level": 4.4, "pressure_potential": 2.2, "distance_bias": 0.20},
    "A02": {"code": "A02", "name": "勾拳", "attack_range": "near", "F": 771.0, "Ek": 25.0, "Tpre": 0.21, "Lreach": 0.28, "theta_sweep": 0.45, "Kbreak": 0.24, "Ccombo": 0.34, "Vvul": 0.22, "Pfall": 0.05, "Trec": 0.36, "Cstam": 220.0, "accuracy_level": 3.9, "pressure_potential": 2.4, "distance_bias": 0.25},
    "A03": {"code": "A03", "name": "组合拳", "attack_range": "near", "F": 1480.0, "Ek": 78.0, "Tpre": 0.32, "Lreach": 0.36, "theta_sweep": 0.70, "Kbreak": 0.38, "Ccombo": 0.62, "Vvul": 0.28, "Pfall": 0.06, "Trec": 0.44, "Cstam": 290.0, "accuracy_level": 3.8, "pressure_potential": 3.1, "distance_bias": 0.35},
    "A04": {"code": "A04", "name": "摆拳", "attack_range": "near", "F": 980.0, "Ek": 42.0, "Tpre": 0.26, "Lreach": 0.33, "theta_sweep": 0.95, "Kbreak": 0.30, "Ccombo": 0.36, "Vvul": 0.26, "Pfall": 0.08, "Trec": 0.40, "Cstam": 260.0, "accuracy_level": 3.5, "pressure_potential": 2.6, "distance_bias": 0.20},
    "A05": {"code": "A05", "name": "前踢", "attack_range": "mid-far", "F": 3712.0, "Ek": 512.0, "Tpre": 0.36, "Lreach": 0.85, "theta_sweep": 0.52, "Kbreak": 0.42, "Ccombo": 0.28, "Vvul": 0.30, "Pfall": 0.15, "Trec": 0.56, "Cstam": 280.0, "accuracy_level": 4.1, "pressure_potential": 2.8, "distance_bias": -0.40},
    "A06": {"code": "A06", "name": "侧踢", "attack_range": "mid-far", "F": 7320.0, "Ek": 674.0, "Tpre": 0.42, "Lreach": 0.93, "theta_sweep": 0.88, "Kbreak": 0.58, "Ccombo": 0.30, "Vvul": 0.42, "Pfall": 0.25, "Trec": 0.70, "Cstam": 450.0, "accuracy_level": 3.9, "pressure_potential": 3.0, "distance_bias": -0.75},
    "A07": {"code": "A07", "name": "回旋踢", "attack_range": "mid", "F": 6680.0, "Ek": 610.0, "Tpre": 0.58, "Lreach": 0.82, "theta_sweep": 2.30, "Kbreak": 0.62, "Ccombo": 0.22, "Vvul": 0.58, "Pfall": 0.34, "Trec": 0.90, "Cstam": 520.0, "accuracy_level": 2.8, "pressure_potential": 3.2, "distance_bias": -0.25},
    "A08": {"code": "A08", "name": "低扫腿", "attack_range": "near-mid", "F": 2480.0, "Ek": 190.0, "Tpre": 0.29, "Lreach": 0.58, "theta_sweep": 1.10, "Kbreak": 0.55, "Ccombo": 0.34, "Vvul": 0.30, "Pfall": 0.18, "Trec": 0.52, "Cstam": 320.0, "accuracy_level": 3.7, "pressure_potential": 2.9, "distance_bias": 0.30},
    "A09": {"code": "A09", "name": "膝撞", "attack_range": "near", "F": 1146.0, "Ek": 87.0, "Tpre": 0.24, "Lreach": 0.22, "theta_sweep": 0.18, "Kbreak": 0.48, "Ccombo": 0.26, "Vvul": 0.24, "Pfall": 0.15, "Trec": 0.44, "Cstam": 250.0, "accuracy_level": 4.0, "pressure_potential": 3.2, "distance_bias": 1.20},
    "A10": {"code": "A10", "name": "拳腿组合", "attack_range": "near-mid", "F": 2900.0, "Ek": 240.0, "Tpre": 0.40, "Lreach": 0.68, "theta_sweep": 0.92, "Kbreak": 0.44, "Ccombo": 0.70, "Vvul": 0.36, "Pfall": 0.22, "Trec": 0.64, "Cstam": 420.0, "accuracy_level": 3.5, "pressure_potential": 3.3, "distance_bias": -0.10},
    "A11": {"code": "A11", "name": "五连踢", "attack_range": "mid", "F": 4120.0, "Ek": 360.0, "Tpre": 0.64, "Lreach": 0.80, "theta_sweep": 1.40, "Kbreak": 0.54, "Ccombo": 0.92, "Vvul": 0.54, "Pfall": 0.30, "Trec": 1.00, "Cstam": 610.0, "accuracy_level": 2.9, "pressure_potential": 3.9, "distance_bias": -0.15},
    "A12": {"code": "A12", "name": "冲撞", "attack_range": "near", "F": 3260.0, "Ek": 210.0, "Tpre": 0.34, "Lreach": 0.30, "theta_sweep": 0.22, "Kbreak": 0.68, "Ccombo": 0.20, "Vvul": 0.48, "Pfall": 0.28, "Trec": 0.72, "Cstam": 520.0, "accuracy_level": 3.1, "pressure_potential": 4.1, "distance_bias": 1.60},
    "A13": {"code": "A13", "name": "倒地反击", "attack_range": "near", "F": 860.0, "Ek": 36.0, "Tpre": 0.45, "Lreach": 0.18, "theta_sweep": 0.15, "Kbreak": 0.18, "Ccombo": 0.10, "Vvul": 0.60, "Pfall": 0.40, "Trec": 1.10, "Cstam": 180.0, "accuracy_level": 2.5, "pressure_potential": 1.5, "distance_bias": 0.00},
}

COMPOSITE_PENALTY_MAP = {
    "A01": 1.0, "A02": 1.6, "A03": 2.6, "A04": 2.8, "A05": 2.8, "A06": 4.9, "A07": 5.0,
    "A08": 4.0, "A09": 2.4, "A10": 3.9, "A11": 4.7, "A12": 4.8, "A13": 5.0,
}

def _attach_legacy_fields(item: Dict) -> Dict:
    x = deepcopy(item)
    x["avg_impact_force"] = x["F"]
    x["kinetic_energy"] = x["Ek"]
    x["time_cost"] = x["Tpre"]
    x["balance_break_level"] = x["Kbreak"]
    x["combo_potential"] = x["Ccombo"]
    x["counter_risk"] = x["Vvul"]
    x["balance_risk"] = x["Pfall"]
    x["recovery_time"] = x["Trec"]
    x["energy_cost"] = x["Cstam"]
    x["stamina_cost_raw"] = x["Cstam"]
    return x

def get_attack_action(code_or_name: str) -> Optional[Dict]:
    for _, item in ATTACK_ACTION_LIBRARY.items():
        if code_or_name in (item["code"], item["name"]):
            return _attach_legacy_fields(item)
    return None

def get_attack_action_library() -> Dict[str, Dict]:
    return {k: _attach_legacy_fields(v) for k, v in ATTACK_ACTION_LIBRARY.items()}

def iter_attack_actions() -> List[Dict]:
    return [_attach_legacy_fields(v) for _, v in ATTACK_ACTION_LIBRARY.items()]

ACTION_LIBRARY = get_attack_action_library()
