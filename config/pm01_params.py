from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple


@dataclass(frozen=True)
class PM01Params:
    """
    众擎 PM01（商业版）+ Q90H 电机参数（Q3/Q4 直接使用）

    说明：
    1. 本文件分为“官网参数”和“建模派生参数”两层。
    2. 官网参数尽量保持原始口径。
    3. 建模派生参数仅用于仿真与状态转移，不代表官方额外声明。
    """

    # =========================
    # 一、官网原始硬件参数
    # =========================
    mass_kg: float = 42.0
    height_m: float = 1.400
    width_m: float = 0.53555
    depth_m: float = 0.25266

    leg_length_m: float = 0.6865           # 小腿+大腿长度
    arm_span_m: float = 1.44

    dof_total: int = 23
    arm_dof_single: int = 5
    leg_dof_single: int = 6

    # 腰部页面存在两种表达：宣传页 320°，参数页 -230°~90°
    waist_rotation_deg_statement: int = 320
    waist_joint_range_deg: Tuple[int, int] = (-230, 90)

    hip_range_roll_deg: Tuple[int, int] = (-15, 130)
    hip_range_pitch_deg: Tuple[int, int] = (-180, 140)
    hip_range_yaw_deg: Tuple[int, int] = (-90, 230)
    knee_range_deg: Tuple[int, int] = (-30, 137)

    official_speed_statement: str = ">2 m/s"

    battery_mAh: int = 10000
    battery_life_hr: float = 2.0
    charge_voltage_v: float = 54.6
    charge_current_a: float = 4.5
    full_charge_time_hr: float = 2.0

    motor_model: str = "Q90H"
    motor_encoder_type: str = "双编码"
    motor_type: str = "全行星"
    motor_max_torque_Nm: float = 145.0
    motor_peak_torque_density_Nm_per_kg: float = 130.0
    motor_max_rpm: float = 6400.0

    # =========================
    # 二、建模保守取值 / 派生值
    # =========================
    speed_mps_conservative: float = 2.0
    battery_life_sec: int = 7200
    full_charge_time_sec: int = 7200

    # 仍沿用当前 Q1/Q3 常用等效参数，便于与现有项目兼容
    torso_mass_ratio: float = 0.50
    arm_mass_ratio: float = 0.05
    leg_mass_ratio: float = 0.20

    arm_length_equivalent_m: float = 0.50
    leg_length_equivalent_m: float = 0.70
    torso_buffer_length_m: float = 0.30

    # 基于质量比例的派生质量
    single_arm_mass_kg: float = 2.10       # 42 * 0.05
    single_leg_mass_kg: float = 8.40       # 42 * 0.20
    torso_mass_kg: float = 21.00           # 42 * 0.50

    # 供 Q3/Q4 状态机使用的保守阈值建议（建模值）
    stability_default: float = 100.0
    energy_default: float = 100.0
    hp_proxy_default: float = 100.0

    # 若后续把电池映射到动作能耗，这个参数可作为单位归一化基底
    battery_energy_proxy_default: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PM01_PARAMS = PM01Params()


def get_pm01_params() -> Dict[str, Any]:
    return PM01_PARAMS.to_dict()


def get_official_pm01_subset() -> Dict[str, Any]:
    """
    只返回官网明确给出的 PM01/Q90H 参数。
    """
    d = PM01_PARAMS.to_dict()
    keys = [
        "mass_kg",
        "height_m",
        "width_m",
        "depth_m",
        "leg_length_m",
        "arm_span_m",
        "dof_total",
        "arm_dof_single",
        "leg_dof_single",
        "waist_rotation_deg_statement",
        "waist_joint_range_deg",
        "hip_range_roll_deg",
        "hip_range_pitch_deg",
        "hip_range_yaw_deg",
        "knee_range_deg",
        "official_speed_statement",
        "battery_mAh",
        "battery_life_hr",
        "charge_voltage_v",
        "charge_current_a",
        "full_charge_time_hr",
        "motor_model",
        "motor_encoder_type",
        "motor_type",
        "motor_max_torque_Nm",
        "motor_peak_torque_density_Nm_per_kg",
        "motor_max_rpm",
    ]
    return {k: d[k] for k in keys}


def get_modeling_pm01_subset() -> Dict[str, Any]:
    """
    返回更适合当前项目 Q1/Q3/Q4 使用的参数集合。
    """
    d = PM01_PARAMS.to_dict()
    keys = [
        "mass_kg",
        "speed_mps_conservative",
        "battery_life_sec",
        "full_charge_time_sec",
        "torso_mass_ratio",
        "arm_mass_ratio",
        "leg_mass_ratio",
        "arm_length_equivalent_m",
        "leg_length_equivalent_m",
        "torso_buffer_length_m",
        "single_arm_mass_kg",
        "single_leg_mass_kg",
        "torso_mass_kg",
        "stability_default",
        "energy_default",
        "hp_proxy_default",
        "battery_energy_proxy_default",
        "motor_max_torque_Nm",
        "motor_max_rpm",
    ]
    return {k: d[k] for k in keys}


if __name__ == "__main__":
    print("pm01_params.py 自测开始")
    params = get_pm01_params()
    print(f"PM01 质量: {params['mass_kg']} kg")
    print(f"总自由度: {params['dof_total']}")
    print(f"腿长: {params['leg_length_m']} m")
    print(f"臂展: {params['arm_span_m']} m")
    print(f"Q90H 最大力矩: {params['motor_max_torque_Nm']} Nm")
    print(f"Q90H 最高转速: {params['motor_max_rpm']} rpm")
    print("pm01_params.py 自测完成")