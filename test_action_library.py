from pprint import pprint

from config.action_library import (
    ATTACK_ACTIONS,
    ACTION_ORDER,
    get_action_dict,
    get_action_list,
)


REQUIRED_FIELDS = [
    "code",
    "name",
    "category",
    "part",
    "target_zones",
    "attack_range",
    "eta",
    "omega",
    "dt",
    "impact_level",
    "accuracy_level",
    "balance_break_level",
    "combo_potential",
    "time_cost",
    "balance_risk",
    "energy_cost",
    "counter_risk",
    "pressure_potential",
    "notes",
]


def check_basic_counts():
    print("=== 1. 检查动作数量 ===")
    n_actions = len(ATTACK_ACTIONS)
    n_order = len(ACTION_ORDER)
    print(f"ATTACK_ACTIONS 数量: {n_actions}")
    print(f"ACTION_ORDER 数量 : {n_order}")

    assert n_actions == 13, f"动作数量应为 13，但当前为 {n_actions}"
    assert n_order == 13, f"ACTION_ORDER 数量应为 13，但当前为 {n_order}"
    print("通过\n")


def check_order_consistency():
    print("=== 2. 检查 ACTION_ORDER 与 ATTACK_ACTIONS 是否一致 ===")
    action_keys = set(ATTACK_ACTIONS.keys())
    order_keys = set(ACTION_ORDER)

    missing_in_order = action_keys - order_keys
    missing_in_actions = order_keys - action_keys

    assert not missing_in_order, f"这些动作存在于 ATTACK_ACTIONS 但不在 ACTION_ORDER 中: {missing_in_order}"
    assert not missing_in_actions, f"这些动作存在于 ACTION_ORDER 但不在 ATTACK_ACTIONS 中: {missing_in_actions}"
    print("通过\n")


def check_required_fields():
    print("=== 3. 检查字段完整性 ===")
    for code in ACTION_ORDER:
        action = ATTACK_ACTIONS[code].to_dict()
        missing = [f for f in REQUIRED_FIELDS if f not in action]
        assert not missing, f"{code} 缺少字段: {missing}"
    print("通过\n")


def check_value_ranges():
    print("=== 4. 检查关键数值范围 ===")
    valid_categories = {"拳法", "腿法", "组合技", "特殊技"}
    valid_parts = {"arm", "leg", "torso"}
    valid_ranges = {"near", "mid", "far", "near-mid", "mid-far"}

    score_fields = [
        "impact_level",
        "accuracy_level",
        "balance_break_level",
        "combo_potential",
        "time_cost",
        "balance_risk",
        "energy_cost",
        "counter_risk",
        "pressure_potential",
    ]

    for code in ACTION_ORDER:
        a = ATTACK_ACTIONS[code].to_dict()

        assert a["category"] in valid_categories, f"{code} category 非法: {a['category']}"
        assert a["part"] in valid_parts, f"{code} part 非法: {a['part']}"
        assert a["attack_range"] in valid_ranges, f"{code} attack_range 非法: {a['attack_range']}"

        assert isinstance(a["target_zones"], list) and len(a["target_zones"]) > 0, \
            f"{code} target_zones 应为非空列表"

        assert 0 <= a["eta"] <= 1.2, f"{code} eta 超出合理范围: {a['eta']}"
        assert a["omega"] > 0, f"{code} omega 应大于 0"
        assert a["dt"] > 0, f"{code} dt 应大于 0"

        for field in score_fields:
            value = a[field]
            assert 0 <= value <= 5, f"{code} 的 {field} 应在 [0, 5]，当前为 {value}"

    print("通过\n")


def preview_data():
    print("=== 5. 预览前 3 个动作 ===")
    action_list = get_action_list()
    for item in action_list[:3]:
        pprint(item)
        print("-" * 60)
    print("通过\n")


def check_conversion_interfaces():
    print("=== 6. 检查导出接口 ===")
    d = get_action_dict()
    lst = get_action_list()

    assert isinstance(d, dict), "get_action_dict() 应返回 dict"
    assert isinstance(lst, list), "get_action_list() 应返回 list"
    assert len(d) == 13, f"get_action_dict() 返回数量不对: {len(d)}"
    assert len(lst) == 13, f"get_action_list() 返回数量不对: {len(lst)}"

    first = lst[0]
    assert isinstance(first, dict), "get_action_list() 的元素应为 dict"

    print("通过\n")


def main():
    print("\n开始测试 action_library.py ...\n")
    check_basic_counts()
    check_order_consistency()
    check_required_fields()
    check_value_ranges()
    preview_data()
    check_conversion_interfaces()
    print("所有测试通过，action_library.py 基本可用。")


if __name__ == "__main__":
    main()