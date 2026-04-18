from pprint import pprint

from config.defense_library import (
    DEFENSE_ACTIONS,
    DEFENSE_ORDER,
    get_defense_dict,
    get_defense_list,
)


REQUIRED_FIELDS = [
    "code",
    "name",
    "category",
    "defense_style",
    "applicable_attack_types",
    "direct_defense_level",
    "stability_recovery",
    "counter_attack_potential",
    "timing_difficulty",
    "energy_cost",
    "risk_if_failed",
    "mobility_loss",
    "notes",
]

VALID_CATEGORIES = {"格挡", "闪避", "姿态", "平衡", "倒地", "组合"}
VALID_ATTACK_TYPES = {
    "直拳", "勾拳", "组合拳", "摆拳", "前踢", "侧踢", "回旋踢",
    "低扫腿", "膝撞", "拳腿组合", "五连踢", "冲撞", "倒地反击"
}
SCORE_FIELDS = [
    "direct_defense_level",
    "stability_recovery",
    "counter_attack_potential",
    "timing_difficulty",
    "energy_cost",
    "risk_if_failed",
    "mobility_loss",
]


def check_basic_counts():
    print("=== 1. 检查防守动作数量 ===")
    n_def = len(DEFENSE_ACTIONS)
    n_order = len(DEFENSE_ORDER)
    print(f"DEFENSE_ACTIONS 数量: {n_def}")
    print(f"DEFENSE_ORDER 数量 : {n_order}")

    assert n_def == 22, f"防守动作数量应为 22，但当前为 {n_def}"
    assert n_order == 22, f"DEFENSE_ORDER 数量应为 22，但当前为 {n_order}"
    print("通过\n")


def check_order_consistency():
    print("=== 2. 检查 DEFENSE_ORDER 与 DEFENSE_ACTIONS 是否一致 ===")
    defense_keys = set(DEFENSE_ACTIONS.keys())
    order_keys = set(DEFENSE_ORDER)

    missing_in_order = defense_keys - order_keys
    missing_in_actions = order_keys - defense_keys

    assert not missing_in_order, f"这些动作存在于 DEFENSE_ACTIONS 但不在 DEFENSE_ORDER 中: {missing_in_order}"
    assert not missing_in_actions, f"这些动作存在于 DEFENSE_ORDER 但不在 DEFENSE_ACTIONS 中: {missing_in_actions}"
    print("通过\n")


def check_required_fields():
    print("=== 3. 检查字段完整性 ===")
    for code in DEFENSE_ORDER:
        defense = DEFENSE_ACTIONS[code].to_dict()
        missing = [f for f in REQUIRED_FIELDS if f not in defense]
        assert not missing, f"{code} 缺少字段: {missing}"
    print("通过\n")


def check_value_ranges():
    print("=== 4. 检查关键字段范围 ===")
    for code in DEFENSE_ORDER:
        d = DEFENSE_ACTIONS[code].to_dict()

        assert d["category"] in VALID_CATEGORIES, f"{code} category 非法: {d['category']}"
        assert isinstance(d["defense_style"], str) and d["defense_style"].strip(), f"{code} defense_style 不能为空"

        applicable = d["applicable_attack_types"]
        assert isinstance(applicable, list) and len(applicable) > 0, f"{code} applicable_attack_types 应为非空列表"
        invalid_types = [x for x in applicable if x not in VALID_ATTACK_TYPES]
        assert not invalid_types, f"{code} 存在非法攻击类型: {invalid_types}"

        for field in SCORE_FIELDS:
            value = d[field]
            assert isinstance(value, (int, float)), f"{code} 的 {field} 应为数值"
            assert 0 <= value <= 5, f"{code} 的 {field} 应在 [0, 5]，当前为 {value}"

    print("通过\n")


def check_category_distribution():
    print("=== 5. 检查类别分布 ===")
    category_count = {}
    for code in DEFENSE_ORDER:
        cat = DEFENSE_ACTIONS[code].category
        category_count[cat] = category_count.get(cat, 0) + 1

    pprint(category_count)

    assert category_count.get("格挡", 0) == 5, "格挡类数量应为 5"
    assert category_count.get("闪避", 0) == 5, "闪避类数量应为 5"
    assert category_count.get("姿态", 0) == 3, "姿态类数量应为 3"
    assert category_count.get("平衡", 0) == 3, "平衡类数量应为 3"
    assert category_count.get("倒地", 0) == 3, "倒地类数量应为 3"
    assert category_count.get("组合", 0) == 3, "组合类数量应为 3"

    print("通过\n")


def preview_data():
    print("=== 6. 预览前 3 个防守动作 ===")
    defense_list = get_defense_list()
    for item in defense_list[:3]:
        pprint(item)
        print("-" * 60)
    print("通过\n")


def check_conversion_interfaces():
    print("=== 7. 检查导出接口 ===")
    d = get_defense_dict()
    lst = get_defense_list()

    assert isinstance(d, dict), "get_defense_dict() 应返回 dict"
    assert isinstance(lst, list), "get_defense_list() 应返回 list"
    assert len(d) == 22, f"get_defense_dict() 返回数量不对: {len(d)}"
    assert len(lst) == 22, f"get_defense_list() 返回数量不对: {len(lst)}"
    assert isinstance(lst[0], dict), "get_defense_list() 的元素应为 dict"

    print("通过\n")


def main():
    print("\n开始测试 defense_library.py ...\n")
    check_basic_counts()
    check_order_consistency()
    check_required_fields()
    check_value_ranges()
    check_category_distribution()
    preview_data()
    check_conversion_interfaces()
    print("所有测试通过，defense_library.py 基本可用。")


if __name__ == "__main__":
    main()