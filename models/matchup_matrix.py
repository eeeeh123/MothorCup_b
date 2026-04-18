from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

try:
    from config.action_library import get_action_list
except ImportError:
    from config.action_library import prepare_attack_records as _prepare_attack_records_fallback  # type: ignore

    def get_action_list() -> List[Dict[str, Any]]:
        return _prepare_attack_records_fallback()

try:
    from config.defense_library import get_defense_list
except ImportError:
    from config.defense_library import prepare_defense_records as _prepare_defense_records_fallback  # type: ignore

    def get_defense_list() -> List[Dict[str, Any]]:
        return _prepare_defense_records_fallback()


# -------------------------------------------------------------------
# 数据来源说明
# -------------------------------------------------------------------
# 本模块已与《Q2_攻防关系采集》最终长表对齐。
# 关系表字段包括：
# - 攻击动作
# - 防守动作
# - 题面显式适配(0/1)
# - 现有参考分(0-5)
# - 最终核定分(0-10)
# - 防守后反击收益(0-10)
# - 失败后损失(0-10)
# - 推荐理由/已知依据
# - 数据来源
# - 置信度
#
# 设计原则：
# 1) 当前 transition.py 的 _defense_quality() 以 0~100 的 matchup_score 归一化使用；
# 2) 因此这里统一将：
#       最终核定分(0~10) -> matchup_score = *10
#       现有参考分(0~5) -> matchup_score = *20
# 3) 若未来 Excel 长表继续更新，只需同步替换 RAW_MATCHUP_ROWS。
# -------------------------------------------------------------------


RAW_MATCHUP_ROWS: List[Dict[str, Any]] = [
    {
        "attack_name": "直拳",
        "defense_name": "十字格挡",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "正面封挡直线快攻",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "单手拍挡",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "改变攻击轨迹，防守效率极高",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "肘挡",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "骨架硬抗，略显笨重",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "下压格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "高位攻击，下压格挡无效",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "可尝试抓取，但风险极高",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "左右侧闪",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "完美规避直线攻击并让出身位",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "下潜闪避",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "可规避高位直拳",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "后跳/撤步",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "安全拉开距离，但丧失反击机会",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "应对直拳过于多余，易送后背",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "滑步环绕",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "侧向绕行效果极佳",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "静态防御，容错率高",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "部分规避高位打击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "减少受击面积，对直拳有效",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "直拳冲击力小，无需重心补偿",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "顺势后移减伤",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "小幅微调距离",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "无意义动作，直接送分",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "非倒地状态不可用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "非倒地状态不可用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "闪→挡→反",
        "explicit_match": 0,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "高端复合防守，完美克制",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "复合反击，收益高",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "直拳",
        "defense_name": "挡→撤→绕",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "稳妥的退防体系",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "可格挡但侧面依然有空挡",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "弧线攻击较难拍挡",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "肘挡",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "护头胸侧面，完美防勾拳",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "下压格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "高位/中位攻击，下压无效",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "贴身肉搏可尝试锁臂",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "横向闪避易撞上勾拳轨迹",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "下潜闪避",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "低头下潜完美躲避高位弧线",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "后跳/撤步",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "脱离近身攻击范围",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用近身缠斗",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "滑步环绕",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "近战需极高预判",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "护头防御",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "有效保护头部两侧",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "降低重心规避高位伤害",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "侧面可能正是攻击点",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "对抗近身摇晃",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "顺势卸力有效",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "破坏对手近战距离",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "闪→挡→反",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "防守成本偏高",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "潜→闪→踢",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "复合连招，完美反制高位勾拳",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "勾拳",
        "defense_name": "挡→撤→绕",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "脱离近战泥潭",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "十字格挡",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "可硬抗横扫，但有轻微位移",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "摆拳力量过大，拍挡易折臂",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "肘挡",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "有效保护侧头部",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "下压格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "高位横扫，下压无效",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "大开大合，难以钳制",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "容易自己撞到另一侧拳头上",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "下潜闪避",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "低头完美规避横扫高位",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "后跳/撤步",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "拉开距离，避开锋芒",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "顺应横扫轨迹卸力",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "滑步环绕",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "需抓准前摇时机",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "龟缩防守有效减伤",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "极佳的降低重心避险",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "侧身容易被横向扫中背部",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "硬抗时必须补偿重心",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "顺势移动大幅降低冲击",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "小步法难以逃离大覆盖面",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "闪→挡→反",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "综合应对大范围攻击",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "下潜后立刻反击",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "摆拳",
        "defense_name": "挡→撤→绕",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "硬抗后撤步拉开，摆拳最佳克星",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "稳定吃下连击伤害",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "无法同时拍开多重攻击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "肘挡",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "可防近身缠斗",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "下压格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "高位连击，下压无效",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "钳制格挡",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "直接锁死对方双臂中断连击",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "易被后续连段打中",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "下潜闪避",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "可规避前两拳，但易被追击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "后跳/撤步",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "拉开距离，让对方打空力竭",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "转身等于把背部交给连击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "滑步环绕",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "一旦绕开，对方硬直极大",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "被动防御，最安逸的选择",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "会被持续压制",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "减少受击面",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "需持续抗冲击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "步步后退化解动能",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "碎步后退是经典拳击防守",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "闪→挡→反",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "化解连击并打断",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "动作过慢可能被打断",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "组合拳",
        "defense_name": "挡→撤→绕",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "边退边防是标准答案",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "强抗腿法易导致机体受损",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "单手拍挡",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "顺势外拨可使对方失去平衡",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "肘挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "防不住下肢直线冲击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "下压格挡",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "向下封堵腿部发力点",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "无法钳制腿部",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "左右侧闪",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "直线攻击最怕横向规避",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "下潜闪避",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "下潜正好把头撞到膝盖上",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "后跳/撤步",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "安全退出距离",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "动作多余",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "滑步环绕",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "绕侧可直接攻击支撑腿",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "前踢打腹部，护头无用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "硬抗中段重击风险大",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "侧身防御",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "侧身让出正面，大幅降低受击面",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "勉强硬抗",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "顺势后移减轻冲击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "控制距离",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "闪→挡→反",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "侧闪接反击是破腿法核心",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "禁止下潜",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "前踢",
        "defense_name": "挡→撤→绕",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "可用但偏保守",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "绝对破防(K=0.9)，硬抗必摔",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "冲击力太大，拍不动",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "肘挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "同理，硬抗即死",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "下压格挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "仅能极微量减伤",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "无法应用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "侧踢覆盖面大，单侧闪不够",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "下潜闪避",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "易被打中背部",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "后跳/撤步",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "退避三舍，避其锋芒最安全",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "转身闪避",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "顺应冲击力方向旋转卸力",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "滑步环绕",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "极限绕侧收益极高，但难度极大",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "完全防错位置",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "硬抗死路一条",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "侧踢本身就是大面积冲击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "补偿能力无法抵消7000N打击力",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "卸力缓冲",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "大幅后撤卸力是唯一软抗手段",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "小步无法逃逸",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "受控倒地",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "如果逃不掉，主动受控倒地免受硬伤",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "闪→挡→反",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "防不住侧踢的冲量",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "危险过大",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "侧踢",
        "defense_name": "挡→撤→绕",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "重点在于撤，不在于挡",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "冲击力极大，硬抗易损",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "无效防守",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "肘挡",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "保护头部被踢中",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "下压格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "部位错误",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "无法应用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "横向覆盖极大(3.1rad)，无法侧闪",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "下潜闪避",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "低头完美躲避高位扫踢！",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "后跳/撤步",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "拉出半径范围",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "顺势旋转",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "滑步环绕",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "时间窗口不够",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "如果躲不掉，只能龟缩护头",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "沉身防御",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "降重心底盘死守，完美规避高位",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "无意义",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "冲击力过大",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "无法完全卸力",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "无法逃出半径",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "受控倒地",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "终极保命技",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "闪→挡→反",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "防不住",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "潜→闪→踢",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "下潜躲过直接反踢支撑腿，绝杀！",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "回旋踢",
        "defense_name": "挡→撤→绕",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "勉强可用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "防守部位完全错误",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "防守部位错误",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "肘挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "防守部位错误",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "下压格挡",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "完美封堵下盘横扫",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "扫腿覆盖面大，侧闪无效",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "下潜闪避",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "下潜只会把脸凑上去",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "后跳/撤步",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "跳出扫腿半径是最佳解法",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "滑步环绕",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "难度极高",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "防守部位错误",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "强行稳底盘抗击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "无意义",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "重心补偿",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "核心对抗手段，防止被扫倒",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "腿部被扫很难缓冲",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "步点调整",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "提膝碎步调整，躲避扫腿",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "无需如此极端",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "闪→挡→反",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "不好防",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "禁止下潜",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "低扫腿",
        "defense_name": "挡→撤→绕",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "标准的退防动作",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "硬抗极易机体损伤",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "推不动",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "肘挡",
        "explicit_match": 1,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "用肘部向下砸防，略有效",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "下压格挡",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "双手下压完美锁死膝盖上提空间",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "可避开直线冲撞",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "下潜闪避",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "下潜等于爆头",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "后跳/撤步",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "膝撞极短，撤步即空",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "动作多余",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "滑步环绕",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "近战侧绕收益高",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "防错部位",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "降重心易吃满伤害",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "侧身让出着力点",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "勉强硬抗",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "后仰卸力",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "控制距离",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "不需要",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不需要",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "不需要",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "闪→挡→反",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "防线过长",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "绝对禁止下潜",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "膝撞",
        "defense_name": "挡→撤→绕",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "退守标准动作",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "全面吃下伤害，保守有效",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "顾此失彼",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "肘挡",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "防护面积不足",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "下压格挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "会被上面拳法打中",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "极难捕捉",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "易被组合攻击刮蹭",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "下潜闪避",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "易被腿法追击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "后跳/撤步",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "脱离连段距离是唯一解",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "滑步环绕",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "预判极难",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "护头防御",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "放弃下盘，死守头部",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "易吃高位伤害",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "防守不全面",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "硬吃连段",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "卸力缓冲",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "顺势移动大幅减伤",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "小幅后退化解",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "不至于",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "闪→挡→反",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "防住一套接反击",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "风险极大",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "拳腿组合",
        "defense_name": "挡→撤→绕",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "教科书式的破连招退防",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "久守必破",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "无力抵挡",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "肘挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "无力抵挡",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "下压格挡",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "只能挡一脚",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "扫荡面积过大",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "下潜闪避",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "风险极高",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "后跳/撤步",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "一退再退，等其力竭",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "找死",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "滑步环绕",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "时间不充裕",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "守不住底盘",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "硬扛必摔",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "极其勉强的支撑",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "随波逐流式退让",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "步点调整",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "连续碎步化解压迫",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "如被逼死角可考虑倒地保全",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "倒地防御",
        "explicit_match": 1,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "如已倒地只能死守",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "闪→挡→反",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "防线被撕裂",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "危险极大",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "五连踢",
        "defense_name": "挡→撤→绕",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "唯一复合退防解法",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "冲撞破防(K=1)，格挡必倒",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "螳臂当车",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "肘挡",
        "explicit_match": 1,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "用肘部做楔子勉强顶住",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "下压格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "部位错误",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "可尝试锁肩，极度危险",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "冲撞面积大，难以单次侧闪逃出",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "下潜闪避",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "会被直接压倒",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "后跳/撤步",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "速度比后退快，撤步无用",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "转身闪避",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "斗牛士动作，顺势旋身完美避让",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "滑步环绕",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "一旦绕开，对方背门大开",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "强行降重心做肉盾",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "减少受力面",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "重心补偿",
        "explicit_match": 1,
        "reference_score_0_5": 4.0,
        "final_score_0_10": 8.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "拼电机扭矩强行顶牛",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "被推着走",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "被逼死角干脆倒地防止冲撞毁伤",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "快速起身",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "不适用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "闪→挡→反",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "难以执行",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "极其危险",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "冲撞",
        "defense_name": "挡→撤→绕",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "可尝试圆周运动化解",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "十字格挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 4.0,
        "reason": "距离不对",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "单手拍挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 6.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "肘挡",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "下压格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 7.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "钳制格挡",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "左右侧闪",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 9.0,
        "reason": "没必要闪",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "下潜闪避",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 8.5,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "后跳/撤步",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "拉开距离让对方在地上空挥",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "转身闪避",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 4.0,
        "fail_loss_0_10": 8.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "滑步环绕",
        "explicit_match": 0,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 9.0,
        "fail_loss_0_10": 7.0,
        "reason": "绕到头后方攻击",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "护头防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 2.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "沉身防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 1.0,
        "fail_loss_0_10": 3.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "侧身防御",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 3.0,
        "fail_loss_0_10": 4.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "重心补偿",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 8.0,
        "reason": "防对方抓腿",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "卸力缓冲",
        "explicit_match": 0,
        "reference_score_0_5": 1.0,
        "final_score_0_10": 2.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 5.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "步点调整",
        "explicit_match": 0,
        "reference_score_0_5": 2.0,
        "final_score_0_10": 4.0,
        "counter_gain_0_10": 5.0,
        "fail_loss_0_10": 6.0,
        "reason": "小碎步走开即可",
        "source": "物理推演",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "受控倒地",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 10.0,
        "reason": "自己也躺下无意义",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "快速起身",
        "explicit_match": 1,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 2.0,
        "fail_loss_0_10": 8.0,
        "reason": "若双方倒地，我方快速起身即占优",
        "source": "表2_攻防克制表",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "倒地防御",
        "explicit_match": 0,
        "reference_score_0_5": 5.0,
        "final_score_0_10": 10.0,
        "counter_gain_0_10": 0.0,
        "fail_loss_0_10": 9.0,
        "reason": "地面缠斗防御",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "闪→挡→反",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 7.0,
        "fail_loss_0_10": 8.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "潜→闪→踢",
        "explicit_match": 0,
        "reference_score_0_5": 0.0,
        "final_score_0_10": 0.0,
        "counter_gain_0_10": 8.0,
        "fail_loss_0_10": 9.0,
        "reason": "无用",
        "source": "物理推演",
        "confidence": "高",
        "remark": ""
    },
    {
        "attack_name": "倒地反击",
        "defense_name": "挡→撤→绕",
        "explicit_match": 1,
        "reference_score_0_5": 3.0,
        "final_score_0_10": 6.0,
        "counter_gain_0_10": 6.0,
        "fail_loss_0_10": 6.0,
        "reason": "稳扎稳打拉开",
        "source": "表2_攻防克制表",
        "confidence": "中",
        "remark": ""
    }
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _compute_matchup_score(
    explicit_match: int,
    reference_score_0_5: Optional[float],
    final_score_0_10: Optional[float],
) -> float:
    """
    生成供 Q2/Q3/Q4 使用的 0~100 matchup_score。

    优先级：
    1. 最终核定分(0~10) * 10
    2. 现有参考分(0~5) * 20
    3. 若都缺失，则按显式适配给一个兜底分
    """
    if final_score_0_10 is not None:
        return 10.0 * _safe_float(final_score_0_10)
    if reference_score_0_5 is not None:
        return 20.0 * _safe_float(reference_score_0_5)
    return 60.0 if explicit_match else 30.0


def prepare_attack_records() -> List[Dict[str, Any]]:
    """
    兼容 transition.py 等模块的旧接口。
    直接从 action_library 读取最终动作字典。
    """
    return get_action_list()


def prepare_defense_records() -> List[Dict[str, Any]]:
    """
    兼容 transition.py 等模块的旧接口。
    直接从 defense_library 读取最终防守字典。
    """
    return get_defense_list()


def _build_attack_maps(attack_records: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    name_to_code: Dict[str, str] = {}
    for item in attack_records:
        name = _clean_text(item.get("name"))
        code = _clean_text(item.get("code"))
        if name:
            by_name[name] = item
            if code:
                name_to_code[name] = code
    return by_name, name_to_code


def _build_defense_maps(defense_records: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    name_to_code: Dict[str, str] = {}
    for item in defense_records:
        name = _clean_text(item.get("name"))
        code = _clean_text(item.get("code"))
        if name:
            by_name[name] = item
            if code:
                name_to_code[name] = code
    return by_name, name_to_code


def build_matchup_matrix(
    attack_records: Optional[List[Dict[str, Any]]] = None,
    defense_records: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    生成完整的攻防关系 DataFrame。

    返回字段包括：
    - attack_code / attack_name / attack_category
    - defense_code / defense_name / defense_category
    - explicit_match
    - reference_score_0_5
    - final_score_0_10
    - counter_gain_0_10
    - fail_loss_0_10
    - matchup_score (0~100，供 transition.py 使用)
    - reason / source / confidence / remark
    """
    if attack_records is None:
        attack_records = prepare_attack_records()
    if defense_records is None:
        defense_records = prepare_defense_records()

    attack_by_name, attack_name_to_code = _build_attack_maps(attack_records)
    defense_by_name, defense_name_to_code = _build_defense_maps(defense_records)

    rows: List[Dict[str, Any]] = []

    for item in RAW_MATCHUP_ROWS:
        attack_name = _clean_text(item["attack_name"])
        defense_name = _clean_text(item["defense_name"])

        attack_info = attack_by_name.get(attack_name, {})
        defense_info = defense_by_name.get(defense_name, {})

        explicit_match = int(item.get("explicit_match", 0) or 0)
        reference_score_0_5 = item.get("reference_score_0_5")
        final_score_0_10 = item.get("final_score_0_10")
        counter_gain_0_10 = item.get("counter_gain_0_10")
        fail_loss_0_10 = item.get("fail_loss_0_10")

        matchup_score = _compute_matchup_score(
            explicit_match=explicit_match,
            reference_score_0_5=reference_score_0_5,
            final_score_0_10=final_score_0_10,
        )

        rows.append({
            "attack_code": attack_name_to_code.get(attack_name, _clean_text(attack_info.get("code"))),
            "attack_name": attack_name,
            "attack_category": _clean_text(attack_info.get("category")),
            "defense_code": defense_name_to_code.get(defense_name, _clean_text(defense_info.get("code"))),
            "defense_name": defense_name,
            "defense_category": _clean_text(defense_info.get("category")),
            "explicit_match": explicit_match,
            "reference_score_0_5": _safe_float(reference_score_0_5) if reference_score_0_5 is not None else None,
            "final_score_0_10": _safe_float(final_score_0_10) if final_score_0_10 is not None else None,
            "counter_gain_0_10": _safe_float(counter_gain_0_10) if counter_gain_0_10 is not None else None,
            "fail_loss_0_10": _safe_float(fail_loss_0_10) if fail_loss_0_10 is not None else None,
            "matchup_score": matchup_score,
            "reason": _clean_text(item.get("reason")),
            "source": _clean_text(item.get("source")),
            "confidence": _clean_text(item.get("confidence")),
            "remark": _clean_text(item.get("remark")),
        })

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["attack_code", "defense_code", "attack_name", "defense_name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def build_matchup_lookup(
    attack_records: Optional[List[Dict[str, Any]]] = None,
    defense_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    构建 (attack_name, defense_name) -> row 的查找字典。
    """
    df = build_matchup_matrix(attack_records, defense_records)
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = (row["attack_name"], row["defense_name"])
        lookup[key] = row.to_dict()
    return lookup


def get_matchup_row(
    attack_name: str,
    defense_name: str,
    attack_records: Optional[List[Dict[str, Any]]] = None,
    defense_records: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    获取某一组 攻击动作 × 防守动作 的关系记录。
    """
    lookup = build_matchup_lookup(attack_records, defense_records)
    return lookup.get((attack_name, defense_name))


def get_top_defenses_for_attack(
    attack_name: str,
    top_k: int = 3,
    attack_records: Optional[List[Dict[str, Any]]] = None,
    defense_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    给定攻击动作名称，返回 Top-K 防守动作。
    排序优先级：
    1. matchup_score 降序
    2. explicit_match 降序
    3. counter_gain_0_10 降序
    4. fail_loss_0_10 升序
    """
    df = build_matchup_matrix(attack_records, defense_records)
    sub = df[df["attack_name"] == attack_name].copy()
    if sub.empty:
        return []

    sub["counter_gain_0_10"] = sub["counter_gain_0_10"].fillna(0.0)
    sub["fail_loss_0_10"] = sub["fail_loss_0_10"].fillna(10.0)

    sub = sub.sort_values(
        by=["matchup_score", "explicit_match", "counter_gain_0_10", "fail_loss_0_10"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    return sub.head(top_k).to_dict(orient="records")


def summarize_top_defenses(
    attack_names: Optional[List[str]] = None,
    top_k: int = 3,
    attack_records: Optional[List[Dict[str, Any]]] = None,
    defense_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    返回多个攻击动作的 Top-K 防守动作摘要。
    """
    if attack_records is None:
        attack_records = prepare_attack_records()
    if defense_records is None:
        defense_records = prepare_defense_records()

    if attack_names is None:
        attack_names = [item["name"] for item in attack_records]

    result: Dict[str, List[Dict[str, Any]]] = {}
    for attack_name in attack_names:
        result[attack_name] = get_top_defenses_for_attack(
            attack_name=attack_name,
            top_k=top_k,
            attack_records=attack_records,
            defense_records=defense_records,
        )
    return result


def print_top_defenses_demo(
    attack_names: Optional[List[str]] = None,
    top_k: int = 3,
) -> None:
    """
    命令行演示输出。
    """
    summary = summarize_top_defenses(attack_names=attack_names, top_k=top_k)
    for attack_name, rows in summary.items():
        print(f"【{attack_name}】")
        if not rows:
            print("  无匹配记录")
            print()
            continue
        for i, row in enumerate(rows, start=1):
            reason = row.get("reason", "")
            print(
                f"  Rank {i}: {row['defense_name']} | "
                f"score={row['matchup_score']:.2f} | "
                f"explicit={row['explicit_match']} | "
                f"counter={row.get('counter_gain_0_10')} | "
                f"fail_loss={row.get('fail_loss_0_10')} | "
                f"reason={reason}"
            )
        print()


if __name__ == "__main__":
    print("matchup_matrix.py 自测开始")

    attack_records = prepare_attack_records()
    defense_records = prepare_defense_records()
    df = build_matchup_matrix(attack_records, defense_records)

    print(f"攻击动作数: {len({r['name'] for r in attack_records})}")
    print(f"防守动作数: {len({r['name'] for r in defense_records})}")
    print(f"关系记录数: {len(df)}")

    demo_attacks = ["五连踢", "低扫腿", "侧踢"]
    print_top_defenses_demo(attack_names=demo_attacks, top_k=3)

    print("matchup_matrix.py 自测完成")
