"""web_demo 多轮 prompt 拼接的回归测试。

快捷问题按钮会预填 (问题, None) 占位轮次，历史上曾把 None 拼成
"助手:None" 污染 prompt，导致模型答非所问（塌缩到高频答案）。
"""

from train.web_demo import build_multi_turn_prompt


class TestBuildMultiTurnPrompt:
    def test_empty_history_single_turn(self):
        assert build_multi_turn_prompt([], "Q") == "用户:Q\n助手:"

    def test_placeholder_round_is_skipped(self):
        # 快捷问题按钮路径：history 预填 [(Q, None)]，新消息同 Q
        assert build_multi_turn_prompt([("Q", None)], "Q") == "用户:Q\n助手:"

    def test_empty_assistant_round_is_skipped(self):
        assert build_multi_turn_prompt([("Q1", "")], "Q2") == "用户:Q2\n助手:"

    def test_normal_history_is_joined(self):
        history = [("Q1", "A1")]
        assert build_multi_turn_prompt(history, "Q2") == "用户:Q1\n助手:A1\n用户:Q2\n助手:"

    def test_mixed_history_keeps_answered_rounds_only(self):
        history = [("Q1", "A1"), ("Q2", None)]
        assert build_multi_turn_prompt(history, "Q3") == "用户:Q1\n助手:A1\n用户:Q3\n助手:"
