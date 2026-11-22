"""distill.data._parse_topic_list 解析健壮性测试。

1.5B teacher 的主题输出形态不稳定（Python repr / 围栏 / 编号 / 多余引号），
解析器必须全部剥壳；这里锁定各形态的解析结果，防退化。
"""

from distill.data import _parse_topic_list


def test_plain_lines() -> None:
    raw = "什么是过拟合？\n什么是梯度下降？\n什么是反向传播？"
    assert _parse_topic_list(raw) == ["什么是过拟合？", "什么是梯度下降？", "什么是反向传播？"]


def test_numbered_lines() -> None:
    raw = "1. 什么是算法？\n2. 什么是互联网？\n- 圆周率是多少？"
    assert _parse_topic_list(raw) == ["什么是算法？", "什么是互联网？", "圆周率是多少？"]


def test_python_repr_dict() -> None:
    raw = "{'question': '唐朝时期最著名的皇帝是谁？'}\n{'question': '中国四大发明是什么？'}"
    assert _parse_topic_list(raw) == ["唐朝时期最著名的皇帝是谁？", "中国四大发明是什么？"]


def test_json_list() -> None:
    raw = '["什么是 DNA？", "光速是多少？"]'
    assert _parse_topic_list(raw) == ["什么是 DNA？", "光速是多少？"]


def test_fenced_json() -> None:
    raw = '```json\n["地球到月球有多远？"]\n```'
    assert _parse_topic_list(raw) == ["地球到月球有多远？"]


def test_dedupe_and_noise_filter() -> None:
    raw = "什么是算法？\n什么是算法？\n这不是问题\n太短？\n好的。"
    assert _parse_topic_list(raw) == ["什么是算法？"]
