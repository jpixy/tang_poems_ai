import json
import os
import re
import numpy as np
from collections import Counter


def load_poems(data_dir):
    """增强数据清洗的加载函数"""
    poems = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

                # 统一处理数组/对象格式
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if not isinstance(item, dict):
                        continue

                    # 提取paragraphs并清洗
                    paragraphs = item.get("paragraphs", [])
                    for p in paragraphs:
                        p = p.strip()
                        # 新增正则过滤
                        if (
                            len(p) >= 4  # 最小长度
                            and re.fullmatch(
                                r"[\u4e00-\u9fff，。！？、；：]+", p
                            )  # 仅保留中文和常用标点
                            and len(set(p)) >= len(p) // 2
                        ):  # 避免重复字过多
                            poems.append(p)

        except Exception as e:
            print(f"加载 {filename} 失败: {str(e)}")
            continue

    print(f"共加载 {len(poems)} 条有效诗句")
    return poems


def build_vocab(poems, min_freq=5, max_freq_ratio=0.5):
    """安全构建词汇表"""
    chars = [char for poem in poems for char in poem]
    counter = Counter(chars)
    total = sum(counter.values())

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for char, count in counter.items():
        # 过滤条件：频率适中且是中文或常用标点
        if min_freq <= count <= total * max_freq_ratio and re.fullmatch(
            r"[\u4e00-\u9fff，。！？]", char
        ):
            vocab[char] = len(vocab)

    print(f"最终词汇表大小: {len(vocab)}")
    return vocab


def build_vocab(poems, min_freq=10, max_freq_ratio=0.3):
    """构建词汇表（过滤空字符串）"""
    chars = [char for poem in poems for char in poem if poem.strip()]
    counter = Counter(chars)
    # 过滤条件
    total = sum(counter.values())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for char, count in counter.items():
        # 剔除生僻字和异常高频字
        if (min_freq <= count <= total * max_freq_ratio) and (
            "\u4e00" <= char <= "\u9fff" or char in "，。！？"
        ):
            vocab[char] = len(vocab)
    return vocab


def poem_to_tensor(poem, vocab, max_len=64):
    """转换时过滤空行"""
    poem = poem.strip()
    if not poem:
        return np.array([vocab["<PAD>"]] * max_len)
    tensor = [vocab.get(char, vocab["<UNK>"]) for char in poem]
    return np.array((tensor + [vocab["<PAD>"]] * max_len)[:max_len])


def prepare_data(data_dir, max_len=64):
    poems = load_poems(data_dir)
    if not poems:
        raise ValueError(
            "没有加载到有效数据！请检查：\n"
            "1. JSON文件是否包含'paragraphs'字段\n"
            "2. 文件编码是否为UTF-8\n"
            "3. 文件路径是否正确"
        )
    vocab = build_vocab(poems)
    data = [poem_to_tensor(p, vocab, max_len) for p in poems]
    return np.array(data), vocab
