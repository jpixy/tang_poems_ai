import json
import os
import numpy as np
from collections import Counter


def load_poems(data_dir):
    """增强健壮性的数据加载函数"""
    poems = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

                # 处理数组格式
                if isinstance(data, list):
                    for obj in data:
                        if not isinstance(obj, dict):
                            continue
                        if "paragraphs" in obj and isinstance(obj["paragraphs"], list):
                            poems.extend(p for p in obj["paragraphs"] if p.strip())
                        else:
                            print(
                                f"警告：{filename} 中的对象缺少有效paragraphs: {obj.keys()}"
                            )

                # 处理单个对象格式
                elif isinstance(data, dict):
                    if "paragraphs" in data and isinstance(data["paragraphs"], list):
                        poems.extend(p for p in data["paragraphs"] if p.strip())
                    else:
                        print(f"警告：{filename} 缺少有效paragraphs: {data.keys()}")

                else:
                    print(f"警告：{filename} 不是有效JSON对象或数组")

        except Exception as e:
            print(f"加载 {filename} 失败: {str(e)}")
            continue

    print(f"共加载 {len(poems)} 条有效诗句")
    return poems


def build_vocab(poems, min_freq=2):
    """构建词汇表（过滤空字符串）"""
    chars = [char for poem in poems for char in poem if poem.strip()]
    if not chars:
        raise ValueError("没有有效字符数据，请检查输入")
    counter = Counter(chars)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for char, freq in counter.items():
        if freq >= min_freq:
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
