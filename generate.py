import torch
import random
from model import PoetryModel
from utils import load_poems, build_vocab


def generate_poem(model_path="poetry_model.pth", data_dir="data", max_len=64):
    # 加载词汇表
    poems = load_poems(data_dir)
    vocab = build_vocab(poems)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    # 加载模型
    device = torch.device("cpu")
    model = PoetryModel(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 随机选择一个起始字符
    start_char = random.choice([c for c in vocab if c not in ["<PAD>", "<UNK>"]])
    input_seq = [vocab[start_char]] + [0] * (max_len - 1)
    input_tensor = torch.LongTensor(input_seq).unsqueeze(0).to(device)

    # 生成古诗
    generated = []
    hidden = None
    for _ in range(max_len):
        output, hidden = model(input_tensor, hidden)
        prob = torch.softmax(output[0, -1], dim=-1)
        char_idx = torch.multinomial(prob, 1).item()
        generated.append(inv_vocab.get(char_idx, "<UNK>"))
        input_tensor = torch.LongTensor([char_idx]).unsqueeze(0).to(device)

    # 拼接结果
    poem = "".join([c for c in generated if c not in ["<PAD>", "<UNK>"]])
    print("生成古诗：")
    print(poem)


if __name__ == "__main__":
    generate_poem()
