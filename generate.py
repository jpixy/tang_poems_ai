# generate.py
import torch
import random
from model import PoetryModel
from utils import load_poems, build_vocab
from common import setup_logger
from config import CONFIG


def generate_poem():
    logger = setup_logger("PoetryGenerator")

    # 从配置加载参数
    max_length = CONFIG.getint("Generation", "max_length")
    num_poems = CONFIG.getint("Generation", "num_poems")
    temperature = CONFIG.getfloat("Generation", "temperature")

    logger.info(
        f"\n生成配置:\n"
        f"- 最大长度: {max_length}\n"
        f"- 生成数量: {num_poems}\n"
        f"- 随机温度: {temperature}"
    )

    # 加载词汇表和模型
    poems = load_poems("data")
    vocab = build_vocab(poems)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoetryModel(vocab_size).to(device)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.eval()

    # 生成诗歌
    valid_chars = [c for c in vocab if c not in ["<PAD>", "<UNK>"]]
    for i in range(num_poems):
        start_char = random.choice(valid_chars)
        logger.info(f"\n生成诗歌 {i + 1}/{num_poems}，起始字: '{start_char}'")

        input_seq = [vocab[start_char]] + [0] * (max_length - 1)
        input_tensor = torch.LongTensor(input_seq).unsqueeze(0).to(device)

        poem = []
        hidden = None
        for _ in range(max_length):
            with torch.no_grad():
                output, hidden = model(input_tensor, hidden)
                prob = torch.softmax(output[0, -1] / temperature, dim=-1)
                char_idx = torch.multinomial(prob, 1).item()

            char = inv_vocab.get(char_idx, "<UNK>")
            if char in ["<PAD>", "<UNK>"]:
                break

            poem.append(char)
            input_tensor = torch.LongTensor([char_idx]).unsqueeze(0).to(device)

        logger.info("=" * 40)
        logger.info("".join(poem))
        logger.info("=" * 40)


if __name__ == "__main__":
    generate_poem()

