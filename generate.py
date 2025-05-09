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
    top_k = CONFIG.getint("Generation", "top_k", fallback=5)  # 新增top_k参数
    model_path = CONFIG.get(
        "Generation", "model_path", fallback="models/best_model.pth"
    )

    logger.info(
        f"\n生成配置:\n"
        f"- 最大长度: {max_length}\n"
        f"- 生成数量: {num_poems}\n"
        f"- 随机温度: {temperature}\n"
        f"- Top-K采样: {top_k}\n"
        f"- 模型路径: {model_path}"
    )

    # 加载词汇表和模型
    poems = load_poems("data")
    vocab = build_vocab(poems)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载模型
    model = PoetryModel(vocab_size).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return

    # 改进的采样策略
    def sample_next_char(logits):
        """带温度和top-k限制的采样"""
        logits = logits / temperature
        if top_k > 0:
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            return indices[torch.multinomial(probs, 1)].item()
        return torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

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
                char_idx = sample_next_char(output[0, -1])

            char = inv_vocab.get(char_idx, "<UNK>")
            if char in ["<PAD>", "<UNK>"]:
                break

            poem.append(char)
            input_tensor = torch.LongTensor([char_idx]).unsqueeze(0).to(device)

        # 格式化输出
        formatted_poem = ""
        for j, char in enumerate(poem):
            formatted_poem += char
            if char in "，。！？" and j != len(poem) - 1:
                formatted_poem += "\n"

        logger.info("=" * 40)
        logger.info(formatted_poem)
        logger.info("=" * 40)


if __name__ == "__main__":
    generate_poem()

