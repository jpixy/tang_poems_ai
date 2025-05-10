import torch
import random
from model import PoetryModel
from common import setup_logger
from config import CONFIG

logger = setup_logger("PoetryGenerator")


def generate_poem():
    # Load config
    max_length = CONFIG.getint("Model", "max_length")
    num_poems = CONFIG.getint("Generation", "num_poems")
    temperature = CONFIG.getfloat("Generation", "temperature")
    top_k = CONFIG.getint("Generation", "top_k")
    model_path = CONFIG["Paths"]["model_save_dir"]

    logger.info(
        f"\nGeneration Config:\n"
        f"- Max Length: {max_length}\n"
        f"- Num Poems: {num_poems}\n"
        f"- Temperature: {temperature}\n"
        f"- Top-K: {top_k}\n"
        f"- Model Path: {model_path}"
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoetryModel().to(device)
    tokenizer = model.tokenizer

    # Load model weights if available
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return

    # Generation function
    def sample_next_token(logits):
        logits = logits / temperature
        if top_k > 0:
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)
            return indices[torch.multinomial(probs, 1)].item()
        return torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

    # Generate poems
    for i in range(num_poems):
        input_text = random.choice(["春", "夏", "秋", "冬", "山", "水", "月"])
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        poem = input_text
        for _ in range(max_length):
            with torch.no_grad():
                outputs = model(input_ids)
                next_token = sample_next_token(outputs.logits[0, -1])

            if next_token == tokenizer.eos_token_id:
                break

            next_char = tokenizer.decode([next_token])
            poem += next_char
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=device)], dim=1
            )

        # Format output
        formatted_poem = poem.replace(" ", "").replace("\n", " ")
        logger.info(f"\nGenerated Poem {i + 1}:\n{formatted_poem}")


if __name__ == "__main__":
    generate_poem()

