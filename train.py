# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import PoetryModel
from utils import prepare_data
from tqdm import tqdm
from common import setup_logger
from config import CONFIG


def train():
    logger = setup_logger("PoetryTrainer")

    # 从配置加载参数
    epochs = CONFIG.getint("Training", "epochs")
    batch_size = CONFIG.getint("Training", "batch_size")
    lr = CONFIG.getfloat("Training", "learning_rate")
    save_dir = CONFIG["Training"]["save_dir"]

    logger.info(
        f"\n训练配置:\n"
        f"- Epochs: {epochs}\n"
        f"- Batch Size: {batch_size}\n"
        f"- Learning Rate: {lr}\n"
        f"- Save Dir: {save_dir}"
    )

    # 准备数据
    data, vocab = prepare_data("data")
    vocab_size = len(vocab)

    # 转换为Tensor
    data = torch.LongTensor(data)
    X, y = data[:, :-1], data[:, 1:]
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoetryModel(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_X, batch_y in progress:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss = criterion(output.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)

        # 保存检查点
        model_path = f"{save_dir}/model_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), model_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")

    logger.info(f"训练完成! 最佳loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
