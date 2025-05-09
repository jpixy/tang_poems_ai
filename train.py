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
import os


def train():
    logger = setup_logger("PoetryTrainer")

    # 从配置加载参数
    epochs = CONFIG.getint("Training", "epochs")
    batch_size = CONFIG.getint("Training", "batch_size")
    lr = CONFIG.getfloat("Optimizer", "Learning_rate")
    betas = tuple(map(float, CONFIG.get("Optimizer", "betas").split(",")))
    weight_decay = CONFIG.getfloat("Optimizer", "weight_decay")
    grad_clip = CONFIG.getfloat("Training", "grad_clip")
    weight_clip = CONFIG.getfloat("Training", "weight_clip")
    save_dir = CONFIG["Training"]["save_dir"]

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    logger.info(
        f"\n训练配置:\n"
        f"- Epochs: {epochs}\n"
        f"- Batch Size: {batch_size}\n"
        f"- Learning Rate: {lr}\n"
        f"- Gradient Clip: {grad_clip}\n"
        f"- Weight Clip: {weight_clip}\n"
        f"- Save Dir: {save_dir}"
    )

    # 准备数据
    data, vocab = prepare_data("data")
    vocab_size = len(vocab)
    logger.info(f"词汇表大小: {vocab_size}")

    # 转换为Tensor
    data = torch.LongTensor(data)
    X, y = data[:, :-1], data[:, 1:]
    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoetryModel(vocab_size).to(device)
    logger.info(f"Device using: {device}")
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
    )

    # 训练循环
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_X, batch_y in progress:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 前向传播
            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss = criterion(output.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()

            # 梯度裁剪
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # 权重约束
            if weight_clip > 0:
                with torch.no_grad():
                    for param in model.parameters():
                        param.clamp_(-weight_clip, weight_clip)

            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        # 保存检查点
        model_path = os.path.join(save_dir, f"model_epoch{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"发现新最佳模型，Loss: {best_loss:.4f}")

    logger.info(f"\n训练完成! 最佳loss: {best_loss:.4f}")
    logger.info(f"最佳模型已保存至: {best_model_path}")


if __name__ == "__main__":
    train()
