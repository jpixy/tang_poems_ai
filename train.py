import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import PoetryModel
from utils import prepare_data
from tqdm import tqdm
import os


def train(resume=False, checkpoint_path="poetry_model.pth"):
    # 准备数据
    data_dir = "data"
    data, vocab = prepare_data(data_dir)
    vocab_size = len(vocab)

    # 转换为PyTorch Tensor
    data = torch.LongTensor(data)
    X = data[:, :-1]  # 输入序列
    y = data[:, 1:]  # 目标序列
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 设置设备（自动检测GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = PoetryModel(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略<PAD>
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_epoch = 0
    # 如果恢复训练，加载检查点
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # 训练
    epochs = 50
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss = criterion(output.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())  # 动态显示 loss
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # 每个epoch结束后保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    print("训练完成！最终模型已保存为 poetry_model.pth")


if __name__ == "__main__":
    # 使用示例：
    # 从头开始训练
    # train()
    
    # 从上次检查点恢复训练
    train(resume=True)