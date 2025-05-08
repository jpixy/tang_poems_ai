import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import PoetryModel
from utils import prepare_data
from tqdm import tqdm


def train():
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

    # 初始化模型
    device = torch.device("cpu")
    model = PoetryModel(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略<PAD>
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss = criterion(output.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), "poetry_model.pth")
    print("训练完成！模型已保存为 poetry_model.pth")


if __name__ == "__main__":
    train()
