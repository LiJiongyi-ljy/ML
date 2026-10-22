#Multilayer Perceptron MLP 多层感知机

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:   #"-> None" belongs to "Return Type Annotation(返回值类型注解)", it can also be "-> int" , "-> str"
    random.seed(seed)                   #在定义函数时，参数后的 ":" 表示 "introduce type annotation(引出类型注解)" ，而 seed=42 是设置默认值
    np.random.seed(seed)
    torch.manual_seed(seed)


class DigitMLP(nn.Module):
    """输入 64 个像素特征，输出 10 个数字类别的分数。"""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 输出 logits；不要在这里加 Softmax
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def main() -> None:     
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备：{device}")

    # 1. 加载数据：共 1797 张 8x8 灰度数字图片，每张图有 64 个像素特征。
    digits = load_digits()

    X = digits.data.astype(np.float32) / 16.0  # 原像素值为 0~16，缩放到 0~1 (Normalization 归一化)
    y = digits.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 2. 定义网络、损失函数与优化器。
    model = DigitMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   #Adam优化器，lr(Learning Rate)

    print("\n网络结构：")
    print(model)
    print(f"训练样本：{len(train_dataset)}；测试样本：{len(test_dataset)}")

    # 3. 训练：前向传播 -> 计算损失 -> 反向传播 -> 更新参数。
    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()         # 清空上一步的梯度
            logits = model(features)      # 前向传播
            loss = loss_fn(logits, labels)
            loss.backward()               # 自动计算梯度（反向传播）
            optimizer.step()              # 根据梯度更新参数

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += batch_size

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            train_loss = train_loss_sum / train_total
            train_acc = train_correct / train_total
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss: {train_loss:.4f} | train acc: {train_acc:.2%} | "
                f"test loss: {test_loss:.4f} | test acc: {test_acc:.2%}"
            )

    # 4. 保存模型参数。
    model_path = Path("digit_mlp.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n模型参数已保存到：{model_path.resolve()}")

    # 5. 看看模型对几张测试图片的预测。
    model.eval()
    sample_x, sample_y = next(iter(test_loader))
    with torch.no_grad():
        predictions = model(sample_x.to(device)).argmax(dim=1).cpu()

    print("\n部分测试样本（真实标签 -> 预测标签）：")
    for true_label, predicted_label in zip(sample_y[:12].tolist(), predictions[:12].tolist()):
        mark = "✓" if true_label == predicted_label else "✗"
        print(f"  {true_label} -> {predicted_label} {mark}")


if __name__ == "__main__":
    main()
