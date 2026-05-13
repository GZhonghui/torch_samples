"""
=============================================================================
  PyTorch 案例 04：Dataset / DataLoader / 标准训练循环
=============================================================================
  前几个案例已经学习了 Tensor、Autograd 和 nn.Module。本例把它们组合成
  一个更接近真实项目的训练流程：

    1. 自定义 Dataset
    2. 用 DataLoader 产生 mini-batch
    3. 训练集 / 验证集拆分
    4. 标准训练循环：zero_grad -> forward -> loss -> backward -> step
    5. 标准评估循环：model.eval() + torch.inference_mode()
    6. device 统一迁移
    7. 保存验证集表现最好的 checkpoint
    8. 加载 checkpoint 并做推理

  本例使用一个二维二分类任务，不需要下载数据。数据规则是：
      如果 x0 + x1 > 0，则类别为 1；否则类别为 0。

  运行方式：
    ../torch_venv/bin/python training_loop.py
=============================================================================
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


torch.manual_seed(0)


# ============================================================================
#  第 1 部分：定义 Dataset
# ============================================================================
#  Dataset 负责描述“一个样本长什么样”：
#    - __len__：数据集中有多少个样本
#    - __getitem__：根据 index 返回一个样本
#
#  DataLoader 会基于 Dataset 自动做 batch、shuffle、多进程读取等事情。

print("=" * 60)
print("1. 定义 Dataset")
print("=" * 60)


class ToyClassificationDataset(Dataset):
    """一个二维二分类数据集。

    每个样本包含：
      features: shape = (2,)
      label:    shape = ()，取值为 0 或 1
    """

    def __init__(self, num_samples=1000):
        super().__init__()

        self.features = torch.randn(num_samples, 2)

        # 规则：两个特征相加大于 0 就是类别 1，否则类别 0。
        # CrossEntropyLoss 要求 label 是整数类别，dtype 应该是 torch.long。
        self.labels = (self.features[:, 0] + self.features[:, 1] > 0).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


dataset = ToyClassificationDataset(num_samples=1000)
sample_x, sample_y = dataset[0]

print(f"数据集大小: {len(dataset)}")
print(f"单个 feature: {sample_x}, shape={sample_x.shape}, dtype={sample_x.dtype}")
print(f"单个 label:   {sample_y}, shape={sample_y.shape}, dtype={sample_y.dtype}")


# ============================================================================
#  第 2 部分：训练集 / 验证集拆分
# ============================================================================
#  训练集用于更新参数，验证集用于观察模型在没见过的数据上的表现。
#  generator 设置随机种子，让 random_split 每次拆分结果一致，方便复现。

print("\n" + "=" * 60)
print("2. 训练集 / 验证集拆分")
print("=" * 60)

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(0),
)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")


# ============================================================================
#  第 3 部分：DataLoader 产生 batch
# ============================================================================
#  DataLoader 会把多个样本自动堆叠成一个 batch。
#  如果单个 feature shape 是 (2,)，batch_size=32，则 batch_x shape 是 (32, 2)。
#
#  shuffle=True 通常只用于训练集，让每个 epoch 的样本顺序不同。
#  验证集一般不需要 shuffle。

print("\n" + "=" * 60)
print("3. DataLoader 产生 batch")
print("=" * 60)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

batch_x, batch_y = next(iter(train_loader))
print(f"batch_x.shape = {batch_x.shape}")
print(f"batch_y.shape = {batch_y.shape}")
print(f"batch_y 前 10 个: {batch_y[:10]}")


# ============================================================================
#  第 4 部分：定义模型
# ============================================================================
#  这是一个很小的 MLP：
#      Linear(2 -> 16) -> ReLU -> Linear(16 -> 2)
#
#  最后一层输出 2 个数，叫 logits，分别表示两个类别的未归一化分数。
#  使用 CrossEntropyLoss 时，不需要手动 softmax：
#  CrossEntropyLoss 内部已经包含了 log_softmax + negative log likelihood。

print("\n" + "=" * 60)
print("4. 定义模型")
print("=" * 60)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


model = TinyClassifier()
print(model)

with torch.inference_mode():
    logits = model(batch_x)

print(f"\nlogits.shape = {logits.shape}")
print(f"logits 前 3 行:\n{logits[:3]}")
print(f"预测类别前 10 个: {logits.argmax(dim=1)[:10]}")


# ============================================================================
#  第 5 部分：训练 / 评估工具函数
# ============================================================================
#  真实项目里通常会把一个 epoch 的训练和评估封装成函数。
#  这样主循环只负责调度 epoch、记录指标、保存 checkpoint。

print("\n" + "=" * 60)
print("5. 训练 / 评估工具函数")
print("=" * 60)


def accuracy_from_logits(logits, labels):
    """根据 logits 计算分类准确率。"""
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch，返回平均 loss 和 accuracy。"""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in dataloader:
        # DataLoader 默认产出的 Tensor 在 CPU 上。
        # 如果模型在 GPU 上，输入也必须移动到同一个 device。
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, labels)
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """评估模型，返回平均 loss 和 accuracy。"""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 评估 / 推理时不需要构建计算图。
    # inference_mode 比 no_grad 更适合纯推理场景。
    with torch.inference_mode():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_correct += accuracy_from_logits(logits, labels)
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


print("已定义 train_one_epoch() 和 evaluate()")


# ============================================================================
#  第 6 部分：完整训练循环 + best checkpoint
# ============================================================================
#  常见训练主循环：
#    1. 把模型移动到 device
#    2. 每个 epoch 训练一次
#    3. 每个 epoch 验证一次
#    4. 如果验证指标更好，就保存 checkpoint

print("\n" + "=" * 60)
print("6. 完整训练循环 + best checkpoint")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前 device: {device}")

model = TinyClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
best_checkpoint_path = checkpoint_dir / "best_classifier_state_dict.pt"

best_val_accuracy = 0.0
num_epochs = 20

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc

        # 保存 checkpoint 时推荐先转到 CPU。
        # 这样在没有 GPU 的机器上也能更方便地加载。
        torch.save(model.cpu().state_dict(), best_checkpoint_path)
        model.to(device)

    if epoch in [1, 2, 5, 10, 20]:
        print(
            f"epoch={epoch:2d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
        )

print(f"\n最佳验证集准确率: {best_val_accuracy:.3f}")
print(f"best checkpoint: {best_checkpoint_path}")


# ============================================================================
#  第 7 部分：加载 checkpoint 并做推理
# ============================================================================
#  推理流程通常是：
#    1. 创建同样结构的模型
#    2. load_state_dict 加载权重
#    3. model.eval()
#    4. inference_mode 下执行 forward
#    5. 从 logits 得到预测类别

print("\n" + "=" * 60)
print("7. 加载 checkpoint 并做推理")
print("=" * 60)

loaded_model = TinyClassifier()
loaded_state = torch.load(best_checkpoint_path, map_location="cpu")
loaded_model.load_state_dict(loaded_state)
loaded_model.eval()

test_points = torch.tensor(
    [
        [2.0, 1.0],     # x0 + x1 > 0，应该是类别 1
        [-2.0, -1.0],   # x0 + x1 < 0，应该是类别 0
        [1.0, -0.2],    # x0 + x1 > 0，应该是类别 1
        [-0.1, -0.3],   # x0 + x1 < 0，应该是类别 0
    ]
)

with torch.inference_mode():
    test_logits = loaded_model(test_points)
    test_probs = torch.softmax(test_logits, dim=1)
    test_predictions = test_logits.argmax(dim=1)

print(f"test_points:\n{test_points}")
print(f"test_logits:\n{test_logits}")
print(f"test_probs:\n{test_probs}")
print(f"预测类别: {test_predictions.tolist()}")

print("\n案例 04 完成。下一个案例可以专门学习 device / CUDA / dtype / autocast。")
