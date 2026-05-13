"""
=============================================================================
  PyTorch 案例 03：nn.Module 与模型基础
=============================================================================
  nn.Module 是 PyTorch 组织模型的核心抽象。无论是一个简单线性模型，
  还是 LLM 里的 Embedding、Attention、MLP、Transformer Block，
  最终都会被组织成一个个 nn.Module。

  本例重点不是讲复杂网络结构，而是讲清楚 PyTorch Model 相关的基础机制：
    1. 如何定义一个 nn.Module
    2. 子模块和参数如何被自动注册
    3. forward() 如何被调用
    4. parameters() / named_parameters() / state_dict()
    5. optimizer 如何配合 Module 训练
    6. train() / eval() 的区别
    7. buffer 是什么
    8. 保存和加载模型权重

  运行方式：
    ../torch_venv/bin/python module_basics.py
=============================================================================
"""

from pathlib import Path

import torch
from torch import nn


torch.manual_seed(0)


# ============================================================================
#  第 1 部分：定义一个最小 nn.Module
# ============================================================================
#  nn.Module 的基本写法：
#    1. 继承 nn.Module
#    2. 在 __init__ 中定义子模块或参数
#    3. 在 forward 中描述前向计算
#
#  注意：调用模型时通常写 model(x)，而不是 model.forward(x)。
#  因为 model(x) 会走 nn.Module.__call__，PyTorch 可以在这里处理 hook、
#  autocast、编译等机制，然后再调用 forward。

print("=" * 60)
print("1. 定义一个最小 nn.Module")
print("=" * 60)


class LinearRegressionModel(nn.Module):
    """最小线性回归模型：y = x @ weight.T + bias。"""

    def __init__(self):
        super().__init__()

        # nn.Linear 是一个子模块，里面包含 weight 和 bias 两个 Parameter。
        # 输入特征数 in_features=1，输出特征数 out_features=1。
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel()
print(model)

x = torch.tensor([[1.0], [2.0], [3.0]])
y = model(x)

print(f"\n输入 x.shape = {x.shape}")
print(f"输出 y.shape = {y.shape}")
print(f"输出 y =\n{y}")


# ============================================================================
#  第 2 部分：参数注册与 named_parameters()
# ============================================================================
#  只要把 nn.Module 或 nn.Parameter 赋值给 self.xxx，PyTorch 就会自动注册。
#  注册后的参数会出现在 model.parameters() / model.named_parameters() 中，
#  optimizer 也正是通过这些参数来更新模型。

print("\n" + "=" * 60)
print("2. 参数注册与 named_parameters()")
print("=" * 60)

for name, param in model.named_parameters():
    print(
        f"{name:13s} | shape={tuple(param.shape)} | "
        f"requires_grad={param.requires_grad}"
    )

total_params = sum(p.numel() for p in model.parameters())
print(f"\n参数总数: {total_params}")


# ============================================================================
#  第 3 部分：state_dict()
# ============================================================================
#  state_dict 是模型“可保存状态”的字典：
#    - 包含所有注册过的 Parameter
#    - 也包含所有注册过的 buffer
#
#  对推理部署来说，state_dict 很重要：通常保存 / 加载的就是权重字典，
#  而不是整个 Python 模型对象。

print("\n" + "=" * 60)
print("3. state_dict()")
print("=" * 60)

state = model.state_dict()
for name, tensor in state.items():
    print(f"{name:13s} | shape={tuple(tensor.shape)} | dtype={tensor.dtype}")


# ============================================================================
#  第 4 部分：用 Module + optimizer 训练
# ============================================================================
#  上一个案例里我们手写了：
#      loss.backward()
#      with torch.no_grad():
#          param -= lr * param.grad
#          param.grad = None
#
#  实际项目里通常用 optimizer 来做参数更新：
#      optimizer.zero_grad()
#      loss.backward()
#      optimizer.step()

print("\n" + "=" * 60)
print("4. 用 Module + optimizer 训练")
print("=" * 60)

# 构造数据：真实关系 y = 2*x + 1
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

train_model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(train_model.parameters(), lr=0.05)

print("训练前参数:")
for name, param in train_model.named_parameters():
    print(f"  {name:13s} = {param.detach().flatten().tolist()}")

for step in range(1, 201):
    # 1. forward
    y_pred = train_model(x_train)

    # 2. loss
    loss = criterion(y_pred, y_train)

    # 3. 清空上一轮梯度
    optimizer.zero_grad()

    # 4. backward
    loss.backward()

    # 5. 根据梯度更新参数
    optimizer.step()

    if step in [1, 2, 5, 10, 50, 100, 200]:
        weight = train_model.linear.weight.item()
        bias = train_model.linear.bias.item()
        print(
            f"step={step:3d} | loss={loss.item():.6f} | "
            f"weight={weight:.4f} | bias={bias:.4f}"
        )


# ============================================================================
#  第 5 部分：train() 与 eval()
# ============================================================================
#  model.train() 和 model.eval() 不会打开或关闭梯度。
#  它们影响的是某些层的行为，例如：
#    - Dropout：训练时随机丢弃部分元素，eval 时不丢弃
#    - BatchNorm：训练时更新统计量，eval 时使用已有统计量
#
#  推理时常见组合是：
#      model.eval()
#      with torch.inference_mode():
#          output = model(input)

print("\n" + "=" * 60)
print("5. train() 与 eval()")
print("=" * 60)


class DropoutDemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.dropout(x)


dropout_model = DropoutDemoModel()
dropout_input = torch.ones(10)

dropout_model.train()
train_output_1 = dropout_model(dropout_input)
train_output_2 = dropout_model(dropout_input)

dropout_model.eval()
eval_output_1 = dropout_model(dropout_input)
eval_output_2 = dropout_model(dropout_input)

print(f"train 模式第 1 次输出: {train_output_1}")
print(f"train 模式第 2 次输出: {train_output_2}")
print(f"eval  模式第 1 次输出: {eval_output_1}")
print(f"eval  模式第 2 次输出: {eval_output_2}")
print("解释：Dropout 在 train 模式随机生效，在 eval 模式直接返回输入")

print(f"\n切到 eval 后，参数仍然可以求导吗？")
print(f"linear.weight.requires_grad = {train_model.linear.weight.requires_grad}")
print("解释：eval() 不等于 no_grad()，它只切换模块的训练/推理行为")


# ============================================================================
#  第 6 部分：buffer 是什么
# ============================================================================
#  Parameter 是需要学习的权重，会被 optimizer 更新。
#  buffer 是模型状态，但不是可训练参数，例如：
#    - BatchNorm 的 running_mean / running_var
#    - attention mask 的固定缓存
#    - 位置编码里不需要训练的常量
#
#  buffer 会出现在 state_dict 中，并且会跟随 model.to(device/dtype) 移动。

print("\n" + "=" * 60)
print("6. buffer 是什么")
print("=" * 60)


class ScaleWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

        # register_buffer 注册的是“模型状态”，但不是可训练参数。
        self.register_buffer("offset", torch.tensor(1.0))

    def forward(self, x):
        return x * self.weight + self.offset


buffer_model = ScaleWithBuffer()

print("named_parameters():")
for name, param in buffer_model.named_parameters():
    print(f"  {name}: {param}")

print("\nnamed_buffers():")
for name, buffer in buffer_model.named_buffers():
    print(f"  {name}: {buffer}")

print("\nstate_dict():")
for name, tensor in buffer_model.state_dict().items():
    print(f"  {name}: {tensor}")


# ============================================================================
#  第 7 部分：device / dtype 迁移
# ============================================================================
#  对 Module 调用 .to(...) 会递归移动所有子模块中的 Parameter 和 buffer。
#  这比手动移动每个 Tensor 更可靠。

print("\n" + "=" * 60)
print("7. device / dtype 迁移")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

train_model = train_model.to(device=device, dtype=dtype)
x_device = x_train.to(device=device, dtype=dtype)

with torch.inference_mode():
    pred_device = train_model(x_device)

print(f"当前 device: {device}")
print(f"模型 weight device: {train_model.linear.weight.device}")
print(f"输入 x_device device: {x_device.device}")
print(f"输出 pred_device device: {pred_device.device}")


# ============================================================================
#  第 8 部分：保存和加载模型权重
# ============================================================================
#  推荐保存 state_dict，而不是直接 torch.save(model)。
#  保存整个模型对象会依赖 Python 类路径，项目结构变化后更容易出问题。

print("\n" + "=" * 60)
print("8. 保存和加载模型权重")
print("=" * 60)

checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
checkpoint_path = checkpoint_dir / "linear_regression_state_dict.pt"

# 保存前先移回 CPU。这样 checkpoint 在没有 GPU 的机器上也更容易加载。
torch.save(train_model.cpu().state_dict(), checkpoint_path)
print(f"已保存到: {checkpoint_path}")

loaded_model = LinearRegressionModel()
loaded_state = torch.load(checkpoint_path, map_location="cpu")
load_result = loaded_model.load_state_dict(loaded_state)

print(f"load_state_dict 结果: {load_result}")

with torch.inference_mode():
    original_pred = train_model.cpu()(x_train)
    loaded_pred = loaded_model(x_train)

print(f"加载前后输出是否一致: {torch.allclose(original_pred, loaded_pred)}")
print(f"loaded_model 预测:\n{loaded_pred}")

print("\n案例 03 完成。下一个案例可以进入 Dataset/DataLoader 或手写小型网络结构。")
