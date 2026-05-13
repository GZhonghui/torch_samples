"""
=============================================================================
  PyTorch 案例 02：Autograd 自动求导基础
=============================================================================
  Autograd 是 PyTorch 的自动求导系统。只要 Tensor 设置了
  requires_grad=True，PyTorch 就会记录基于它产生的运算，并在 backward()
  时根据计算图自动计算梯度。

  学习路线（按代码分块）：
    1. 标量函数求导
    2. 向量 / 矩阵求导：为什么 backward 通常需要标量
    3. 梯度会累积：为什么训练循环里要 zero_grad
    4. leaf Tensor 与非 leaf Tensor
    5. no_grad / inference_mode / detach
    6. 用 Autograd 手写一个最小线性回归训练循环

  运行方式：
    ../torch_venv/bin/python autograd_basics.py
=============================================================================
"""

import warnings

import torch


torch.manual_seed(0)


# ============================================================================
#  第 1 部分：标量函数求导
# ============================================================================
#  对一个简单函数：
#      y = x^2 + 3x + 1
#  它的导数是：
#      dy/dx = 2x + 3
#  当 x = 2 时，dy/dx = 7。

print("=" * 60)
print("1. 标量函数求导")
print("=" * 60)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

print(f"x = {x}")
print(f"y = x^2 + 3x + 1 = {y}")
print(f"y.requires_grad = {y.requires_grad}")
print(f"y.grad_fn       = {y.grad_fn}")  # grad_fn 记录 y 是由什么运算产生的

y.backward()  # 从 y 开始反向传播，计算 dy/dx

print(f"x.grad = {x.grad}")  # 7.0
print("解释：x.grad 保存的是 dy/dx；当 x=2 时，2*x+3=7")


# ============================================================================
#  第 2 部分：向量 / 矩阵求导
# ============================================================================
#  PyTorch 的 backward() 默认适合“标量输出”。
#  深度学习训练里通常也是先把一批样本的 loss 聚合成一个标量，
#  再调用 loss.backward()。

print("\n" + "=" * 60)
print("2. 向量 / 矩阵求导")
print("=" * 60)

v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out = v ** 2

print(f"v = {v}")
print(f"out = v ** 2 = {out}")

# out 是一个向量，不能直接 out.backward()，因为 PyTorch 不知道你想对
# 哪个标量目标求导。常见做法是先聚合成标量。
loss = out.sum()
loss.backward()

print(f"loss = out.sum() = {loss}")
print(f"v.grad = {v.grad}")  # d(v^2 的和)/dv = 2v = [2, 4, 6]

# 如果确实想对向量输出调用 backward，需要手动传入同 shape 的 gradient。
# 这等价于先计算加权和：
#     weighted = out[0]*1 + out[1]*0.1 + out[2]*0.01
v2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
out2 = v2 ** 2
external_grad = torch.tensor([1.0, 0.1, 0.01])
out2.backward(gradient=external_grad)

print(f"\n对向量 out2 传入 external_grad = {external_grad}")
print(f"v2.grad = {v2.grad}")
print("解释：v2.grad = 2*v2*external_grad")


# ============================================================================
#  第 3 部分：梯度会累积
# ============================================================================
#  backward() 不会自动清空 .grad，而是把新梯度加到已有梯度上。
#  这是 PyTorch 的设计：有些场景需要累积多个 loss 的梯度。
#  所以训练循环里通常每一步都要先 optimizer.zero_grad()。

print("\n" + "=" * 60)
print("3. 梯度会累积")
print("=" * 60)

w = torch.tensor(2.0, requires_grad=True)

loss1 = w ** 2      # dloss1/dw = 2w = 4
loss1.backward()
print(f"第一次 backward 后 w.grad = {w.grad}")

loss2 = w ** 3      # dloss2/dw = 3w^2 = 12
loss2.backward()
print(f"第二次 backward 后 w.grad = {w.grad}")
print("解释：4 + 12 = 16，梯度被累积了")

w.grad.zero_()
print(f"手动 w.grad.zero_() 后 w.grad = {w.grad}")


# ============================================================================
#  第 4 部分：leaf Tensor 与非 leaf Tensor
# ============================================================================
#  leaf Tensor 通常是用户直接创建的、requires_grad=True 的 Tensor。
#  PyTorch 默认只把梯度保存到 leaf Tensor 的 .grad 上。
#  中间结果虽然参与反向传播，但默认不保存 .grad。

print("\n" + "=" * 60)
print("4. leaf Tensor 与非 leaf Tensor")
print("=" * 60)

a = torch.tensor(3.0, requires_grad=True)  # leaf Tensor
b = a * 2                                 # 非 leaf Tensor，是运算结果
c = b ** 2

print(f"a.is_leaf = {a.is_leaf}")
print(f"b.is_leaf = {b.is_leaf}")
print(f"c.is_leaf = {c.is_leaf}")

c.backward()

print(f"a.grad = {a.grad}")
# 直接访问非 leaf Tensor 的 .grad 时，PyTorch 会提醒你：
# 默认不会为非 leaf Tensor 保存 .grad。这里为了演示，临时隐藏这条警告。
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    b_grad = b.grad
print(f"b.grad = {b_grad}")
print("解释：b 参与了计算图，但默认不会在 b.grad 保存梯度")

# 如果确实想观察中间 Tensor 的梯度，可以调用 retain_grad()。
a2 = torch.tensor(3.0, requires_grad=True)
b2 = a2 * 2
b2.retain_grad()
c2 = b2 ** 2
c2.backward()

print(f"\n调用 b2.retain_grad() 后：")
print(f"a2.grad = {a2.grad}")
print(f"b2.grad = {b2.grad}")


# ============================================================================
#  第 5 部分：no_grad / inference_mode / detach
# ============================================================================
#  no_grad 和 inference_mode 都表示“不记录计算图”，常用于推理。
#  inference_mode 更激进，通常推理时更省资源；训练时不要包住 forward。
#  detach() 会返回一个和原 Tensor 共享数据、但脱离当前计算图的新 Tensor。

print("\n" + "=" * 60)
print("5. no_grad / inference_mode / detach")
print("=" * 60)

p = torch.tensor(2.0, requires_grad=True)
q = p * 3
r = q.detach()

print(f"q.requires_grad = {q.requires_grad}")
print(f"r.requires_grad = {r.requires_grad}")
print("解释：r 从 q 分离出来，不再追踪 q 之前的计算图")

with torch.no_grad():
    no_grad_value = p * 3

with torch.inference_mode():
    inference_value = p * 3

print(f"no_grad_value.requires_grad     = {no_grad_value.requires_grad}")
print(f"inference_value.requires_grad   = {inference_value.requires_grad}")

# 常见用途：参数更新时不希望更新操作本身被记录进计算图。
param = torch.tensor(1.0, requires_grad=True)
loss = (param - 5) ** 2
loss.backward()

print(f"\n更新前 param = {param.item():.4f}, grad = {param.grad.item():.4f}")
with torch.no_grad():
    param -= 0.1 * param.grad
print(f"更新后 param = {param.item():.4f}")


# ============================================================================
#  第 6 部分：手写一个最小线性回归训练循环
# ============================================================================
#  这里不用 nn.Module 和 optimizer，先只看 Autograd 的核心训练流程：
#      1. forward：用当前参数计算预测值
#      2. loss：计算预测值和真实值的误差
#      3. backward：计算参数梯度
#      4. update：用梯度更新参数
#      5. zero_grad：清空梯度，准备下一步

print("\n" + "=" * 60)
print("6. 手写最小线性回归")
print("=" * 60)

# 构造一组简单数据：真实关系是 y = 2*x + 1
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 初始化两个待学习参数。这里先不用 nn.Linear，方便直接观察 Autograd。
weight = torch.randn(1, 1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
learning_rate = 0.05

print(f"初始 weight = {weight.item():.4f}, bias = {bias.item():.4f}")

for step in range(1, 201):
    # forward
    y_pred = x_train @ weight + bias

    # mean squared error: ((预测 - 真实)^2).mean()
    loss = ((y_pred - y_train) ** 2).mean()

    # backward
    loss.backward()

    # update
    # 参数更新不应该被 Autograd 记录，所以放进 no_grad。
    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

        # 清空梯度。也可以写成 weight.grad.zero_() / bias.grad.zero_()。
        weight.grad = None
        bias.grad = None

    if step in [1, 2, 5, 10, 50, 100, 200]:
        print(
            f"step={step:3d} | loss={loss.item():.6f} | "
            f"weight={weight.item():.4f} | bias={bias.item():.4f}"
        )

print("\n训练完成后，weight 接近 2，bias 接近 1。")
print("这就是 PyTorch 训练的最小核心：forward -> loss -> backward -> update。")

print("\n案例 02 完成。下一个案例可以进入 nn.Module / optimizer / state_dict。")
