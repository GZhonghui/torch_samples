"""
=============================================================================
  PyTorch 案例 01：Tensor 基础操作
=============================================================================
  Tensor（张量）是 PyTorch 最核心的数据结构，类似于 NumPy 的 ndarray，
  但额外支持 GPU 加速和自动求导。

  学习路线（按代码分块）：
    1. 创建 Tensor 的多种方式
    2. Tensor 的基本属性
    3. 索引、切片、变形
    4. 数学运算
    5. 广播机制
    6. CPU / GPU 切换
    7. 与 NumPy 互操作
    8. 自动求导初体验（Autograd 预览）

  运行方式：
    python tensor_basics.py
=============================================================================
"""

import torch
import numpy as np


# ============================================================================
#  第 1 部分：创建 Tensor
# ============================================================================
#  PyTorch 提供了多种创建 Tensor 的方式，最常用的如下。

print("=" * 60)
print("1. 创建 Tensor")
print("=" * 60)

# ---- 1.1 从 Python 列表直接创建 ----
#  torch.tensor() 会根据输入数据推断元素类型（dtype）
data = [[1, 2], [3, 4]]
x = torch.tensor(data)
print(f"从列表创建:\n{x}")
# 输出：tensor([[1, 2],
#               [3, 4]])

# ---- 1.2 全零 / 全一 / 未初始化 ----
zeros = torch.zeros(2, 3)          # 2 行 3 列，全是 0
ones  = torch.ones(2, 3)           # 2 行 3 列，全是 1
empty = torch.empty(2, 3)          # 2 行 3 列，内容随机（未初始化，不保证为 0）

print(f"\n全零 (2x3):\n{zeros}")
print(f"\n全一 (2x3):\n{ones}")
print(f"\n未初始化 (2x3):\n{empty}")

# ---- 1.3 随机 Tensor ----
rand  = torch.rand(2, 3)           # [0, 1) 均匀分布
randn = torch.randn(2, 3)          # 标准正态分布 N(0, 1)
randint = torch.randint(0, 10, (2, 3))  # [0, 10) 随机整数

print(f"\n均匀分布 rand:\n{rand}")
print(f"\n正态分布 randn:\n{randn}")
print(f"\n随机整数 randint:\n{randint}")

# ---- 1.4 等差数列 ----
#  torch.arange(start, end, step)：类似 Python 的 range，但不包含 end
arange = torch.arange(0, 10, 2)    # 0, 2, 4, 6, 8
#  torch.linspace(start, end, steps)：均匀分成 steps 份，包含 end
linspace = torch.linspace(0, 1, 5) # 0.00, 0.25, 0.50, 0.75, 1.00

print(f"\narange (0~10, step=2): {arange}")
print(f"linspace (0~1, 5 份): {linspace}")

# ---- 1.5 单位矩阵 ----
eye = torch.eye(3)                 # 3x3 单位矩阵，对角线为 1
print(f"\n单位矩阵 (3x3):\n{eye}")

# ---- 1.6 指定数据类型 ----
#  dtype 参数可以控制精度：torch.float32（默认）、torch.float64、torch.int32 等
float64_tensor = torch.ones(2, 3, dtype=torch.float64)
print(f"\n指定 dtype=float64:\n{float64_tensor}  → dtype={float64_tensor.dtype}")


# ============================================================================
#  第 2 部分：Tensor 的基本属性
# ============================================================================
#  每个 Tensor 都有几个关键属性，理解它们对 debug 非常重要。

print("\n" + "=" * 60)
print("2. Tensor 属性")
print("=" * 60)

t = torch.randn(3, 4, 5)  # 创建一个 3 维 Tensor（形状 3x4x5）

print(f"t.shape   = {t.shape}")     # 形状，等价于 t.size()
print(f"t.ndim    = {t.ndim}")      # 维度数（这里是 3）
print(f"t.dtype   = {t.dtype}")     # 元素类型（默认 float32）
print(f"t.device  = {t.device}")    # 存储位置（cpu 或 cuda:0）


# ============================================================================
#  第 3 部分：索引、切片、变形
# ============================================================================

print("\n" + "=" * 60)
print("3. 索引、切片、变形")
print("=" * 60)

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(f"原始 Tensor:\n{x}")

# ---- 3.1 索引（和 NumPy 语法基本一致）----
print(f"\nx[0, 1]      = {x[0, 1]}")       # 第 0 行第 1 列 → 2
print(f"x[1, :]      = {x[1, :]}")         # 第 1 行所有列 → [4, 5, 6]
print(f"x[:, 2]      = {x[:, 2]}")         # 所有行的第 2 列 → [3, 6, 9]
print(f"x[:2, 1:]    = \n{x[:2, 1:]}")     # 前 2 行 × 后 2 列

# ---- 3.2 条件索引（布尔索引）----
# x 是一个 tensor，所以可以使用这个运算符（由 pytorch 重载）
mask = x > 4                              # 生成 bool 矩阵
print(f"\nx > 4 (mask):\n{mask}")
print(f"mask's type is {mask.dtype}")  # bool 类型
# x 和 mask 的 shape 需要一致，得到的结果是 1d 的
print(f"x[x > 4]:       {x[mask]}")       # 用 mask 提取满足条件的元素 → [5, 6, 7, 8, 9]

# ---- 3.3 变形（reshape / view / unsqueeze / squeeze）----
#  reshape：返回新形状的 Tensor（需要时复制数据）
#  view：   返回新形状的 Tensor（共享同一块内存，要求数据连续）
reshaped = x.reshape(1, 9)    # 变成 1 行 9 列
viewed   = x.view(9)          # 变成 1 维，9 个元素

print(f"\nreshape(1, 9): {reshaped}")
print(f"view(9):       {viewed}")

# 思考
# 如何理解 reshape 和 view 的区别？
# view：只换解释方式，不搬数据。我要求你只改 shape，不要复制数据。做不到就报错
# reshape：能不搬就不搬，实在不行就复制一份再换形状。我想要这个 shape。能不复制最好，不能的话你复制也行。
# 当一个 Tensor 是连续的（contiguous）时，view 和 reshape 的结果一样，且都不复制数据
# 但当 Tensor 是非连续的（例如经过转置或切片后），view 就无法工作（会报错），而 reshape 会自动复制数据以满足要求
# reshape 在可能时返回 view，否则返回 copy；并且不要依赖它到底是 view 还是 copy

# 用 untyped_storage().data_ptr() 可以观察 Tensor 是否可能共享同一块底层内存。
# 注意：它只是教学调试用，实际业务代码一般不需要直接看 data_ptr。
base = torch.arange(12).reshape(3, 4)
base_view = base.view(2, 6)
base_reshape = base.reshape(2, 6)

print("\n--- view vs reshape：连续 Tensor ---")
print(f"base:\n{base}")
print(f"base.is_contiguous()         = {base.is_contiguous()}") # 判断是否连续
print(f"base.stride()                = {base.stride()}") # stride 是步长，告诉我们在内存中如何跳跃访问元素。对于连续 Tensor，stride 的值是递减的，且最后一个维度的 stride 是 1。
print(f"base_view.shape              = {base_view.shape}")
print(f"base_reshape.shape           = {base_reshape.shape}")
# id 是 Python 对象的身份标识。view / reshape 返回的是新的 Tensor 对象，
# 所以 id 通常不同；但它们仍然可能共享同一块底层数据。
print(f"id(base)                     = {id(base)}")
print(f"id(base_view)                = {id(base_view)}")
print(f"id(base_reshape)             = {id(base_reshape)}")
print(f"base is base_view?           = {base is base_view}")
print(f"base is base_reshape?        = {base is base_reshape}")
# 通过判断起始地址的方式来验证它们是否共享内存
# 共享内存也就是说，修改 base 的数据会影响 base_view 和 base_reshape，反之亦然。
print(f"base 和 base_view 共享内存?    {base.untyped_storage().data_ptr() == base_view.untyped_storage().data_ptr()}")
print(f"base 和 base_reshape 共享内存? {base.untyped_storage().data_ptr() == base_reshape.untyped_storage().data_ptr()}")

# transpose / T 会交换维度，但通常不会重新排列底层数据；（transpose() 是典型的 view 操作，不发生数据移动）
# 它只是改变 shape 和 stride，所以结果经常是非连续的。
non_contiguous = base.T

print("\n--- 制造非连续 Tensor：transpose / T ---")
print(f"non_contiguous = base.T:\n{non_contiguous}")
print(f"non_contiguous.is_contiguous() = {non_contiguous.is_contiguous()}")
print(f"non_contiguous.stride()        = {non_contiguous.stride()}")

print("\n--- 非连续 Tensor 上尝试 view / reshape ---")
try:
    print(non_contiguous.view(2, 6))
except RuntimeError as e:
    print(f"non_contiguous.view(2, 6) 报错: {e}")

reshaped_from_non_contiguous = non_contiguous.reshape(2, 6)
print(f"non_contiguous.reshape(2, 6):\n{reshaped_from_non_contiguous}")
print(f"reshape 后是否连续? {reshaped_from_non_contiguous.is_contiguous()}")
print(
    "reshape 结果和 non_contiguous 共享内存? "
    f"{non_contiguous.untyped_storage().data_ptr() == reshaped_from_non_contiguous.untyped_storage().data_ptr()}"
)

# 思考
# 如何理解 stride（步长）？
# stride 是一个元组，告诉我们在内存中如何跳跃访问元素。
# 对于连续 Tensor，stride 的值是递减的，且最后一个维度的 stride 是 1。
# 交换两个维度，就会交换对应的 stride，从而导致数据访问模式改变，变成非连续的。（数据是没变的，访问方式变了）

# contiguous() 会按当前逻辑顺序复制一份连续内存；
# 有了连续内存后，就可以安全使用 view。
made_contiguous = non_contiguous.contiguous()
view_after_contiguous = made_contiguous.view(2, 6)

print("\n--- 用 contiguous() 把非连续 Tensor 变连续 ---")
print(f"made_contiguous:\n{made_contiguous}")
print(f"made_contiguous.is_contiguous() = {made_contiguous.is_contiguous()}")
print(f"made_contiguous.stride()        = {made_contiguous.stride()}")
print(f"made_contiguous.view(2, 6):\n{view_after_contiguous}")
print(
    "made_contiguous 和原 non_contiguous 共享内存? "
    f"{made_contiguous.untyped_storage().data_ptr() == non_contiguous.untyped_storage().data_ptr()}"
)

#  unsqueeze：在指定位置插入一个大小为 1 的维度
#  squeeze：  移除所有大小为 1 的维度
a = torch.tensor([1, 2, 3])          # shape (3,)
print(f"\n原始:      shape={a.shape}")
print(f"unsqueeze(0): shape={a.unsqueeze(0).shape}")   # (1, 3)
print(f"unsqueeze(1): shape={a.unsqueeze(1).shape}")   # (3, 1)

# unsqueeze 的等价写法（切片语法）：
_ = a[None, :]     # 等价于 a.unsqueeze(0)
_ = a[:, None]     # 等价于 a.unsqueeze(1)

# 思考
# unsqueeze是插入一个为1的维度，reshape不是可以直接做到吗？
# 很多情况下 reshape 能做到和 unsqueeze 一样的事。但两者语义和安全性不同
# unsqueeze(dim) 的意思非常明确：只在指定位置插入一个大小为 1 的维度
# 而且，unsqueeze 通常只是创建一个 view，不复制数据，和原 tensor 共享底层存储
# 而 reshape 的意思是：把 tensor 改成一个新的整体形状，只要求元素总数不变
# reshape 也可能返回 view，但在某些情况下可能会复制数据，尤其是原 tensor 非连续时。它的语义更宽泛

b = torch.zeros(1, 2, 1, 3)          # shape (1, 2, 1, 3)
print(f"\nsqueeze 前:  {b.shape}")
print(f"squeeze 后:  {b.squeeze().shape}")             # 去掉所有 1 → (2, 3)

# ---- 3.4 转置 / 维度重排 ----
y = torch.randn(2, 3)                # shape (2, 3)
print(f"\n转置前 (2, 3):\n{y}")
print(f"转置后 (3, 2):\n{y.T}")       # 或 y.transpose(0, 1)

#  permute：通用的维度重排（对 3+ 维尤其有用）
z = torch.randn(2, 3, 4)             # shape (2, 3, 4)
print(f"\npermute 前 shape: {z.shape}")
print(f"permute(2, 0, 1) shape: {z.permute(2, 0, 1).shape}")  # → (4, 2, 3)


# ============================================================================
#  第 4 部分：数学运算
# ============================================================================

print("\n" + "=" * 60)
print("4. 数学运算")
print("=" * 60)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# ---- 4.1 逐元素运算（element-wise）----
print(f"a + b   = {a + b}")             # 加法
print(f"a - b   = {a - b}")             # 减法
print(f"a * b   = {a * b}")             # 逐元素乘法（不是矩阵乘法！）
print(f"a / b   = {a / b}")             # 除法
print(f"a ** 2  = {a ** 2}")            # 幂

# ---- 4.2 矩阵乘法 ----
#  torch.matmul 或 @ 运算符
A = torch.randn(2, 3)                   # shape (2, 3)
B = torch.randn(3, 2)                   # shape (3, 2)
C = torch.matmul(A, B)                  # 矩阵乘法，(2, 3) × (3, 2) → (2, 2)
D = A @ B                               # 等价写法，更简洁

print(f"\n矩阵乘法: (2,3) × (3,2) → (2,2):\n{C}")
# allclose() 用于比较两个 Tensor 是否元素近似相等（考虑浮点误差）
print(f"@ 运算符结果相同: {torch.allclose(C, D)}")  # True

# ---- 4.3 聚合函数 ----
x = torch.randn(3, 4)
print(f"\n原始 Tensor (3x4):\n{x}")
print(f"x.sum():         {x.sum():.4f}")          # 所有元素求和
print(f"x.mean():        {x.mean():.4f}")         # 均值
print(f"x.max():         {x.max():.4f}")          # 最大值
print(f"x.min():         {x.min():.4f}")          # 最小值
print(f"x.sum(dim=0):    {x.sum(dim=0)}")         # 沿第 0 维（行方向）求和 → shape (4,)
print(f"x.sum(dim=1):    {x.sum(dim=1)}")         # 沿第 1 维（列方向）求和 → shape (3,)
#  dim 参数的含义："把哪一维消掉"
#  dim=0 → 沿行方向压缩，每列单独求和，结果形状 (4,)
#  dim=1 → 沿列方向压缩，每行单独求和，结果形状 (3,)

# ---- 4.4 原地操作（in-place）----
#  方法名带下划线 _ 的是原地操作，直接修改原 Tensor，不创建新对象
y = torch.tensor([1.0, 2.0, 3.0])
print(f"\n原地操作前: {y}")
y.add_(5)                                # 等价于 y = y + 5，但不创建新 Tensor
print(f"y.add_(5) 后: {y}")             # [6.0, 7.0, 8.0]
# 小心：原地操作会破坏计算图，在需要反向传播时避免使用带 _ 的方法


# ============================================================================
#  第 5 部分：广播机制（Broadcasting）
# ============================================================================
#  不同形状的 Tensor 在进行运算时，PyTorch 会自动"广播"较小的 Tensor，
#  使其形状与较大的 Tensor 兼容。
#  规则（从尾到头比较）：
#    1. 维度数不同时，在维度少的 Tensor 形状前面补 1
#    2. 每个维度要么相等，要么其中一个为 1

print("\n" + "=" * 60)
print("5. 广播机制")
print("=" * 60)

# 示例：(3, 1) + (1, 4) → 广播为 (3, 4) + (3, 4)
A = torch.ones(3, 1)                     # shape (3, 1)
B = torch.arange(1, 5).reshape(1, 4)     # shape (1, 4)，值为 [[1, 2, 3, 4]]
C = A + B                                # 广播后 shape (3, 4)

print(f"A (3,1):\n{A}")
print(f"\nB (1,4):\n{B}")
print(f"\nA + B → (3,4):\n{C}")
# 结果：A 的每一列都是 1，每一行复制 4 次；
#       B 的每一行都是 [1,2,3,4]，复制 3 行

# 标量与 Tensor 的广播
x = torch.arange(1, 6)                   # [1, 2, 3, 4, 5]
print(f"\n标量广播: {x} + 10 = {x + 10}")  # 10 被广播到和 x 一样大


# 思考
# 关于广播的两条规则，和numpy的广播规则是一样的。

# ============================================================================
#  第 6 部分：CPU ↔ GPU
# ============================================================================

print("\n" + "=" * 60)
print("6. CPU / GPU 切换")
print("=" * 60)

# 检查 GPU 是否可用
if torch.cuda.is_available():
    # 创建设备
    device = torch.device("cuda")
    print(f"GPU 可用: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("没有 GPU，使用 CPU")

# ---- 6.1 把 Tensor 移到 GPU ----
# 默认情况下，Tensor 创建在 CPU 上。要使用 GPU 加速，需要把 Tensor 移到 GPU。
x_cpu = torch.randn(3, 3)
x_gpu = x_cpu.to(device)                 # .to() 返回新 Tensor，原 Tensor 不变
# 因为是不同的设备，所以肯定是不共享内存的，修改 x_gpu 不会影响 x_cpu，反之亦然。
print(f"\nCPU Tensor device: {x_cpu.device}")
print(f"移动后 device:     {x_gpu.device}")

# ---- 6.2 直接在 GPU 上创建 ----
x_direct = torch.randn(3, 3, device=device) # 直接指定 device 参数
print(f"直接在 GPU 创建:   {x_direct.device}")

# ---- 6.3 移回 CPU ----
x_back = x_gpu.cpu()                     # 等价于 x_gpu.to("cpu")
print(f"移回 CPU:          {x_back.device}")


# ============================================================================
#  第 7 部分：与 NumPy 互操作
# ============================================================================

print("\n" + "=" * 60)
print("7. NumPy ↔ Tensor")
print("=" * 60)

# ---- 7.1 Tensor → NumPy ----
t = torch.ones(5)
n = t.numpy()                            # 转为 NumPy ndarray
print(f"Tensor: {t}")
print(f"NumPy:  {n}")
print(f"type:   {type(n)}")
# 注意：t 和 n 共享同一块内存！修改 t 会影响 n，反之亦然。
t.add_(1)                                # 原地 +1
print(f"\nTensor +1 后: {t}")
print(f"NumPy 同步变化: {n}")            # n 也变成了全 2

# ---- 7.2 NumPy → Tensor ----
arr = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(arr)        # 同样共享内存
t_also = torch.tensor(arr)               # 这个会复制数据，不共享内存

print(f"\nfrom_numpy 创建: {t_from_np}")
print(f"tensor() 创建:    {t_also}")
#  torch.from_numpy() 速度更快（零拷贝），但共享内存有副作用
#  torch.tensor() 更安全（独立拷贝），适合不确定是否会被修改的场景
# 注意这两个操作的区别


# ============================================================================
#  第 8 部分：自动求导初体验（Autograd 预览）
# ============================================================================
#  这是 PyTorch 最强大的功能之一：自动计算梯度。
#  下个案例会深入讲解，这里只做一个最简单的演示。

print("\n" + "=" * 60)
print("8. 自动求导预览")
print("=" * 60)

#  requires_grad=True 告诉 PyTorch："我要对这个变量求梯度"
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# 定义函数：y = w * x + b
y = w * x + b

# 反向传播：计算 dy/dw, dy/dx, dy/db
y.backward()

#  .grad 属性存储了梯度值
#  dy/dx = w = 3.0
#  dy/dw = x = 2.0
#  dy/db = 1 = 1.0
print(f"y = {w.item()} * {x.item()} + {b.item()} = {y.item()}")
print(f"dy/dx (即 x.grad) = {x.grad.item()}")   # 3.0
print(f"dy/dw (即 w.grad) = {w.grad.item()}")   # 2.0
print(f"dy/db (即 b.grad) = {b.grad.item()}")   # 1.0

print("\n案例 01 完成！你已经掌握了 Tensor 的基础操作。")
print("   下一个案例：自动求导与线性回归。")

# 思考
# 还是要先理解导数的概念
# 自动求导的功能我暂时还不太用得到