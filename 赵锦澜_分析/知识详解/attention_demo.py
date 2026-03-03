"""
Self-Attention 完整计算演示
============================
这个脚本用最简单的数字，一步步展示Attention的完整计算过程。
直接运行即可看到每一步的结果。

对应知识点：01_Transformer与Attention详解.md
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(x):
    """softmax函数：把任意数字变成0-1之间的概率，所有值加起来=1"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 减最大值防止数值溢出
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ============================================================
# 第一部分：最简单的例子（2个词，2维）
# ============================================================
print("=" * 60)
print("第一部分：最简单的Attention（2个词，2维向量）")
print("=" * 60)

# 输入："头痛" "高血压"
# 假设经过Embedding后的向量：
X = np.array([
    [1.0, 0.0],  # "头痛"的embedding
    [0.0, 1.0],  # "高血压"的embedding
])
print(f"\n输入X（2个词，每个2维）：")
print(f"  X_头痛   = {X[0]}")
print(f"  X_高血压 = {X[1]}")

# 预训练好的权重矩阵（这些是固定的参数）
W_Q = np.array([[1.0, 0.0],
                [0.0, 1.0]])

W_K = np.array([[0.0, 1.0],
                [1.0, 0.0]])

W_V = np.array([[1.0, 1.0],
                [0.0, 1.0]])

print(f"\n预训练好的权重矩阵（固定参数）：")
print(f"  W_Q = \n{W_Q}")
print(f"  W_K = \n{W_K}")
print(f"  W_V = \n{W_V}")

# 第1步：计算Q、K、V（动态的，每次输入不同就不同）
Q = X @ W_Q  # @是矩阵乘法
K = X @ W_K
V = X @ W_V

print(f"\n--- 第1步：Q = X × W_Q ---")
print(f"  Q_头痛   = {Q[0]}  （头痛想找什么）")
print(f"  Q_高血压 = {Q[1]}  （高血压想找什么）")

print(f"\n--- 第2步：K = X × W_K ---")
print(f"  K_头痛   = {K[0]}  （头痛是什么）")
print(f"  K_高血压 = {K[1]}  （高血压是什么）")

print(f"\n--- 第3步：V = X × W_V ---")
print(f"  V_头痛   = {V[0]}  （头痛的具体信息）")
print(f"  V_高血压 = {V[1]}  （高血压的具体信息）")

# 第2步：计算注意力分数 = Q × K^T
scores = Q @ K.T  # K.T就是K的转置（K^T）
print(f"\n--- 第4步：注意力分数 = Q × K^T ---")
print(f"  scores = \n{scores}")
print(f"  scores[0][0] = {scores[0][0]}  → 头痛对头痛的关注度")
print(f"  scores[0][1] = {scores[0][1]}  → 头痛对高血压的关注度")
print(f"  scores[1][0] = {scores[1][0]}  → 高血压对头痛的关注度")
print(f"  scores[1][1] = {scores[1][1]}  → 高血压对高血压的关注度")

# 第3步：缩放（除以√d_k）
d_k = Q.shape[-1]  # 维度=2
scaled_scores = scores / np.sqrt(d_k)
print(f"\n--- 第5步：缩放 scores / √d_k (d_k={d_k}, √d_k={np.sqrt(d_k):.4f}) ---")
print(f"  scaled_scores = \n{scaled_scores}")

# 第4步：softmax（变成概率）
attention_weights = softmax(scaled_scores)
print(f"\n--- 第6步：softmax → 注意力权重（概率） ---")
print(f"  weights = \n{attention_weights}")
print(f"  头痛的注意力分配：  自己={attention_weights[0][0]:.2%}, 高血压={attention_weights[0][1]:.2%}")
print(f"  高血压的注意力分配：头痛={attention_weights[1][0]:.2%}, 自己={attention_weights[1][1]:.2%}")

# 第5步：加权求和
output = attention_weights @ V
print(f"\n--- 第7步：输出 = 注意力权重 × V ---")
print(f"  头痛的新表示   = {attention_weights[0][0]:.4f}×{V[0]} + {attention_weights[0][1]:.4f}×{V[1]} = {output[0]}")
print(f"  高血压的新表示 = {attention_weights[1][0]:.4f}×{V[0]} + {attention_weights[1][1]:.4f}×{V[1]} = {output[1]}")
print(f"\n  → 头痛的新向量融合了高血压的信息！")


# ============================================================
# 第二部分：稍大的例子（3个词，4维）—— 更接近真实场景
# ============================================================
print("\n\n" + "=" * 60)
print("第二部分：3个词的Attention（更接近真实场景）")
print("=" * 60)

# 输入："头痛" "高血压" "症状"
X2 = np.array([
    [0.8, 0.2, 0.5, 0.1],  # "头痛"
    [0.3, 0.9, 0.1, 0.7],  # "高血压"
    [0.6, 0.4, 0.7, 0.3],  # "症状"
])

# 随机模拟预训练好的权重（真实模型中这些数字是训练出来的）
np.random.seed(42)
d_model = 4
W_Q2 = np.random.randn(d_model, d_model) * 0.5
W_K2 = np.random.randn(d_model, d_model) * 0.5
W_V2 = np.random.randn(d_model, d_model) * 0.5

print(f"\n输入X（3个词，每个4维）：")
print(f"  X_头痛   = {X2[0]}")
print(f"  X_高血压 = {X2[1]}")
print(f"  X_症状   = {X2[2]}")

# 计算Q、K、V
Q2 = X2 @ W_Q2
K2 = X2 @ W_K2
V2 = X2 @ W_V2

print(f"\nQ（每个词的'想找什么'）：")
for i, name in enumerate(["头痛", "高血压", "症状"]):
    print(f"  Q_{name} = {Q2[i]}")

print(f"\nK（每个词的'是什么'）：")
for i, name in enumerate(["头痛", "高血压", "症状"]):
    print(f"  K_{name} = {K2[i]}")

# 注意力分数
scores2 = Q2 @ K2.T / np.sqrt(d_model)
print(f"\n注意力分数（缩放后）：")
print(f"  {scores2}")

# softmax
weights2 = softmax(scores2)
print(f"\n注意力权重（softmax后）：")
names = ["头痛", "高血压", "症状"]
for i in range(3):
    print(f"  {names[i]}的注意力：", end="")
    for j in range(3):
        print(f"  {names[j]}={weights2[i][j]:.2%}", end="")
    print()

# 输出
output2 = weights2 @ V2
print(f"\n最终输出（融合了上下文信息的新向量）：")
for i, name in enumerate(["头痛", "高血压", "症状"]):
    print(f"  {name}_新 = {output2[i]}")


# ============================================================
# 第三部分：验证"相同输入 → 相同输出"
# ============================================================
print("\n\n" + "=" * 60)
print("第三部分：验证相同输入一定得到相同输出")
print("=" * 60)

# 用完全相同的输入再算一遍
Q2_again = X2 @ W_Q2
K2_again = X2 @ W_K2
V2_again = X2 @ W_V2
scores2_again = Q2_again @ K2_again.T / np.sqrt(d_model)
weights2_again = softmax(scores2_again)
output2_again = weights2_again @ V2_again

print(f"\n第一次输出：{output2[0]}")
print(f"第二次输出：{output2_again[0]}")
print(f"完全相同？ {np.allclose(output2, output2_again)} ✅")
print(f"\n→ 相同输入 + 相同权重 = 相同输出（纯数学运算，没有随机性）")


# ============================================================
# 第四部分：验证"不同输入 → 不同输出"（但用同一套权重）
# ============================================================
print("\n\n" + "=" * 60)
print("第四部分：不同输入 → 不同输出（权重不变）")
print("=" * 60)

X3 = np.array([
    [0.1, 0.5, 0.9, 0.2],  # "发热"
    [0.7, 0.3, 0.2, 0.8],  # "红肿"
    [0.4, 0.6, 0.3, 0.5],  # "疼痛"
])

# 用同一套权重！
Q3 = X3 @ W_Q2  # 注意：W_Q2没变
K3 = X3 @ W_K2
V3 = X3 @ W_V2
scores3 = Q3 @ K3.T / np.sqrt(d_model)
weights3 = softmax(scores3)
output3 = weights3 @ V3

print(f"\n输入1（头痛/高血压/症状）的输出：{output2[0]}")
print(f"输入2（发热/红肿/疼痛）的输出：  {output3[0]}")
print(f"完全相同？ {np.allclose(output2, output3)} ❌ 不同！")
print(f"\n→ 不同输入 + 同样的权重 = 不同的输出")
print(f"→ 权重（W_Q/W_K/W_V）是固定的'工具'")
print(f"→ Q/K/V是用工具处理输入后得到的'结果'，随输入变化")


# ============================================================
# 第五部分：Multi-Head Attention 简化演示
# ============================================================
print("\n\n" + "=" * 60)
print("第五部分：Multi-Head Attention（多头注意力）")
print("=" * 60)

print("""
为什么要多头？一个Attention头只能学一种"关注模式"。
多个头可以同时关注不同的关系：
  头1：关注"疾病-症状"关系
  头2：关注"位置-症状"关系（左下肢 → 红肿）
  头3：关注"时间"关系（近期 → 出现）
""")

n_heads = 2
d_head = d_model // n_heads  # 每个头的维度 = 4/2 = 2

# 把Q拆成多个头
Q_heads = Q2.reshape(3, n_heads, d_head).transpose(1, 0, 2)  # [n_heads, seq_len, d_head]
K_heads = K2.reshape(3, n_heads, d_head).transpose(1, 0, 2)
V_heads = V2.reshape(3, n_heads, d_head).transpose(1, 0, 2)

for h in range(n_heads):
    scores_h = Q_heads[h] @ K_heads[h].T / np.sqrt(d_head)
    weights_h = softmax(scores_h)
    print(f"头{h + 1}的注意力权重：")
    for i in range(3):
        print(f"  {names[i]}：", end="")
        for j in range(3):
            print(f"  {names[j]}={weights_h[i][j]:.2%}", end="")
        print()
    print()

print("→ 不同的头学到了不同的注意力模式！")
print("→ 最后把所有头的输出拼接起来，得到最终结果")


# ============================================================
# 总结
# ============================================================
print("\n\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
Attention的全部计算就这些：

  1. Q = X × W_Q    （输入 × 固定权重 → 动态Query）
  2. K = X × W_K    （输入 × 固定权重 → 动态Key）
  3. V = X × W_V    （输入 × 固定权重 → 动态Value）
  4. scores = Q × K^T / √d_k   （算相似度）
  5. weights = softmax(scores)   （变成概率）
  6. output = weights × V        （加权混合）

其中：
  W_Q, W_K, W_V → 预训练时学好的固定参数
  Q, K, V       → 每次输入动态算出来的中间结果
  
不同大模型（Qwen/LLaMA/GPT）公式完全一样，
区别只在于W里的数字不同（训练数据不同）和矩阵大小不同（模型规模不同）。
""")
