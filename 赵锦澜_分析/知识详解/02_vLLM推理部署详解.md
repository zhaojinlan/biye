# vLLM推理部署详解

> vLLM是当前最主流的LLM推理引擎。你现在用的是Ollama，面试官一定会问"为什么不用vLLM"以及"vLLM快在哪里"。

---

## 一、LLM推理的核心瓶颈

### 1.1 LLM推理的两个阶段

```
阶段1：Prefill（预填充）
  输入: "请问什么是高血压？"（整个prompt一次性处理）
  计算: 并行处理所有input token，生成KV-Cache
  特点: 计算密集型（GPU算力是瓶颈），类似批处理
  耗时: 取决于prompt长度

阶段2：Decode（解码/生成）
  输入: 每次只有1个新token
  计算: 新token的Q与所有历史KV做Attention → 生成下一个token
  特点: 内存密集型（显存带宽是瓶颈），因为每次只算1个token但要读取全部KV-Cache
  耗时: 取决于生成长度

关键瓶颈：
  Prefill → GPU算力利用率高（好）
  Decode → GPU算力利用率极低（坏），大部分时间在等显存读取KV-Cache
  → Decode阶段是LLM推理的主要瓶颈
```

### 1.2 为什么Ollama不够快？

```
Ollama的本质：llama.cpp的封装
  优点：简单易用、CPU/GPU混合推理、GGUF量化
  缺点：
    1. 单请求处理，不支持并发batch
    2. 没有PagedAttention，显存利用率低
    3. 主要针对个人使用场景，不适合生产环境

对比：
  Ollama：适合本地开发/测试（你的医疗项目用它没问题）
  vLLM：适合生产部署/高并发服务

面试答法：
  "开发阶段用Ollama方便调试，生产环境应该切换到vLLM，
   因为vLLM的PagedAttention和Continuous Batching能显著提升吞吐量。"
```

---

## 二、vLLM核心技术

### 2.1 PagedAttention（vLLM的核心创新）

**问题：KV-Cache的内存管理**

```
传统方式：为每个请求预分配固定大小的连续显存

例如：max_seq_len = 2048，每个token的KV大小 = 0.5MB
  预分配: 2048 × 0.5MB = 1GB（每个请求）

问题：
  - 请求实际只用了200个token → 浪费90%显存
  - 必须连续内存 → 显存碎片化严重
  - 能同时处理的请求数很少
```

**PagedAttention的解决方案：借鉴操作系统的虚拟内存/分页机制**

```
核心思想：把KV-Cache分成固定大小的"页"（block），按需分配，不要求连续

类比操作系统：
  传统方式 = 连续内存分配（malloc一大块）→ 内存碎片
  PagedAttention = 分页内存管理（虚拟内存）→ 页表映射，按需分配

具体实现：
  1. 将KV-Cache分成固定大小的block（如每block存16个token的KV）
  2. 每个请求维护一个block table（类似页表）
  3. 生成新token时，只分配需要的block
  4. 请求结束后，释放block供其他请求使用

┌─────────────────────────────────┐
│         GPU显存                   │
│  ┌──────┐ ┌──────┐ ┌──────┐     │
│  │Block0│ │Block1│ │Block2│ ... │
│  │请求A │ │请求B │ │请求A │     │
│  └──────┘ └──────┘ └──────┘     │
│                                  │
│  请求A的block table: [0, 2, 5]  │  ← block不需要连续！
│  请求B的block table: [1, 3]     │
└─────────────────────────────────┘

效果：
  - 显存利用率从 ~50% 提升到 ~95%
  - 能同时处理的请求数提升2-4倍
  - 几乎消除显存碎片
```

### 2.2 Continuous Batching（连续批处理）

**问题：传统Static Batching的低效**

```
Static Batching（传统方式，如Ollama）：

  时间 →  t1    t2    t3    t4    t5    t6
  请求A: [生成] [生成] [生成] [完成] [等待B] [等待B]
  请求B: [生成] [生成] [生成] [生成] [生成]  [完成]
  请求C: [排队] [排队] [排队] [排队] [排队]  [排队]  ← 必须等A和B都完成

  问题：
  - 请求A在t4完成了，但要等请求B完成才能处理新请求
  - GPU在t4-t6对请求A的"槽位"是空闲的 → 浪费算力
  - 请求C一直在排队
```

```
Continuous Batching（vLLM的方式）：

  时间 →  t1    t2    t3    t4    t5    t6
  请求A: [生成] [生成] [生成] [完成]
  请求B: [生成] [生成] [生成] [生成] [生成]  [完成]
  请求C: [排队] [排队] [排队] [生成] [生成]  [生成]  ← A完成后立即插入！

  核心：每个iteration都检查
  - 有请求完成了？→ 释放slot
  - 有新请求等待？→ 立即插入batch
  
  → GPU利用率大幅提升，吞吐量提升2-3倍
```

### 2.3 其他优化技术

```
Speculative Decoding（投机解码）：
  用一个小模型（draft model）快速生成多个候选token
  再用大模型（target model）一次性验证
  如果候选token正确 → 一步生成了多个token
  如果错误 → 回退到大模型重新生成
  → 在不降低质量的前提下加速1.5-2倍

Tensor Parallelism（张量并行）：
  将模型权重分到多张GPU上并行计算
  → 单机多卡推理

Prefix Caching：
  如果多个请求有相同的前缀（如system prompt）
  → 缓存公共前缀的KV-Cache，避免重复计算
  → 对你的医疗系统很有用（所有请求都有相同的系统提示词）

Chunked Prefill：
  将长prompt分块处理，和Decode请求交替执行
  → 避免长prompt的Prefill阶段独占GPU
```

---

## 三、vLLM vs 其他推理引擎

| 特性 | vLLM | Ollama (llama.cpp) | TGI (HuggingFace) | TensorRT-LLM (NVIDIA) |
|------|------|-------|-----|--------------|
| **PagedAttention** | ✅ | ❌ | ✅ | ✅ |
| **Continuous Batching** | ✅ | ❌ | ✅ | ✅ |
| **CPU推理** | ❌ | ✅ | ❌ | ❌ |
| **量化支持** | GPTQ/AWQ/FP8 | GGUF(多种) | GPTQ/AWQ | FP8/INT8/INT4 |
| **易用性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **吞吐量** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **适用场景** | 通用生产部署 | 个人/开发 | HF生态 | NVIDIA优化场景 |
| **OpenAI兼容API** | ✅ | ✅ | ✅ | ✅ |

**面试答法**：
> "如果是开发/原型阶段，Ollama最方便；生产部署首选vLLM，因为PagedAttention和Continuous Batching在高并发下吞吐量是Ollama的3-5倍；如果追求极致性能且用NVIDIA GPU，可以考虑TensorRT-LLM，但部署复杂度更高。"

---

## 四、模型量化（面试常问）

### 4.1 为什么要量化？

```
Qwen2.5-7B 参数量：7B（70亿参数）

FP16精度：7B × 2字节 = 14GB 显存（仅模型权重）
+ KV-Cache + 中间计算 → 实际需要 ~20GB 显存
→ 需要至少一张A100 40G或两张3090 24G

INT4量化后：7B × 0.5字节 = 3.5GB 显存
+ KV-Cache → 实际需要 ~8GB 显存
→ 一张4060 8G就能跑

量化 = 用低精度数值存储模型权重，用空间换精度
```

### 4.2 主流量化方法对比

| 方法 | 原理 | 精度损失 | 速度 | vLLM支持 |
|------|------|----------|------|----------|
| **GPTQ** | 基于二阶信息（Hessian矩阵）的逐层量化 | 较低 | 快 | ✅ |
| **AWQ** | 保护重要权重通道（Activation-aware） | 最低 | 快 | ✅ |
| **GGUF** | llama.cpp格式，支持CPU+GPU混合 | 中等 | 中等 | ❌（Ollama用） |
| **FP8** | 8位浮点（NVIDIA Hopper架构原生） | 极低 | 最快 | ✅ |
| **BitsAndBytes** | HuggingFace集成，NF4量化 | 中等 | 慢 | ❌ |

```
选择建议：
  Ollama用户 → GGUF（Q4_K_M是平衡选择）
  vLLM生产部署 → AWQ（精度最好）或 GPTQ（生态最广）
  H100/H200 → FP8（原生支持，最快）
```

### 4.3 量化的直觉理解

```
FP16: 模型权重 = 0.123456789  → 精确存储
INT8: 模型权重 = 0.12         → 省一半空间，精度损失小
INT4: 模型权重 = 0.1          → 省3/4空间，精度损失中等

类比：
  FP16 = 用尺子量到毫米  → 精确但费空间
  INT4 = 用尺子量到厘米  → 粗糙但省空间
  
  大部分场景下，"量到厘米"已经够用了（模型效果下降<5%）
```

---

## 五、vLLM实战使用

### 5.1 安装与启动

```bash
# 安装（需要CUDA环境）
pip install vllm

# 启动API服务（OpenAI兼容格式）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --dtype auto

# 关键参数说明：
#   --max-model-len    最大序列长度（影响显存占用）
#   --gpu-memory-utilization  GPU显存使用比例（0.9=使用90%）
#   --dtype auto       自动选择精度（FP16/BF16）
#   --tensor-parallel-size 2  使用2张GPU并行
#   --quantization awq       使用AWQ量化模型
```

### 5.2 使用AWQ量化模型

```bash
# 直接使用社区量化好的模型（推荐，省事）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --max-model-len 8192
```

### 5.3 调用API（和Ollama一样是OpenAI格式，你的代码几乎不用改）

```python
from openai import OpenAI

# 指向vLLM服务
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM不需要API key
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一名医学顾问"},
        {"role": "user", "content": "什么是高血压？"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### 5.4 对你的医疗项目的意义

```
当前：Ollama + Qwen2.5:14b（单用户够用）

如果要上生产：
  1. 用vLLM替代Ollama（改config.py的base_url即可）
  2. 加AWQ量化 → 显存需求减半
  3. Continuous Batching → 支持多个医生同时使用
  4. Prefix Caching → 系统提示词只计算一次

代码改动量：只需要改LLM_CONFIG的base_url
  原来: "base_url": "http://xxx:11434/v1"  (Ollama)
  改为: "base_url": "http://xxx:8000/v1"   (vLLM)
```

---

## 六、面试高频问题速答

**Q1：vLLM为什么快？**
> 两个核心技术：1）PagedAttention借鉴操作系统分页机制管理KV-Cache，将显存利用率从50%提升到95%以上；2）Continuous Batching在每个推理步都可以插入/移除请求，GPU利用率大幅提升。两者结合使吞吐量相比传统方案提升3-5倍。

**Q2：PagedAttention的原理？**
> 传统方式为每个请求预分配连续的大块显存存储KV-Cache，造成大量浪费和碎片。PagedAttention将KV-Cache分成固定大小的block，通过block table做映射，按需分配，不要求连续内存。就像操作系统用虚拟内存+页表管理物理内存一样。

**Q3：Continuous Batching和Static Batching的区别？**
> Static Batching必须等一个batch中所有请求都完成才能处理新请求；Continuous Batching在每个推理步检查，完成的请求立即释放slot，等待的请求立即插入。消除了"等最慢请求"的浪费。

**Q4：GPTQ和AWQ的区别？**
> 都是训练后量化（PTQ）方法。GPTQ基于二阶优化（Hessian矩阵）逐层量化权重；AWQ基于激活感知（观察哪些权重通道对激活值影响大），保护重要通道。AWQ通常精度略好，GPTQ生态更广。

**Q5：什么场景下不适合用vLLM？**
> 1）需要CPU推理（vLLM只支持GPU）；2）个人开发/调试（Ollama更方便）；3）需要在边缘设备/手机上跑（用llama.cpp/MLC-LLM）；4）模型不在支持列表中。

---

*建议：先在AutoDL上用vLLM部署Qwen，体验一下和Ollama的速度差异，特别是多并发场景下的吞吐量对比。*
