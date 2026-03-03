# LoRA与SFT微调详解

> 微调是区分"调API的应用工程师"和"懂模型的应用工程师"的分界线。这篇帮你搞懂：什么是SFT、什么是LoRA、怎么用LLaMA-Factory实战。

---

## 一、大模型训练的三个阶段

```
阶段1：预训练（Pre-training）
  目标：让模型"学会语言"
  数据：万亿token的互联网文本（网页、书籍、代码...）
  任务：Next Token Prediction（预测下一个词）
  成本：数百万美元、数千张GPU训练几个月
  产物：Base Model（基座模型），如 Qwen2.5-7B
  特点：会补全文本，但不会"对话"（你说"你好"，它可能续写"你好大的胆子"）

阶段2：有监督微调（SFT, Supervised Fine-Tuning）← 你需要学的
  目标：让模型"学会对话/遵循指令"
  数据：几万到几十万条 指令-回答 对（人工标注或合成）
  任务：学习按照指令格式回答问题
  成本：几百到几千美元、几张GPU训练几小时到几天
  产物：Chat Model，如 Qwen2.5-7B-Instruct
  特点：能理解指令、有礼貌地回答、拒绝有害请求

阶段3：对齐（Alignment）—— RLHF / DPO
  目标：让模型"更符合人类偏好"
  数据：人类偏好数据（给两个回答，标注哪个更好）
  方法：RLHF（强化学习）或 DPO（直接偏好优化）
  特点：减少幻觉、提升安全性、更自然的表达
```

**你需要掌握的重点是阶段2（SFT），因为这是应用开发中最常见的微调场景。**

---

## 二、SFT是什么？

### 2.1 核心概念

```
SFT = Supervised Fine-Tuning = 有监督微调

本质：在一个已经"会语言"的基座模型上，用特定领域/格式的数据继续训练
目的：让模型适应你的业务场景

类比：
  Base Model = 一个学了12年语文的学生（会写作文，但不会写病历）
  SFT = 给他一本《病历写作范例》让他学习
  微调后 = 他既会写作文，也会写病历了
```

### 2.2 SFT数据格式

SFT数据就是一系列"问答对"，主流有两种格式：

**Alpaca格式（单轮对话）：**
```json
[
  {
    "instruction": "请分析以下症状可能对应的疾病",
    "input": "患者出现高热、皮肤红肿、疼痛剧烈，局部有捻发音",
    "output": "根据症状分析，患者可能患有坏死性软组织感染（NSTIs）。主要依据：\n1. 高热提示严重感染\n2. 皮肤红肿伴剧烈疼痛超出表面病变范围\n3. 捻发音提示皮下产气\n建议立即进行LRINEC评分、血常规、CRP检测，并考虑急诊外科会诊。"
  },
  {
    "instruction": "解释LRINEC评分的临床意义",
    "input": "",
    "output": "LRINEC评分（Laboratory Risk Indicator for Necrotizing Fasciitis）是用于辅助诊断坏死性筋膜炎的实验室评分系统..."
  }
]
```

**ShareGPT格式（多轮对话，更常用）：**
```json
[
  {
    "conversations": [
      {"from": "system", "value": "你是一名专业的急诊医学顾问"},
      {"from": "human", "value": "患者65岁男性，糖尿病史，左下肢红肿3天，今日加重伴高热"},
      {"from": "gpt", "value": "根据您的描述，需要高度警惕坏死性软组织感染。建议立即：\n1. 完善LRINEC评分相关检查\n2. 急诊外科会诊评估是否需要手术探查\n3. 经验性抗感染治疗..."},
      {"from": "human", "value": "LRINEC评分结果为8分，该怎么处理？"},
      {"from": "gpt", "value": "LRINEC评分8分属于高危（≥6分），强烈提示坏死性软组织感染。建议：\n1. 立即启动手术探查和清创\n2. 广谱抗生素覆盖..."}
    ]
  }
]
```

### 2.3 全参数微调 vs 参数高效微调

```
全参数微调（Full Fine-Tuning）：
  更新模型的所有参数
  Qwen2.5-7B: 70亿参数全部更新
  显存需求: ~60GB（需要存储参数+梯度+优化器状态）
  → 需要4张A100 80G 或 8张3090 24G
  → 太贵了！

参数高效微调（PEFT, Parameter-Efficient Fine-Tuning）：
  只更新极少量参数，冻结大部分原始参数
  代表方法：LoRA、QLoRA、Adapter、Prefix-Tuning
  显存需求: 大幅降低
  → 一张3090/4090就能微调7B模型
```

---

## 三、LoRA详解（必须掌握）

### 3.1 LoRA的核心思想

LoRA = Low-Rank Adaptation = 低秩适配

```
核心假设：微调时，权重的变化量（ΔW）是低秩的
  即：不需要更新整个大矩阵，只需要更新一个"压缩版"的小矩阵

数学表达：
  原始权重: W₀ ∈ R^(d×d)          （如 4096×4096 = 1677万参数）
  权重变化: ΔW = B × A             （低秩分解）
           B ∈ R^(d×r), A ∈ R^(r×d)  （r=16时: 4096×16 + 16×4096 = 13万参数）
  
  微调后: W = W₀ + ΔW = W₀ + B × A

  参数量对比：
    全参数微调: 更新 4096×4096 = 1677万参数
    LoRA(r=16): 更新 4096×16×2 = 13万参数
    → 减少99.2%的可训练参数！
```

**图示理解：**

```
全参数微调：
  输入x → [W₀ + ΔW](d×d) → 输出
  ΔW有d×d个参数需要训练 ← 太多了

LoRA：
  输入x → [W₀](冻结，不更新) → 输出₁
          ↘ [A](d×r) → [B](r×d) → 输出₂   ← 只训练A和B
  
  最终输出 = 输出₁ + 输出₂

  r（秩）是超参数：
    r越大 → 表达能力越强，但参数越多
    r越小 → 参数越少，但可能欠拟合
    通常 r=8 或 r=16 或 r=32
```

### 3.2 LoRA应用在Transformer的哪里？

```
Transformer中有很多权重矩阵，LoRA可以选择性地应用：

Attention层：
  W_q（Query投影）   ← 通常加LoRA ✅
  W_k（Key投影）     ← 通常加LoRA ✅
  W_v（Value投影）   ← 通常加LoRA ✅
  W_o（输出投影）    ← 可选

FFN层：
  W_gate（门控）     ← 可选
  W_up（上投影）     ← 可选
  W_down（下投影）   ← 可选

LLaMA-Factory默认配置：对所有线性层都加LoRA
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### 3.3 QLoRA（量化LoRA）

```
QLoRA = 先把基座模型量化到4-bit，再在上面加LoRA

原理：
  1. 将W₀量化为4-bit（NF4格式）→ 显存从14GB降到4GB
  2. 在量化后的模型上添加LoRA适配器（FP16精度）
  3. 训练时：前向传播用4-bit，反向传播用FP16
  
  显存需求：
    LoRA(FP16):  ~16GB（7B模型）→ 需要一张A100或两张3090
    QLoRA(4-bit): ~6GB（7B模型）→ 一张4060 8G就够！

  精度损失：相比纯LoRA，QLoRA损失极小（<1%），性价比极高
```

### 3.4 LoRA的关键超参数

```
rank (r): LoRA的秩
  - r=8: 轻量级，适合简单任务
  - r=16: 常用默认值
  - r=32-64: 复杂任务/大数据量
  - r=128+: 接近全参数微调效果

alpha (α): 缩放系数
  - 实际缩放: α/r
  - 通常设 α = 2r（如r=16, α=32）
  - α越大，LoRA的影响越大

dropout: LoRA层的dropout
  - 通常0.05-0.1
  - 防止过拟合

target_modules: 在哪些层加LoRA
  - 最少: ["q_proj", "v_proj"]
  - 推荐: 所有线性层
```

---

## 四、LLaMA-Factory实战（手把手教程）

### 4.1 安装

```bash
# 克隆仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装依赖
pip install -e ".[torch,metrics]"

# 验证安装
llamafactory-cli version
```

### 4.2 准备数据（以你的医疗项目为例）

**方法：从你的知识图谱自动生成SFT数据**

```python
"""
从Neo4j知识图谱自动生成SFT训练数据
保存为ShareGPT格式
"""
import json
from py2neo import Graph

graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# 查询所有疾病及其相关信息
query = """
MATCH (d:Disease)-[r]->(t)
RETURN d.name as disease, type(r) as relation, t.name as target, t.description as desc
ORDER BY d.name
"""
results = graph.run(query).data()

# 按疾病组织数据
diseases = {}
for r in results:
    name = r['disease']
    if name not in diseases:
        diseases[name] = {'symptoms': [], 'tests': [], 'treatments': [], 'risk_factors': []}
    if r['relation'] == 'HAS_SYMPTOM':
        diseases[name]['symptoms'].append(r['target'])
    elif r['relation'] == 'DIAGNOSED_BY':
        diseases[name]['tests'].append({'name': r['target'], 'desc': r['desc']})
    elif r['relation'] == 'TREATED_WITH':
        diseases[name]['treatments'].append(r['target'])
    elif r['relation'] == 'HAS_RISK_FACTOR':
        diseases[name]['risk_factors'].append(r['target'])

# 生成SFT数据
sft_data = []

for disease, info in diseases.items():
    # 类型1：疾病知识问答
    sft_data.append({
        "conversations": [
            {"from": "system", "value": "你是一名专业的急诊医学顾问，根据医学知识回答问题。"},
            {"from": "human", "value": f"请介绍{disease}的主要症状和风险因素。"},
            {"from": "gpt", "value": f"{disease}的主要症状包括：{'、'.join(info['symptoms'])}。\n\n主要风险因素包括：{'、'.join(info['risk_factors'])}。\n\n建议的检查方法：{'、'.join([t['name'] for t in info['tests']])}。"}
        ]
    })
    
    # 类型2：症状分析
    if info['symptoms']:
        symptoms_text = '、'.join(info['symptoms'][:3])
        sft_data.append({
            "conversations": [
                {"from": "system", "value": "你是一名专业的急诊医学顾问。"},
                {"from": "human", "value": f"患者出现{symptoms_text}，可能是什么疾病？"},
                {"from": "gpt", "value": f"根据症状分析，需要考虑{disease}的可能性。\n\n该疾病的典型症状包括：{'、'.join(info['symptoms'])}。\n\n建议进行以下检查以明确诊断：{'、'.join([t['name'] for t in info['tests']])}。"}
            ]
        })

# 保存为JSON
with open("medical_sft_data.json", "w", encoding="utf-8") as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=2)

print(f"生成了 {len(sft_data)} 条SFT训练数据")
```

### 4.3 注册数据集

在 `LLaMA-Factory/data/dataset_info.json` 中添加：

```json
{
  "medical_sft": {
    "file_name": "medical_sft_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

把你的 `medical_sft_data.json` 放到 `LLaMA-Factory/data/` 目录下。

### 4.4 训练配置文件

创建 `train_medical.yaml`：

```yaml
### 模型配置
model_name_or_path: Qwen/Qwen2.5-7B-Instruct  # 基座模型

### 微调方法
stage: sft                    # 有监督微调
do_train: true
finetuning_type: lora         # 使用LoRA

### LoRA配置
lora_rank: 16                 # 秩
lora_alpha: 32                # 缩放系数（通常=2×rank）
lora_dropout: 0.05
lora_target: all              # 对所有线性层加LoRA

### 数据配置
dataset: medical_sft          # 数据集名称（对应dataset_info.json中的key）
template: qwen                # 对话模板（Qwen模型用qwen）
cutoff_len: 2048              # 最大序列长度
preprocessing_num_workers: 4

### 训练超参数
per_device_train_batch_size: 2     # 每GPU的batch size
gradient_accumulation_steps: 8     # 梯度累积（等效batch_size=16）
learning_rate: 2.0e-4              # 学习率
num_train_epochs: 3                # 训练轮数
lr_scheduler_type: cosine          # 学习率调度器
warmup_ratio: 0.1                  # 预热比例

### 量化配置（QLoRA，如果显存不够就开启）
# quantization_bit: 4             # 4-bit量化（取消注释启用QLoRA）

### 输出配置
output_dir: output/medical_lora    # 模型输出目录
logging_steps: 10
save_steps: 100
save_total_limit: 3

### 其他
bf16: true                         # 使用BF16精度（A100/4090支持）
# fp16: true                       # 如果GPU不支持BF16，用FP16
```

### 4.5 开始训练

```bash
# 命令行训练
llamafactory-cli train train_medical.yaml

# 或者用Web UI（更直观）
llamafactory-cli webui
```

### 4.6 训练完成后

```bash
# 合并LoRA权重到基座模型（部署用）
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path output/medical_lora \
    --template qwen \
    --finetuning_type lora \
    --export_dir output/medical_merged \
    --export_size 2

# 合并后的模型可以直接用vLLM部署
python -m vllm.entrypoints.openai.api_server \
    --model output/medical_merged \
    --host 0.0.0.0 --port 8000

# 或者不合并，直接用LoRA适配器（vLLM也支持）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules medical=output/medical_lora
```

---

## 五、其他微调方法简介

| 方法 | 原理 | 可训练参数比例 | 适用场景 |
|------|------|--------------|----------|
| **Full Fine-Tuning** | 更新所有参数 | 100% | 数据充足、算力充足 |
| **LoRA** | 低秩矩阵分解 | ~0.1-1% | 最常用，平衡效果和成本 |
| **QLoRA** | 量化+LoRA | ~0.1-1% | 显存受限场景 |
| **Adapter** | 在层间插入小模块 | ~1-5% | 早期方法，现在用得少 |
| **Prefix-Tuning** | 在输入前加可学习前缀 | ~0.1% | 参数极少，但效果一般 |
| **P-Tuning v2** | 每层加可学习前缀 | ~0.1-1% | 清华提出，中文场景用过 |

**面试只需要重点掌握 LoRA 和 QLoRA。**

---

## 六、DPO简介（了解即可）

```
DPO = Direct Preference Optimization = 直接偏好优化

位于训练第三阶段（对齐），在SFT之后进行

原理：
  数据格式：(prompt, chosen_response, rejected_response)
  即：给同一个问题的两个回答，标注哪个更好

  训练目标：让模型更倾向于生成chosen_response
  
  相比RLHF的优势：
    RLHF需要训练一个Reward Model + PPO强化学习 → 复杂、不稳定
    DPO直接用偏好数据优化模型 → 简单、稳定

LLaMA-Factory中使用DPO：
  stage: dpo    # 改为dpo即可
  数据格式要求：每条数据需要有chosen和rejected两个回答
```

---

## 七、面试高频问题速答

**Q1：LoRA的原理？**
> LoRA假设微调时权重变化量是低秩的，将权重更新分解为两个小矩阵的乘积 ΔW=BA。训练时冻结原始权重W₀，只训练B和A。以rank=16为例，可训练参数量仅为全参数微调的0.1%，但效果接近。

**Q2：LoRA的rank怎么选？**
> 取决于任务复杂度和数据量。简单任务（格式调整）r=8即可；通用微调r=16是默认值；复杂任务（领域适配）r=32-64。rank越大表达能力越强但越容易过拟合，通常从16开始尝试。

**Q3：QLoRA和LoRA的区别？**
> QLoRA在LoRA基础上将基座模型量化为4-bit（NF4格式），显存需求降低60-70%。7B模型LoRA需要~16GB显存，QLoRA只需~6GB。精度损失<1%，是显存受限时的最佳选择。

**Q4：SFT数据需要多少条？**
> 取决于任务：格式微调（让模型学习输出格式）几百条即可；领域微调（医疗/法律等）通常需要1K-10K条高质量数据；通用能力提升需要50K+条。数据质量比数量更重要——1000条高质量数据通常优于10000条噪声数据（参考LIMA论文）。

**Q5：SFT会导致灾难性遗忘吗？**
> 会。过度微调某个领域可能导致模型在其他领域的能力下降。缓解方法：1）控制训练epoch（通常1-3个epoch）；2）使用LoRA（只更新少量参数，对原始能力影响小）；3）在训练数据中混入通用数据。

**Q6：你做过微调吗？讲讲你的经验。**
> （做完实战后这样回答）"我用LLaMA-Factory对Qwen2.5-7B-Instruct做了LoRA微调，数据集是从我构建的Neo4j医疗知识图谱中自动生成的X条问答对，使用ShareGPT格式。训练配置是rank=16、alpha=32、learning_rate=2e-4，在AutoDL的A100上训练了3个epoch约X小时。微调后模型在医疗问答任务上的回答更加结构化和准确。"

---

*建议：先通读这篇文章理解概念 → 在AutoDL上用LLaMA-Factory跑通一个最简单的微调（用官方示例数据）→ 再用你自己的医疗数据做微调 → 最后把经验写进简历*
