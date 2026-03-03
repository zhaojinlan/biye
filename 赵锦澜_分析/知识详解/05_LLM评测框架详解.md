# LLM评测框架详解

> 面试官问"你怎么证明你的系统有效？"时，如果你答不上来，所有技术实现都打折扣。这篇帮你建立评测意识和方法论。

---

## 一、为什么需要评测？

```
场景：你做了一个医疗RAG系统，老板问你"效果怎么样？"

❌ 错误回答："我试了几个问题，感觉还不错"
✅ 正确回答："在100条医疗问答测试集上，Faithfulness=0.87，Answer Relevancy=0.92，
             Context Precision=0.85，相比纯向量检索提升了15%"

区别：
  前者 = 主观感受，不可复现
  后者 = 量化指标，可对比可复现
```

---

## 二、评测的三个层面

```
层面1：模型能力评测
  评什么：LLM本身的通用能力（推理、知识、代码...）
  场景：选型时对比不同模型
  工具：OpenCompass、lm-evaluation-harness

层面2：RAG系统评测  ← 你最需要的
  评什么：检索质量、生成质量、端到端效果
  场景：优化RAG pipeline
  工具：RAGAS、TruLens、DeepEval

层面3：Agent评测
  评什么：Agent的任务完成率、工具调用准确率、推理质量
  场景：优化Agent工作流
  工具：AgentBench、ToolBench、自定义评测
```

---

## 三、RAG评测框架详解

### 3.1 RAGAS（最主流的RAG评测框架）⭐⭐⭐⭐⭐

**核心指标（必须掌握）：**

```
RAGAS定义了4个核心指标：

1. Faithfulness（忠实度）⭐⭐⭐⭐⭐
   问题：生成的回答是否忠于检索到的上下文？（有没有幻觉）
   计算：将回答拆成多个陈述 → 逐个检查是否能从context中推导出来
   范围：0-1，越高越好
   
   示例：
     Context: "坏死性软组织感染的LRINEC评分≥6分为高危"
     Answer: "LRINEC评分≥8分为高危"  ← Faithfulness低（数字不对，幻觉）
     Answer: "LRINEC评分≥6分为高危"  ← Faithfulness高（忠于原文）

2. Answer Relevancy（回答相关性）⭐⭐⭐⭐⭐
   问题：回答和问题的相关程度？
   计算：从回答反向生成问题 → 和原始问题计算相似度
   范围：0-1，越高越好
   
   示例：
     Question: "坏死性软组织感染怎么治疗？"
     Answer: "治疗包括手术清创和抗生素治疗"  ← 高相关性
     Answer: "坏死性软组织感染是一种严重疾病"  ← 低相关性（没回答治疗问题）

3. Context Precision（上下文精确度）⭐⭐⭐⭐
   问题：检索到的上下文中，有多少是真正有用的？
   计算：评估排名靠前的chunk是否包含答案相关信息
   范围：0-1，越高越好
   
   示例：
     检索到5个chunk：
       chunk1: 关于坏死性软组织感染治疗（相关✅）
       chunk2: 关于坏死性软组织感染病因（部分相关）
       chunk3: 关于高血压治疗（不相关❌）
       chunk4: 关于坏死性软组织感染手术（相关✅）
       chunk5: 关于感冒症状（不相关❌）
     → Context Precision ≈ 0.6（5个中2-3个相关）

4. Context Recall（上下文召回率）⭐⭐⭐⭐
   问题：所有应该被检索到的信息，实际检索到了多少？
   计算：将ground truth回答拆成陈述 → 检查每个陈述是否在context中
   范围：0-1，越高越好（需要ground truth）
   
   示例：
     Ground Truth包含5个关键信息点
     检索到的context覆盖了4个
     → Context Recall = 0.8
```

**RAGAS使用代码：**

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# 准备评测数据
eval_data = {
    "question": [
        "坏死性软组织感染的主要症状是什么？",
        "LRINEC评分怎么计算？",
    ],
    "answer": [
        "主要症状包括局部皮肤红肿、剧烈疼痛、高热、捻发音等",
        "LRINEC评分包括CRP、白细胞计数、血红蛋白、血钠、肌酐、血糖6项指标，总分0-13分",
    ],
    "contexts": [
        ["坏死性软组织感染的临床表现：局部皮肤红肿热痛，疼痛程度超出表面病变范围，可伴有高热、捻发音..."],
        ["LRINEC评分系统：C反应蛋白(CRP)、白细胞计数(WBC)、血红蛋白(Hb)、血钠、肌酐、血糖，总分0-13分..."],
    ],
    "ground_truth": [
        "坏死性软组织感染的主要症状包括局部皮肤红肿、剧烈疼痛超出病变范围、高热、捻发音、皮下积气等",
        "LRINEC评分由6项实验室指标组成：CRP、WBC、血红蛋白、血钠、肌酐、血糖，总分0-13分，≥6分提示高度怀疑",
    ],
}

dataset = Dataset.from_dict(eval_data)

# 运行评测
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)

print(result)
# 输出类似：
# {'faithfulness': 0.87, 'answer_relevancy': 0.92, 
#  'context_precision': 0.85, 'context_recall': 0.80}
```

### 3.2 TruLens ⭐⭐⭐

```
TruLens的特点：
  - 用LLM评测LLM（LLM-as-Judge）
  - 提供Web Dashboard可视化评测结果
  - 支持trace每次RAG调用的全过程

核心指标：
  - Groundedness（接地性）：回答是否基于检索到的证据
  - Answer Relevance（回答相关性）：同RAGAS
  - Context Relevance（上下文相关性）：检索结果和问题的相关度

vs RAGAS：
  RAGAS：更学术、指标更严格、社区更大
  TruLens：更工程化、可视化更好、适合生产环境监控
```

### 3.3 DeepEval ⭐⭐⭐

```
DeepEval的特点：
  - 类似pytest的评测框架（和你要学的pytest结合很好）
  - 支持CI/CD集成
  - 内置14+评测指标

使用示例：
```

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

test_case = LLMTestCase(
    input="坏死性软组织感染怎么治疗？",
    actual_output="治疗包括手术清创和广谱抗生素治疗",
    retrieval_context=["坏死性软组织感染的治疗以手术清创为核心，辅以广谱抗生素..."],
)

faithfulness = FaithfulnessMetric(threshold=0.7)
relevancy = AnswerRelevancyMetric(threshold=0.7)

assert_test(test_case, [faithfulness, relevancy])
```

---

## 四、模型能力评测框架

### 4.1 OpenCompass（上海AI Lab，国内最主流）⭐⭐⭐⭐

```
特点：
  - 中文评测最全面
  - 支持50+评测数据集
  - 支持主流LLM（Qwen/LLaMA/ChatGLM...）
  - 提供排行榜

评测维度：
  - 语言能力：MMLU、C-Eval、CMMLU
  - 推理能力：GSM8K（数学）、BBH（逻辑）
  - 代码能力：HumanEval、MBPP
  - 知识能力：ARC、TriviaQA
  - 安全性：CValues、SafetyBench

使用场景：
  - 对比不同模型的综合能力
  - 对比微调前后模型的能力变化
  - 评估量化后模型的精度损失
```

### 4.2 lm-evaluation-harness（EleutherAI，国际标准）⭐⭐⭐

```
特点：
  - 学术界最广泛使用
  - 200+评测任务
  - HuggingFace Open LLM Leaderboard使用的评测框架

使用：
  pip install lm-eval
  lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-7B-Instruct --tasks mmlu --batch_size 8
```

### 4.3 常见评测基准（Benchmark）

| 基准 | 评测内容 | 说明 |
|------|----------|------|
| **MMLU** | 57个学科的多选题 | 最常用的通用能力评测 |
| **C-Eval** | 中文多学科评测 | 中文版MMLU |
| **GSM8K** | 小学数学应用题 | 评测数学推理能力 |
| **HumanEval** | Python编程题 | 评测代码生成能力 |
| **MT-Bench** | 多轮对话评测 | GPT-4打分的对话质量评测 |
| **AlpacaEval** | 指令遵循评测 | 和GPT-4对比的胜率 |
| **IFEval** | 指令遵循精确度 | 如"用3个段落回答"是否遵循 |

---

## 五、LLM-as-Judge（用LLM评测LLM）

### 5.1 原理

```
核心思想：用一个强大的LLM（如GPT-4）来评价其他LLM的输出

为什么可行？
  - 人工评测太贵太慢
  - 传统指标（BLEU/ROUGE）和人类判断相关性差
  - GPT-4等强模型的评判和人类高度一致（>80%一致率）

评测方式：
  方式1：直接打分
    "请给以下回答打1-5分：[回答]"
    
  方式2：成对比较
    "以下两个回答，哪个更好？
     回答A：[...]
     回答B：[...]"
    
  方式3：参考评测
    "参考标准答案：[ground truth]
     评估以下回答的准确性：[回答]"
```

### 5.2 实现示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)

judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名医学评测专家。请根据以下标准评估回答质量：
    
    1. 准确性（1-5分）：回答内容是否医学上准确
    2. 完整性（1-5分）：是否覆盖了问题的主要方面
    3. 相关性（1-5分）：回答是否紧扣问题
    4. 安全性（1-5分）：是否有可能误导患者的信息
    
    请以JSON格式输出评分和理由。"""),
    ("user", """
    问题：{question}
    参考答案：{reference}
    待评估回答：{answer}
    """)
])

chain = judge_prompt | judge_llm

result = chain.invoke({
    "question": "坏死性软组织感染的主要症状是什么？",
    "reference": "主要症状包括：局部皮肤红肿、疼痛超出病变范围、高热、捻发音、皮下积气、进展迅速等",
    "answer": "坏死性软组织感染的症状有皮肤红肿和发热"  # 不完整的回答
})

print(result.content)
# 输出类似：
# {"准确性": 4, "完整性": 2, "相关性": 4, "安全性": 4,
#  "理由": "回答虽然准确但不完整，遗漏了疼痛超出病变范围、捻发音等重要症状"}
```

### 5.3 注意事项

```
LLM-as-Judge的陷阱：
  1. 位置偏见：成对比较时偏向第一个回答 → 解决：随机调换顺序
  2. 冗长偏见：倾向给更长的回答更高分 → 解决：在prompt中强调简洁性
  3. 自我偏见：GPT-4倾向给GPT-4的回答更高分 → 解决：用不同模型做judge
  4. 评分不稳定：多次评测结果不一致 → 解决：多次评测取平均，temperature=0
```

---

## 六、Agent评测

### 6.1 Agent评测的挑战

```
Agent评测比LLM评测难得多：
  - 中间步骤不确定（同一任务可能有多种正确的执行路径）
  - 工具调用有副作用（调用了不该调用的API）
  - 多轮交互难以自动化评测
  - 缺乏标准化的评测数据集
```

### 6.2 Agent评测指标

```
1. 任务完成率（Task Success Rate）
   最终是否完成了目标任务
   你的系统：是否给出了正确的诊断和合理的检查推荐

2. 工具调用准确率（Tool Call Accuracy）
   是否调用了正确的工具、传入了正确的参数
   你的系统：是否正确调用了analyze_disease_probability和get_diagnostic_tests

3. 推理步骤质量（Reasoning Quality）
   中间推理过程是否合理
   你的系统：疾病评分计算是否正确、风险因素匹配是否准确

4. 效率（Efficiency）
   完成任务用了多少步骤、调用了多少次工具
   你的系统：是否有不必要的重复工具调用

5. 安全性（Safety）
   是否有可能造成危害的输出
   你的系统：是否有误诊风险、是否给出了不合理的医疗建议
```

### 6.3 适合你的评测方案

```python
"""
医疗Agent评测脚本示例
评测你的多智能体系统的诊断准确率
"""
import json

# 测试用例（人工构造）
test_cases = [
    {
        "patient_info": "65岁男性，糖尿病史，左下肢红肿3天，疼痛剧烈，体温39.5°C",
        "expected_disease": "坏死性软组织感染",
        "expected_tests": ["LRINEC评分", "血常规", "CRP"],
        "expected_triage_level": "一级"  # 紧急
    },
    {
        "patient_info": "30岁女性，反复喘息，有哮喘家族史，近日接触猫毛后加重",
        "expected_disease": "支气管哮喘急性发作",
        "expected_tests": ["肺功能检查", "血气分析"],
        "expected_triage_level": "二级"
    },
    # ... 更多测试用例
]

# 评测函数
def evaluate_diagnosis(agent_output, expected):
    scores = {}
    
    # 1. 疾病诊断准确率
    predicted_disease = extract_disease(agent_output)
    scores["disease_accuracy"] = 1.0 if predicted_disease == expected["expected_disease"] else 0.0
    
    # 2. 检查推荐覆盖率
    predicted_tests = extract_tests(agent_output)
    expected_tests = set(expected["expected_tests"])
    covered = len(set(predicted_tests) & expected_tests)
    scores["test_recall"] = covered / len(expected_tests) if expected_tests else 0
    
    # 3. 分诊等级准确率
    predicted_triage = extract_triage_level(agent_output)
    scores["triage_accuracy"] = 1.0 if predicted_triage == expected["expected_triage_level"] else 0.0
    
    return scores

# 运行评测
results = []
for case in test_cases:
    output = run_agent(case["patient_info"])  # 调用你的Agent
    scores = evaluate_diagnosis(output, case)
    results.append(scores)

# 汇总结果
avg_disease_acc = sum(r["disease_accuracy"] for r in results) / len(results)
avg_test_recall = sum(r["test_recall"] for r in results) / len(results)
avg_triage_acc = sum(r["triage_accuracy"] for r in results) / len(results)

print(f"疾病诊断准确率: {avg_disease_acc:.2%}")
print(f"检查推荐召回率: {avg_test_recall:.2%}")
print(f"分诊等级准确率: {avg_triage_acc:.2%}")
```

---

## 七、传统NLP指标（了解即可）

| 指标 | 评测内容 | 计算方式 | 局限性 |
|------|----------|----------|--------|
| **BLEU** | 机器翻译质量 | n-gram精确度 | 只看词汇重叠，不理解语义 |
| **ROUGE** | 摘要质量 | n-gram召回率 | 同上 |
| **BERTScore** | 语义相似度 | 用BERT计算token级语义相似度 | 比BLEU/ROUGE好，但仍不完美 |
| **Perplexity** | 语言模型质量 | 模型对测试集的困惑度 | 困惑度低不代表回答好 |

```
面试如果问你"怎么评测你的系统"：
  ❌ 不要说BLEU/ROUGE（过时了，和LLM应用不太匹配）
  ✅ 要说RAGAS指标 + LLM-as-Judge + 自定义评测集 + 人工抽检
```

---

## 八、面试高频问题速答

**Q1：你怎么评测你的RAG系统效果？**
> 三个层面：1）离线评测：用RAGAS框架评测Faithfulness、Answer Relevancy、Context Precision等指标；2）自动化评测：用LLM-as-Judge对回答质量打分；3）人工评测：医学专家抽样检查诊断建议的准确性。目前我的项目还缺乏系统化评测，这是需要改进的方向。

**Q2：RAGAS的核心指标有哪些？**
> 四个核心指标：Faithfulness（回答是否忠于检索内容，检测幻觉）、Answer Relevancy（回答与问题的相关度）、Context Precision（检索结果中有用内容的比例）、Context Recall（应检索到的信息实际检索到的比例）。

**Q3：LLM-as-Judge有什么问题？**
> 三个主要偏见：1）位置偏见（成对比较时偏向排在前面的回答）；2）冗长偏见（倾向给更长的回答更高分）；3）自我偏见（GPT-4倾向给自己的输出更高分）。缓解方法：随机调换顺序、多次评测取平均、使用不同模型做judge。

**Q4：怎么评测Agent的效果？**
> 从5个维度评测：任务完成率（是否正确完成目标）、工具调用准确率（是否调用正确工具和参数）、推理步骤质量（中间过程是否合理）、效率（步骤数和延迟）、安全性（是否有危害性输出）。具体到我的医疗系统，我会构造标准化测试用例，评测疾病诊断准确率、检查推荐覆盖率和分诊等级准确率。

---

*最实用的行动：用RAGAS评测你的医疗RAG系统，构造20-50条测试数据，跑一次评测，把结果数据写进简历。*
