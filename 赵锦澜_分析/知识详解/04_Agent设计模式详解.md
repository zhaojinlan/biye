# Agent设计模式详解

> 这是你的优势领域。你的医疗项目已经用了Supervisor和ReAct模式，但面试官会问你"还有哪些模式？各自适用什么场景？"这篇帮你建立完整的认知框架。

---

## 一、什么是AI Agent？

```
Agent = LLM + 工具调用 + 记忆 + 规划

传统LLM：输入问题 → 输出回答（一次性，无法行动）
Agent：输入目标 → 思考 → 调用工具 → 观察结果 → 继续思考 → ... → 完成目标

核心区别：Agent能**自主决策**要调用什么工具、执行什么步骤
```

---

## 二、单Agent设计模式

### 2.1 ReAct（Reasoning + Acting）⭐⭐⭐⭐⭐

**你已经在用的模式。**

```
核心思想：交替进行"推理"和"行动"

循环流程：
  Thought（思考）→ Action（行动）→ Observation（观察）→ Thought → Action → ...

示例（你的医疗系统中的recommend_node）：

  Thought: 需要分析疾病概率，我有各疾病的得分数据
  Action: 调用 analyze_disease_probability({"支气管哮喘": 1, "坏死性软组织感染": 2}, 7)
  Observation: 最可能疾病=坏死性软组织感染，置信度=73.1%
  
  Thought: 需要获取这个疾病的推荐检查
  Action: 调用 get_diagnostic_tests_for_disease("坏死性软组织感染")
  Observation: 返回LRINEC评分、血常规、CRP等检查方法
  
  Thought: 已获得所有信息，可以生成最终诊断建议
  Action: 输出最终结论

优点：
  - 实现简单（LangChain的create_react_agent一行搞定）
  - 推理过程可解释
  - 适合工具调用场景

缺点：
  - 每步只能做一个动作（串行）
  - 容易陷入循环（反复调用同一工具）
  - 对复杂任务缺乏整体规划
```

**LangGraph实现：**
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=[tool1, tool2, tool3],
    prompt="你是一名医学分析师..."
)

result = await agent.ainvoke({"messages": [HumanMessage(content="...")]})
```

### 2.2 Plan-and-Execute（规划-执行）⭐⭐⭐⭐

```
核心思想：先制定完整计划，再逐步执行

流程：
  步骤1：Planner（规划器）根据目标生成完整的执行计划
  步骤2：Executor（执行器）逐步执行每个子任务
  步骤3：执行完一步后，可以重新规划（Replan）

示例：
  用户目标："帮我分析患者的病情并推荐检查"
  
  Planner生成计划：
    1. 分析患者症状，提取关键信息
    2. 查询知识图谱匹配可能的疾病
    3. 计算各疾病的概率
    4. 获取最可能疾病的推荐检查
    5. 生成综合诊断报告
  
  Executor逐步执行：
    执行步骤1 → 结果：提取到发热、皮肤红肿、疼痛
    执行步骤2 → 结果：匹配到3种可能疾病
    ...
    
  如果执行中发现计划需要调整（如步骤2没找到匹配疾病）：
    Replanner重新规划 → 新计划：扩大搜索范围或使用向量检索

vs ReAct的区别：
  ReAct：边想边做（像散步，走一步看一步）
  Plan-and-Execute：先想好再做（像导航，先规划路线再出发）

适用场景：
  - 复杂多步骤任务
  - 需要全局规划的场景
  - 任务步骤之间有依赖关系
```

**LangGraph实现框架：**
```python
from langgraph.graph import StateGraph, END

class PlanExecuteState(TypedDict):
    input: str
    plan: list[str]
    current_step: int
    results: list[str]
    response: str

def planner(state):
    """生成执行计划"""
    plan = llm.invoke(f"为以下目标制定步骤计划：{state['input']}")
    return {"plan": parse_plan(plan)}

def executor(state):
    """执行当前步骤"""
    step = state["plan"][state["current_step"]]
    result = react_agent.invoke(step)  # 用ReAct执行单步
    return {"results": state["results"] + [result], "current_step": state["current_step"] + 1}

def replanner(state):
    """检查是否需要重新规划"""
    if needs_replan(state):
        new_plan = llm.invoke(f"根据当前结果重新规划：{state['results']}")
        return {"plan": parse_plan(new_plan)}
    return state

def should_continue(state):
    if state["current_step"] >= len(state["plan"]):
        return "finish"
    return "execute"

graph = StateGraph(PlanExecuteState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("replanner", replanner)
graph.add_edge("planner", "executor")
graph.add_edge("executor", "replanner")
graph.add_conditional_edges("replanner", should_continue, {"execute": "executor", "finish": END})
graph.set_entry_point("planner")
```

### 2.3 Reflection（反思）⭐⭐⭐⭐

```
核心思想：Agent生成输出后，自我审视并改进

流程：
  步骤1：Generator 生成初始输出
  步骤2：Critic 评估输出质量，指出问题
  步骤3：Generator 根据反馈改进输出
  步骤4：重复2-3直到质量满意

示例（用在你的医疗系统中）：
  
  Generator生成诊断建议：
    "患者可能是坏死性软组织感染，建议做血常规"
  
  Critic反思：
    "问题1：诊断建议过于简单，缺少依据
     问题2：只推荐了一项检查，应该推荐LRINEC评分等多项
     问题3：缺少紧急程度评估"
  
  Generator改进：
    "基于以下分析，患者高度怀疑坏死性软组织感染：
     1. 风险因素匹配：糖尿病(+1)、高龄(+1)、皮肤损伤(+1)
     2. 置信度：73.1%
     推荐检查（按紧急程度排序）：
     - LRINEC评分（首要，6分以上高度怀疑）
     - 血常规+CRP（评估感染程度）
     - CT扫描（评估组织受累范围）
     紧急程度：高，建议立即急诊外科会诊"

适用场景：
  - 需要高质量输出的场景（代码生成、报告撰写）
  - 有明确质量标准的场景
  - 一次性生成质量不够的场景
```

### 2.4 Tool Use / Function Calling ⭐⭐⭐⭐⭐

```
核心思想：LLM决定何时调用什么工具

这是最基础的Agent能力，几乎所有Agent模式都依赖它。

你的项目中的体现：
  - MCP工具：get_diagnostic_tests_for_disease, search_medical_knowledge
  - 本地工具：analyze_disease_probability

实现方式：
  方式1：OpenAI Function Calling（模型原生支持）
  方式2：ReAct Prompting（通过Prompt让模型输出工具调用格式）
  方式3：MCP协议（你用的方式，标准化工具接口）

Tool Use的关键设计：
  1. 工具描述要清晰（模型通过描述决定是否调用）
  2. 参数定义要精确（用JSON Schema）
  3. 返回值要结构化（方便模型理解和使用）
```

---

## 三、多Agent设计模式

### 3.1 Supervisor模式（你的项目正在使用）⭐⭐⭐⭐⭐

```
核心思想：一个"监督者"Agent负责协调多个"工作者"Agent

架构：
                    ┌──────────┐
                    │Supervisor│ ← 决定调度哪个Worker
                    │(路由器)   │
                    └────┬─────┘
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌────────┐
         │Worker A│ │Worker B│ │Worker C│
         │(分诊)  │ │(查询)  │ │(诊断)  │
         └────────┘ └────────┘ └────────┘

你的医疗系统：
  Supervisor = supervisor_node（根据has_triaged/has_diagnosis等状态路由）
  Worker A = triage_node（分诊评估）
  Worker B = query_node（知识查询）
  Worker C = recommend_node（风险分析）
  Worker D = experts_node（专家会诊）

优点：
  - 架构清晰，易于理解和调试
  - 每个Worker专注于一个任务
  - 容易扩展（添加新Worker）

缺点：
  - Supervisor是单点，如果路由错误整个流程出错
  - Worker之间不能直接通信（必须通过Supervisor中转）
  - 不适合需要Worker自由协商的场景
```

### 3.2 Hierarchical（层级式）⭐⭐⭐

```
核心思想：多层Supervisor，树状结构

架构：
                    ┌───────────────┐
                    │ Top Supervisor│
                    └───────┬───────┘
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
          ┌──────────┐  ┌──────┐  ┌──────────┐
          │Sub-Sup A │  │Agent │  │Sub-Sup B │
          └────┬─────┘  └──────┘  └────┬─────┘
           ┌───┼───┐                ┌───┼───┐
           ▼   ▼   ▼               ▼   ▼   ▼
          A1  A2  A3              B1  B2  B3

适用场景：
  - 大型复杂系统（10+个Agent）
  - 不同领域的Agent需要分组管理
  - 例如：医疗系统中，分诊组（内科/外科/急诊分诊）和诊断组（影像/检验/病理诊断）

你的项目中的体现：
  experts_node内部就是一个小型层级结构：
    experts_node（Sub-Supervisor）
      ├── 医学专家A
      ├── 医学专家B
      └── 医学专家C
```

### 3.3 Debate（辩论式）⭐⭐⭐

```
核心思想：多个Agent对同一问题各自发表观点，通过辩论达成更好的结论

架构：
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ Agent A  │────→│ Agent B  │────→│ Agent C  │
  │ (正方)   │←────│ (反方)   │←────│ (裁判)   │
  └──────────┘     └──────────┘     └──────────┘

流程：
  Round 1: Agent A提出观点 → Agent B反驳 → Agent C记录
  Round 2: Agent A回应反驳 → Agent B再反驳 → Agent C记录
  ...
  最终: Agent C综合所有观点给出裁决

你的项目中的类似设计：
  triage_node中的双Agent分诊：
    - 医学顾问Agent提出初步分诊
    - 急诊分诊Agent独立评估
    - 系统融合两方意见
  → 这不是严格的辩论，但思路类似

适用场景：
  - 需要多角度分析的决策
  - 减少单Agent的偏见和幻觉
  - 高风险场景（医疗、金融）需要交叉验证
```

### 3.4 Collaboration / Swarm（协作/蜂群）⭐⭐⭐

```
核心思想：Agent之间平等协作，没有中心控制者

架构：
  ┌──────────┐     ┌──────────┐
  │ Agent A  │←───→│ Agent B  │
  └────┬─────┘     └────┬─────┘
       │                │
       └───────┬────────┘
               ▼
         ┌──────────┐
         │ Agent C  │
         └──────────┘

特点：
  - 没有Supervisor，Agent自主决定何时与谁协作
  - 通过共享状态或消息传递通信
  - 类似OpenAI Swarm的设计

vs Supervisor模式：
  Supervisor：老板指挥员工干活
  Swarm：一群同事自己商量着干活

适用场景：
  - Agent能力相近，无明显主从关系
  - 需要灵活协作的创意类任务
  - 实验性场景
```

### 3.5 Map-Reduce（映射-归约）⭐⭐⭐

```
核心思想：将任务分解给多个Agent并行处理，再汇总结果

架构：
         输入任务
            │
    ┌───────┼───────┐        Map阶段
    ▼       ▼       ▼       （并行处理）
  Agent1  Agent2  Agent3
    │       │       │
    └───────┼───────┘        Reduce阶段
            ▼               （汇总结果）
        汇总Agent
            │
         最终结果

你的项目中的体现：
  experts_node 就是Map-Reduce模式！
    Map: 三个专家Agent并行处理同一患者数据
    Reduce: 最终的汇总逻辑合并三个专家的意见

适用场景：
  - 同一任务可以从多个角度并行分析
  - 处理大规模数据（每个Agent处理一部分）
  - 需要多个"专家意见"的场景
```

---

## 三点五、Agent设计模式的本质：模仿人类协作方式

你的直觉是对的——**每一种Agent架构都能在人类社会中找到原型**。这不是巧合，而是因为：

> AI Agent的目标是"像人一样完成任务"，而人类已经用了几千年时间优化出各种协作模式。
> 直接借鉴这些模式，比从零发明更高效。

### 完整对应表：Agent模式 ↔ 人类协作原型

```
┌──────────────────┬──────────────────────────────┬─────────────────────────┐
│  Agent模式        │  人类协作原型                  │  现实例子                │
├──────────────────┼──────────────────────────────┼─────────────────────────┤
│  ReAct           │  个人独立工作                  │  一个医生边思考边查资料   │
│  (边想边做)       │  （想一步做一步）              │  边做诊断                │
│                  │                              │                         │
│  Plan-and-Execute│  项目经理制定计划              │  手术前制定手术方案       │
│  (先规划再执行)   │  团队按计划执行                │  然后按步骤执行手术       │
│                  │                              │                         │
│  Reflection      │  写论文后自我审稿修改          │  医生写完病历自己检查     │
│  (自我反思)       │  老师批改作业后重做            │  发现遗漏再补充          │
│                  │                              │                         │
│  Supervisor      │  公司部门主管分配任务          │  急诊科主任分诊：         │
│  (主管调度)       │  员工各司其职汇报主管          │  "你去拍CT，你去抽血"     │
│                  │                              │                         │
│  Hierarchical    │  集团→子公司→部门→员工        │  医院→科室→医疗组→医生   │
│  (层级管理)       │  多层管理结构                  │  院长→科主任→主治→住院医  │
│                  │                              │                         │
│  Debate          │  学术辩论/法庭辩论             │  MDT多学科会诊：          │
│  (辩论对抗)       │  正方反方各自论证、裁判判决     │  外科说要手术，内科说保守  │
│                  │                              │  最终主任综合决策         │
│                  │                              │                         │
│  Map-Reduce      │  考试阅卷多人分工              │  体检中心：               │
│  (分工汇总)       │  每人批一部分，最后汇总成绩     │  抽血/心电图/B超并行做    │
│                  │                              │  最后汇总出体检报告       │
│                  │                              │                         │
│  Swarm/协作      │  开源社区/头脑风暴             │  几个医生一起讨论病例     │
│  (平等协作)       │  没有领导，自组织协作          │  谁有想法谁发言          │
│                  │                              │                         │
│  Pipeline        │  工厂流水线                    │  病人看病流程：           │
│  (流水线)         │  每个工位做一步，传给下一个     │  挂号→问诊→检查→取药     │
│                  │                              │                         │
│  Blackboard      │  白板讨论                     │  病房里的白板：           │
│  (黑板模式)       │  大家在同一个白板上写写画画     │  护士/医生都在上面更新    │
│                  │  看到别人写的内容触发自己的思考  │  患者状态，互相参考       │
└──────────────────┴──────────────────────────────┴─────────────────────────┘
```

### 为什么这个视角重要？

```
1. 帮你记忆：不用死记硬背架构图，想想对应的人类场景就记住了

2. 帮你选型：遇到新业务时，先想"人类会怎么协作完成这件事？"
   → 答案自然指向对应的Agent模式
   
   例如：
     "需要多个专家各自分析再汇总" → 你会想到"多学科会诊" → Map-Reduce
     "需要一步步审批流程" → 你会想到"公司审批链" → Pipeline
     "需要反复修改直到满意" → 你会想到"论文改稿" → Reflection

3. 帮你面试：面试官问"为什么选这个架构？"
   你可以说："医疗诊断的现实流程就是这样——先分诊再专科再会诊，
   我的Supervisor架构直接映射了这个临床流程。"
   → 这比纯技术回答更有说服力，说明你理解业务
```

### 你的医疗系统 ↔ 真实医院的完整对应

```
你的系统架构：                    真实医院流程：

用户描述症状                      患者到急诊描述症状
      ↓                                ↓
supervisor_node（路由）            分诊台护士（判断去哪个科）
      ↓                                ↓
triage_node（双Agent分诊）         急诊预检分诊（护士+医生初评）
      ↓                                ↓
query_node（知识查询）             医生查阅指南/文献
      ↓                                ↓
recommend_node（风险分析）         医生做风险评估+开检查单
      ↓                                ↓
experts_node（多专家会诊）         MDT多学科会诊
      ↓                                ↓
输出诊断建议                       给出诊疗方案

→ 你的架构不是凭空设计的，而是对真实临床流程的数字化映射！
→ 这也是为什么Supervisor模式适合你的场景——因为医院就是这么运作的
```

### 还有哪些你没用到但值得了解的模式？

除了上表列的，学术界和工业界还在探索的Agent协作模式：

```
1. Auction（拍卖模式）
   人类原型：招标会
   Agent互相"竞标"任务，谁最有信心处理就分配给谁
   适用：任务分配不明确，需要动态匹配能力

2. Voting（投票模式）
   人类原型：委员会投票表决
   多个Agent各自给出答案，少数服从多数
   适用：减少单个Agent的错误（集成学习的思想）

3. Teacher-Student（师徒模式）
   人类原型：导师带学生
   强Agent指导弱Agent，弱Agent逐步学会
   适用：知识蒸馏、渐进式训练

4. Adversarial（对抗模式）
   人类原型：红蓝军对抗演练
   一个Agent生成，另一个Agent找漏洞
   适用：安全测试、对抗样本生成

5. Market（市场模式）
   人类原型：自由市场经济
   Agent之间有"货币"，通过交易协商资源分配
   适用：多Agent资源调度
```

**核心规律：人类社会中的每种协作模式，都能变成一种Agent架构。AI没有发明新的协作方式，只是用代码实现了人类已有的组织智慧。**

---

## 四、Agent核心组件

### 4.1 Memory（记忆）

```
短期记忆（Working Memory）：
  - 当前对话的上下文
  - 你的项目用InMemorySaver实现
  - 每次对话结束后清除（或保留为历史）

长期记忆（Long-term Memory）：
  - 持久化存储，跨对话保留
  - 实现方式：向量数据库、数据库、文件
  - 例如：记住患者历史就诊记录

  代表项目：
    - Zep：专门的Agent记忆服务
    - MemGPT：将记忆管理视为操作系统的虚拟内存
    - LangGraph的Checkpointer：状态持久化

情景记忆（Episodic Memory）：
  - 记住过去的"经验"
  - 例如："上次遇到类似症状的患者，最终确诊是X"
  - 实现方式：将历史Agent轨迹存入向量数据库，相似场景时检索
```

### 4.2 Planning（规划）

```
任务分解（Task Decomposition）：
  - 将复杂任务拆分为子任务
  - Chain of Thought（CoT）：逐步推理
  - Tree of Thoughts（ToT）：生成多条推理路径，选最优的

自我反思（Self-Reflection）：
  - Agent评估自己的输出质量
  - Reflexion框架：失败后反思原因，改进策略

工具选择（Tool Selection）：
  - 根据任务需求选择合适的工具
  - 如果没有合适的工具，可以请求人类帮助
```

### 4.3 Tool Use（工具使用）

```
工具类型：
  - API调用（搜索、数据库查询）
  - 代码执行（Python解释器）
  - 文件操作（读写文件）
  - 浏览器操作（Browser Use）
  - MCP工具（标准化协议，你的项目用的方式）

工具描述的重要性：
  工具的描述决定了LLM是否会正确调用它
  
  ❌ 差的描述：
    name: "search"
    description: "搜索功能"
  
  ✅ 好的描述：
    name: "search_medical_knowledge"  
    description: "在医学知识库中搜索相关文献。输入查询关键词，返回最相关的医学知识片段。
                  适用于：需要查询疾病症状、治疗方案、诊断标准等医学知识时使用。
                  不适用于：查询患者个人信息、预约挂号等非知识类查询。"
```

---

## 五、框架对比

| 框架 | 设计哲学 | 核心特点 | 适用场景 |
|------|----------|----------|----------|
| **LangGraph** | 状态图驱动 | 显式定义节点/边/状态、条件路由、持久化 | 有明确流程的复杂工作流（你的项目） |
| **AutoGen** | 对话驱动 | Agent之间通过对话协作、代码执行能力强 | 多Agent讨论、代码生成 |
| **CrewAI** | 角色扮演 | 定义Agent角色/目标/背景、任务编排 | 快速搭建多Agent团队 |
| **OpenAI Swarm** | 最小化 | 极简API、Agent之间handoff | 轻量级多Agent、客服场景 |
| **Dify** | Low-Code | 可视化工作流编排、丰富的集成 | 快速原型、非技术人员 |

**面试中你应该这样回答框架选型问题：**

> "我选LangGraph是因为医疗诊断有明确的流程步骤（分诊→评估→会诊→诊断），需要显式定义状态流转和条件路由。LangGraph的StateGraph天然适合这种场景。如果是开放讨论类场景（如多专家自由讨论病例），AutoGen的对话驱动模式可能更适合。如果需要快速原型验证，CrewAI上手最快。"

---

## 六、你的项目中Agent模式的优化方向

### 6.1 当前架构的不足

```
1. Supervisor路由是硬编码的条件判断
   现在：if has_triaged and not has_diagnosis → recommend
   改进：用LLM做路由决策（更灵活但更慢，需要权衡）

2. 没有Reflection机制
   现在：Agent一次性输出结果
   改进：加入自检步骤，诊断建议输出前先自我评估

3. 没有长期记忆
   现在：InMemorySaver只在session内有效
   改进：用SQLite/PostgreSQL做Checkpointer，跨session保留

4. experts_node没有真正的辩论
   现在：三个专家并行输出 → 简单拼接
   改进：加入辩论/投票环节，专家之间可以质疑对方观点
```

### 6.2 可以加的新模式

```
方案1：给recommend_node加Reflection
  recommend_node输出诊断建议 → 自检Agent评估 → 发现问题 → 修正建议
  
方案2：给Supervisor加LLM路由
  用LLM分析当前状态决定下一步，而不是硬编码if-else
  
方案3：给experts_node加辩论
  专家A: "我认为是坏死性软组织感染，因为..."
  专家B: "我不同意，LRINEC评分只有4分，更可能是蜂窝织炎"
  专家A: "但患者有糖尿病和高龄这两个高危因素..."
  裁判C: "综合两位意见，建议先按坏死性软组织感染的标准处理..."
```

---

## 七、面试高频问题速答

**Q1：你用了什么Agent设计模式？为什么？**
> 主架构用Supervisor模式：supervisor_node作为路由器，根据诊疗状态（has_triaged/has_diagnosis）分派任务给4个专业节点。选Supervisor是因为医疗诊断有明确的流程递进关系。在experts_node内部用了Map-Reduce模式：三个专家并行分析后汇总意见。在triage_node用了类似Debate的双Agent评估。

**Q2：ReAct和Plan-and-Execute的区别？各自适用什么场景？**
> ReAct是边思考边行动（Thought-Action-Observation循环），适合工具调用和简单推理场景。Plan-and-Execute是先规划完整步骤再逐步执行，适合复杂多步骤任务。我的项目在单节点内部用ReAct（如recommend_node调用工具），在整体架构层面更接近Plan-and-Execute（Supervisor规划流程，各节点执行）。

**Q3：Agent如何处理错误和失败？**
> 三个层面：1）工具调用失败：fallback策略，如get_diagnostic_tests失败后调用get_common_diagnostic_methods获取通用检查；2）Agent推理错误：可以加Reflection自检；3）整体流程异常：try-catch + 错误消息注入状态，让下游节点知道上游出了问题。

**Q4：Supervisor模式的缺点是什么？怎么改进？**
> 缺点：1）Supervisor是单点故障，路由错误影响全局；2）Worker不能直接通信。改进：1）用LLM做路由决策替代硬编码规则，提升灵活性；2）引入共享状态让Worker可以读取其他Worker的中间结果；3）对关键路由加入Reflection验证。

**Q5：LangGraph和AutoGen有什么区别？**
> LangGraph基于状态图，显式定义节点、边和状态流转，适合有明确流程的场景（如医疗诊断流程）。AutoGen基于对话，Agent通过消息交流协作，适合开放讨论场景（如多人头脑风暴）。LangGraph的优势是可控性和可调试性，AutoGen的优势是灵活性和代码执行能力。

---

*你的项目已经用了Supervisor + ReAct + Map-Reduce三种模式，面试时把这些讲清楚就很有说服力。进阶方向：加入Reflection和Plan-and-Execute。*
