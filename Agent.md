# Agent 修改日志

## 修改记录

| 时间 | 修改位置 | 修改方法 | 说明 |
|------|----------|----------|------|
| 2026-02-26 19:14 | `o:\BiyeCompsition\毕业论文框架.md` | 新建文件 | 根据项目实际内容创建毕业论文框架，包含目录结构、各章节写作思路、配图建议、文献引用建议 |
| 2026-02-26 19:14 | `o:\BiyeCompsition\Agent.md` | 新建文件 | 创建Agent修改日志 |
| 2026-02-26 19:20 | `o:\BiyeCompsition\毕业论文框架.md` 第二章目录+写作思路+图表清单 | 内容替换 | 用户为AI专业，重构第二章：去掉前端(Vue/Element Plus/Electron)和后端(FastAPI)Web技术栈介绍，改为以AI核心技术为主线（LLM→智能体→知识图谱→RAG→MCP） |
| 2026-02-26 19:25 | `o:\BiyeCompsition\毕业论文框架.md` 全文多处 | 内容精简 | 用户为本科生，降低论文深度：①第二章目录去掉三级标题，每节1-2页；②写作思路改为通俗介绍（去掉公式推导、框架对比），多用类比；③图表从33张精简到20张；④字数从2-3万降为1.5-2万；⑤文献从20-30篇降为15-20篇；⑥新增本科论文常见问题提醒 |
| 2026-02-26 19:40 | `o:\BiyeCompsition\毕业论文框架.md` 2.2节+4.5节 | 内容补充 | 用户提供modelserver/start.sh，补充模型部署方案：AutoDL云GPU + Ollama部署Qwen2.5-14B + cpolar内网穿透(zjlchat.vip.cpolar.cn:6006)；4.5节部署架构改为三部分（客户端本地+服务端云端+AutoDL模型服务器） |
| 2026-02-26 19:45 | `o:\BiyeCompsition\毕业论文框架.md` 多处 | 纠正+插入 | ①经代码验证，项目实际使用SSE而非WebSocket（后端有WS代码但前端未调用），修正所有WebSocket→SSE；②插入14张Mermaid预绘图 |
| 2026-02-26 19:48 | `o:\BiyeCompsition\毕业论文框架.md` 3.3节+4.x节+目录 | 内容补充 | 用户指出服务端设计缺失。经代码验证，服务端mcp_server_service.py(800+行)是完整的知识管理平台，远不只RAG。补充：①3.3.4服务端知识管理模块（文档管理API/异步处理+SSE进度/数据一致性管理）；②4.4服务端知识管理实现（REST API/异步SSE/KnowledgeDataManager三方数据同步）；③新增图4-2 PDF异步处理时序图；④更新功能模块结构图；⑤修复章节编号顺延 |
| 2026-02-26 20:13 | `o:\BiyeCompsition\毕业论文\` | 新建目录+文件 | 创建毕业论文文档文件夹，含8个章节子文件夹（第一章～第六章+参考文献+致谢）及综合注意事项文档 |
| 2026-02-26 20:24 | `o:\BiyeCompsition\毕业论文\` 全部章节文件夹 | 新建文件+内容更新 | ①每章创建`写作思路.md`（8个）和`图表.md`（6个，含35张Mermaid图）；②去掉Transformer架构示意图，替换为LLM部署架构图（AutoDL+Ollama+cpolar），所有图均与本系统直接相关；③更新`注意事项.md`为总览索引，同步图表总清单 |

## 本次修改详情

### 1. 创建毕业论文框架 (`毕业论文框架.md`)

<!-- 
背景说明：
用户提供了一个"淘宝评论情感分析"的论文目录模板，但项目实际内容是
"基于多智能体与知识图谱的智能医疗辅助诊断系统"，因此需要完全重新设计目录结构。

项目分析过程：
1. 探索了项目根目录，发现 medical_client 和 medical_server 两个核心目录
2. 阅读了项目文档（一、作品基本情况.txt、项目文档.txt、技术架构说明.txt）
3. 深入阅读了核心代码文件：
   - Agent/flow.py — LangGraph状态图，5节点多智能体工作流
   - MCP/mcp_server.py — FastMCP工具服务，4个知识查询工具
   - Construct/knowledge_workflow.py — 知识图谱自动化构建流水线
   - front_c/src/App.vue — 客户端Vue3前端（诊断界面）
   - front_s/src/components/ — 服务端前端（知识管理界面）
4. 阅读了开题报告，获取已有的10篇参考文献
5. 阅读了requirements文件，掌握了完整的技术栈信息

框架设计原则：
- 按照标准毕业论文六章结构（前言→技术基础→需求设计→实现→测试→总结）
- 每个小节标注了📊（需要配图）、📚（需要文献）、💡（写作思路）
- 末尾附有图表清单（约25张图表）和写作建议
-->

### 2. 关键知识点

<!-- 
项目技术架构：
- 三层架构：客户端层(medical_client) + 服务端层(medical_server) + 数据层(Neo4j/Redis)
- 客户端技术栈：Python + LangChain 0.3.27 + LangGraph 0.6.8 + FastAPI + Vue3 + Electron
- 服务端技术栈：Python + FastAPI + FastMCP + Docling + Neo4j + Redis + Vue3 + ECharts
- 通信协议：MCP(SSE) + REST API + WebSocket

多智能体工作流（5节点）：
- supervisor_node: 任务分类与路由（LLM意图识别）
- triage_node: 并行分诊（医学顾问 + 急诊分诊双智能体）
- recommend_node: 风险评估（症状匹配 + Softmax概率计算）
- agen_node(experts): 多专家会诊（诊断/治疗/影像三专家 + RAG检索）
- other_node(query): 知识查询（ReAct智能体 + MCP工具）

知识图谱构建七步流水线：
PDF解析(Docling) → HTML清洗(BeautifulSoup) → Markdown转换 → 文本分块(RecursiveCharacterTextSplitter) → 实体关系抽取(LLM) → Neo4j导入(py2neo) → 向量化索引(M3E+Redis)

MCP工具（4个）：
1. symptom_search_analyze — 症状向量检索与疾病分析
2. get_common_diagnostic_methods — 通用诊断方法查询
3. get_diagnostic_tests_for_disease — 特定疾病诊断方法
4. retrieve_medical_knowledge_vector — Redis向量检索医学知识

Neo4j图谱模型：
- 节点：Disease, Symptom, RiskFactor, Pathogen, Treatment, DiagnosticTest
- 关系：HAS_SYMPTOM, HAS_RISK_FACTOR, HAS_PATHOGEN, HAS_TREATMENT, REQUIRES_TEST/DIAGNOSED_BY
- 向量索引：enhanced_symptom_vectors (768维, 余弦相似度)

前端组件：
- 客户端(front_c): PatientList.vue + PatientDetail.vue + ChatPanel.vue (三栏布局)
- 服务端(front_s): KnowledgeConstruction.vue + PDFProgressDialog.vue
-->
