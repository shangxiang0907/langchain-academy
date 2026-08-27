---
title: "Module 1 Resources / 模块1资源"
source: "https://academy.langchain.com/courses/take/intro-to-langgraph/texts/58260773-module-1-resources"
saved_at: "2026-08-22"
---

# Module 1 Resources

**中文：模块1资源**

![Course image](https://files.cdn.thinkific.com/file_uploads/967498/images/d16/3c0/07a/Course_Banner.png)

### Lesson 1: Motivation

**中文：第1课：动机**

- [LangChain Academy - Introduction to LangGraph - Motivation.pdf](https://files.cdn.thinkific.com/file_uploads/967498/attachments/ecd/3cc/6d3/LangChain_Academy_-_Introduction_to_LangGraph_-_Motivation.pdf)
  - 中文：LangChain Academy——LangGraph 入门——动机 PDF。

### LangGraph FAQ

**中文：LangGraph 常见问题**

- **Do I need to use LangChain to use LangGraph? What’s the difference?**
  - **中文：使用 LangGraph 是否必须使用 LangChain？两者有什么区别？**
  - No.
    - 中文：不需要。
  - LangGraph is an orchestration framework for complex agentic systems and is more low-level and controllable than LangChain agents.
    - 中文：LangGraph 是一个面向复杂智能体系统的编排框架，与 LangChain 智能体相比，它更底层，也更具可控性。
  - On the other hand, LangChain provides a standard interface to interact with models and other components, useful for straight-forward chains and retrieval flows.
    - 中文：另一方面，LangChain 提供了与模型及其他组件交互的标准接口，适用于直接的链式流程和检索流程。

- **How is LangGraph different from other agent frameworks?**
  - **中文：LangGraph 与其他智能体框架有何不同？**
  - Other agentic frameworks can work for simple, generic tasks but fall short for complex tasks bespoke to a company’s needs.
    - 中文：其他智能体框架可以处理简单、通用的任务，但面对需要根据企业需求定制的复杂任务时，往往力有不逮。
  - LangGraph provides a more expressive framework to handle companies’ unique tasks without restricting users to a single black-box cognitive architecture.
    - 中文：LangGraph 提供了表达能力更强的框架，可处理企业特有的任务，同时不会把用户限制在单一的黑盒认知架构中。

- **Does LangGraph impact the performance of my app?**
  - **中文：LangGraph 会影响我的应用性能吗？**
  - LangGraph will not add any overhead to your code and is specifically designed with streaming workflows in mind.
    - 中文：LangGraph 不会给代码增加额外开销，并且在设计时专门考虑了流式工作流。

- **Is LangGraph open source? Is it free?**
  - **中文：LangGraph 是开源的吗？可以免费使用吗？**
  - Yes.
    - 中文：是的。
  - LangGraph is an MIT-licensed open-source library and is free to use.
    - 中文：LangGraph 是采用 MIT 许可证的开源库，可以免费使用。

- **Is LangSmith Deployment (formerly LangGraph Platform/Cloud) open source?**
  - **中文：LangSmith Deployment（原 LangGraph Platform/Cloud）是开源的吗？**
  - No.
    - 中文：不是。
  - [LangSmith Deployment](https://docs.langchain.com/langgraph-platform) is proprietary software that will eventually be a paid service for certain tiers of usage.
    - 中文：[LangSmith Deployment](https://docs.langchain.com/langgraph-platform) 是专有软件，最终会针对某些使用层级提供付费服务。
  - We will always give ample notice before charging for a service and reward our early adopters with preferential pricing.
    - 中文：我们会在服务开始收费前充分提前通知，并以优惠价格回馈早期采用者。

- **How do I enable LangSmith Deployment?**
  - **中文：如何启用 LangSmith Deployment？**
  - All LangSmith users on Plus and Enterprise plans can access LangSmith Deployment.
    - 中文：所有使用 LangSmith Plus 和 Enterprise 套餐的用户都可以访问 LangSmith Deployment。
  - Check out the [docs](https://docs.langchain.com/langgraph-platform).
    - 中文：请查看相关[文档](https://docs.langchain.com/langgraph-platform)。

- **How are LangGraph and LangSmith Deployment different?**
  - **中文：LangGraph 与 LangSmith Deployment 有何不同？**
  - LangGraph is a stateful, orchestration framework that brings added control to agent workflows.
    - 中文：LangGraph 是一个有状态的编排框架，可为智能体工作流提供更强的控制能力。
  - LangSmith Deployment is a service for deploying and scaling LangGraph applications.
    - 中文：LangSmith Deployment 是用于部署和扩展 LangGraph 应用的服务。

- **How does LangGraph fit into the LangChain ecosystem?**
  - **中文：LangGraph 在 LangChain 生态系统中处于什么位置？**
  - Our open source frameworks help you build agents:
    - 中文：我们的开源框架可帮助你构建智能体：
    - **LangChain** helps you quickly get started building agents, with any model provider of your choice.
      - 中文：**LangChain** 帮助你快速开始构建智能体，并可自由选择任意模型提供商。
    - **LangGraph** allows you to control every step of your custom agent with low-level orchestration, memory, and human-in-the-loop support.
      - 中文：**LangGraph** 通过底层编排、记忆和人在回路支持，让你能够控制自定义智能体的每一步。
    - You can manage long-running tasks with durable execution.
      - 中文：借助持久化执行，你可以管理长时间运行的任务。
  - **LangSmith** is a platform that helps AI teams use live production data for continuous testing and improvement.
    - 中文：**LangSmith** 是一个帮助 AI 团队利用真实生产数据进行持续测试和改进的平台。
  - LangSmith provides:
    - 中文：LangSmith 提供以下能力：
    - **Observability** to see exactly how your agent thinks and acts with detailed tracing and aggregate trend metrics.
      - 中文：**可观测性**：通过详细追踪和聚合趋势指标，准确了解智能体如何思考和行动。
    - **Evaluation** to test and score agent behavior on production data and offline datasets for continuous improvement.
      - 中文：**评估**：使用生产数据和离线数据集测试并评分智能体行为，以实现持续改进。
    - **Deployment** to ship your agent in one click, using scalable infrastructure built for long-running tasks.
      - 中文：**部署**：借助专为长时间运行任务构建的可扩展基础设施，一键发布智能体。

![Course image](https://files.cdn.thinkific.com/file_uploads/967498/images/11c/7e3/e93/Agent_Stack.png)

---

### Lesson 2: Simple Graph

**中文：第2课：简单图**

- Notebook Reference: simple-graph.ipynb
  - 中文：笔记本参考文件：`simple-graph.ipynb`
- Download Notebook on [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/simple-graph.ipynb)
  - 中文：在 [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/simple-graph.ipynb) 上下载笔记本。
- View Notebook on [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/simple-graph.ipynb)
  - 中文：在 [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/simple-graph.ipynb) 上查看笔记本。

---

### Lesson 3: LangSmith Studio

**中文：第3课：LangSmith Studio**

- [LangSmith Studio](https://studio.langchain.com/)
  - 中文：[LangSmith Studio](https://studio.langchain.com/)
- Download Module 1 LangSmith Studio Files on [**GitHub**](https://github.com/langchain-ai/langchain-academy/tree/main/module-1/studio)
  - 中文：在 [**GitHub**](https://github.com/langchain-ai/langchain-academy/tree/main/module-1/studio) 上下载模块1的 LangSmith Studio 文件。

---

### Lesson 4: Chain

**中文：第4课：链**

- Notebook Reference: chain.ipynb
  - 中文：笔记本参考文件：`chain.ipynb`
- Download Notebook on [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/chain.ipynb)
  - 中文：在 [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/chain.ipynb) 上下载笔记本。
- View Notebook on [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/chain.ipynb)
  - 中文：在 [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/chain.ipynb) 上查看笔记本。

---

### Lesson 5: Router

**中文：第5课：路由器**

- Notebook Reference: router.ipynb
  - 中文：笔记本参考文件：`router.ipynb`
- Download Notebook on [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/router.ipynb)
  - 中文：在 [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/router.ipynb) 上下载笔记本。
- View Notebook on [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/router.ipynb)
  - 中文：在 [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/router.ipynb) 上查看笔记本。

---

### Lesson 6: Agent

**中文：第6课：智能体**

- Notebook Reference: agent.ipynb
  - 中文：笔记本参考文件：`agent.ipynb`
- Download Notebook on [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/agent.ipynb)
  - 中文：在 [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/agent.ipynb) 上下载笔记本。
- View Notebook on [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent.ipynb)
  - 中文：在 [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent.ipynb) 上查看笔记本。

---

### Lesson 7: Agent with Memory

**中文：第7课：具备记忆功能的智能体**

- Notebook Reference: agent-memory.ipynb
  - 中文：笔记本参考文件：`agent-memory.ipynb`
- Download Notebook on [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/agent-memory.ipynb)
  - 中文：在 [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/agent-memory.ipynb) 上下载笔记本。
- View Notebook on [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent-memory.ipynb)
  - 中文：在 [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent-memory.ipynb) 上查看笔记本。

---

### [Optional] Lesson 8: Deployment

**中文：[可选] 第8课：部署**

Currently, LangSmith Deployment is available only for LangSmith Plus plan users.

中文：目前，LangSmith Deployment 仅向 LangSmith Plus 套餐用户开放。

Lesson 8 in this course is optional.

中文：本课程的第8课为可选内容。

### LangSmith Deployment was formerly LangGraph Platform.

**中文：LangSmith Deployment 原名为 LangGraph Platform。**

- Notebook Reference: deployment.ipynb
  - 中文：笔记本参考文件：`deployment.ipynb`
- Download Notebook on [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/deployment.ipynb)
  - 中文：在 [**GitHub**](https://github.com/langchain-ai/langchain-academy/blob/main/module-1/deployment.ipynb) 上下载笔记本。
- View Notebook on [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/deployment.ipynb)
  - 中文：在 [**Google Colab**](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/deployment.ipynb) 上查看笔记本。
