[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-0/basics.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/56295530-getting-set-up-video-guide)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-0/basics.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/56295530-getting-set-up-video-guide)


# LangChain Academy LangChain 学院

Welcome to LangChain Academy!

欢迎来到 LangChain 学院！

## Context 背景

At LangChain, we aim to make it easy to build LLM applications.

在 LangChain，我们的目标是让构建 LLM 应用程序变得简单。

One type of LLM application you can build is an agent.

你可以构建的一种 LLM 应用程序是智能体（agent）。

There’s a lot of excitement around building agents because they can automate a wide range of tasks that were previously impossible.

围绕智能体构建存在大量热情，因为它们可以自动化许多此前无法实现的广泛任务。

In practice though, it is incredibly difficult to build systems that reliably execute on these tasks.

但在实践中，构建能可靠执行这些任务的系统却异常困难。

As we’ve worked with our users to put agents into production, we’ve learned that more control is often necessary.

随着我们与用户合作将智能体投入生产环境，我们认识到往往需要更强的控制能力。

You might need an agent to always call a specific tool first or use different prompts based on its state.

你可能需要智能体始终优先调用某个特定工具，或根据其状态使用不同的提示词（prompt）。

To tackle this problem, we’ve built [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — a framework for building agent and multi-agent applications.

为解决这一问题，我们构建了 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) —— 一个用于构建智能体及多智能体应用程序的框架。

Separate from the LangChain package, LangGraph’s core design philosophy is to help developers add better precision and control into agent workflows, suitable for the complexity of real-world systems.

LangGraph 独立于 LangChain 包，其核心设计理念是帮助开发者在智能体工作流中加入更高精度与更强控制力，以适配真实世界系统的复杂性。

## Course Structure 课程结构

The course is structured as a set of modules, with each module focused on a particular theme related to LangGraph.

本课程由一系列模块构成，每个模块聚焦 LangGraph 的某一特定主题。

You will see a folder for each module, which contains a series of notebooks.

你将看到每个模块对应一个文件夹，其中包含一系列笔记本（notebook）。

A video will accompany each notebook to help walk through the concepts, but the notebooks are also stand-alone, meaning that they contain explanations and can be viewed independently of the videos.

每个笔记本均配有配套视频，以辅助讲解相关概念；但这些笔记本本身也是独立可运行的，即内含完整说明，可脱离视频单独查看。

Each module folder also contains a `studio` folder, which contains a set of graphs that can be loaded into [LangSmith Studio](https://docs.langchain.com/langsmith/quick-start-studio), our IDE for building LangGraph applications.

每个模块文件夹还包含一个 `studio` 文件夹，其中存放一组图（graph），可加载至 [LangSmith Studio](https://docs.langchain.com/langsmith/quick-start-studio)（我们专为构建 LangGraph 应用而设计的集成开发环境 IDE）。

## Setup 环境准备

Before you begin, please follow the instructions in the `README` to create an environment and install dependencies.

开始学习前，请按 `README` 中的说明创建运行环境并安装依赖项。

## Chat models 聊天模型

In this course, we'll use Chat Models, which take a sequence of messages as input and return messages as output.

本课程将使用聊天模型（Chat Models），其输入为消息序列，输出也为消息。

LangChain supports many models via [third-party integrations](https://docs.langchain.com/oss/python/integrations/chat).

LangChain 通过 [第三方集成](https://docs.langchain.com/oss/python/integrations/chat) 支持多种模型。

By default, the course will use  [ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai) because it is both popular and performant.

默认情况下，本课程将使用 [ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai)，因其兼具流行性与高性能。

As noted, please ensure that you have an `OPENAI_API_KEY`.

如前所述，请确保已设置 `OPENAI_API_KEY`。

Let's check that your `OPENAI_API_KEY` is set and, if not, you will be asked to enter it.

我们将检查你的 `OPENAI_API_KEY` 是否已设置；若未设置，系统将提示你输入。



```python
%%capture --no-stderr
%pip install --quiet -U langchain_openai langchain_core langchain_community langchain-tavily
```


```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))
_set_env("OPENAI_API_KEY")
```

[Here](https://docs.langchain.com/oss/python/langchain/models) is a useful how-to for all the things that you can do with chat models, but we'll show a few highlights below.

[此处](https://docs.langchain.com/oss/python/langchain/models) 提供了一份关于聊天模型所有可用功能的实用指南，但下方我们将展示其中几个重点功能。

If you've run `pip install -r requirements.txt` as noted in the README, then you've installed the `langchain-openai` package.

如 `README` 所述，若你已运行 `pip install -r requirements.txt`，则已安装 `langchain-openai` 包。

With this, we can instantiate our `ChatOpenAI` model object.

借助该包，我们可以实例化 `ChatOpenAI` 模型对象。

You can see pricing for various models [here](https://openai.com/api/pricing/).

各类模型的价格信息请参见 [此处](https://openai.com/api/pricing/)。

The notebooks will default to `gpt-4o` because it offers a good balance of quality, price, and speed, but you can also opt for the lower-priced `gpt-3.5` series or more recent models.

笔记本默认使用 `gpt-4o`，因其在质量、价格与速度之间取得了良好平衡；但你也可选择价格更低的 `gpt-3.5` 系列，或更新的模型。

There are [a few standard parameters](https://docs.langchain.com/oss/python/langchain/models#parameters) that we can set with chat models.

聊天模型支持 [若干标准参数](https://docs.langchain.com/oss/python/langchain/models#parameters)。

Two of the most common are:

其中最常用的两个参数是：

* `model`: the name of the model
  - `model`：模型名称

* `temperature`: the sampling temperature
  - `temperature`：采样温度

`Temperature` controls the randomness or creativity of the model's output where low temperature (close to 0) is more deterministic and focused outputs.

`temperature` 控制模型输出的随机性或创造性：较低温度（接近 0）使输出更确定、更聚焦。

This is good for tasks requiring accuracy or factual responses.

这适用于对准确性或事实性响应要求较高的任务。

High temperature (close to 1) is good for creative tasks or generating varied responses.

较高温度（接近 1）则适用于创意类任务或需生成多样化响应的场景。



```python
import os
from langchain_openai import ChatOpenAI
gpt4o_chat = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "qwen-plus"), base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), temperature=0)
gpt35_chat = ChatOpenAI(model=os.getenv("OPENAI_SECONDARY_MODEL", os.getenv("OPENAI_MODEL", "qwen-plus")), base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), temperature=0)
```

Chat models in LangChain have a number of [default methods](https://reference.langchain.com/python/langchain_core/runnables).

LangChain 中的聊天模型提供了若干 [默认方法](https://reference.langchain.com/python/langchain_core/runnables)。

For the most part, we'll be using:

大多数情况下，我们将使用：

* [stream](https://docs.langchain.com/oss/python/langchain/models#stream): stream back chunks of the response
  - [stream](https://docs.langchain.com/oss/python/langchain/models#stream)：流式返回响应的分块内容

* [invoke](https://docs.langchain.com/oss/python/langchain/models#invoke): call the chain on an input
  - [invoke](https://docs.langchain.com/oss/python/langchain/models#invoke)：在输入上运行链（chain）

And, as mentioned, chat models take [messages](https://docs.langchain.com/oss/python/langchain/messages) as input.

如前所述，聊天模型以 [消息](https://docs.langchain.com/oss/python/langchain/messages) 作为输入。

Messages have a role (that describes who is saying the message) and a content property.

每条消息具有一个角色（role，用于标识消息发送者）和一个内容（content）属性。

We'll be talking a lot more about this later, but here let's just show the basics.

我们将在后续深入探讨此话题，此处仅先介绍基本用法。



```python
from langchain_core.messages import HumanMessage

# Create a message
msg = HumanMessage(content="Hello world", name="Lance")

# Message list
messages = [msg]

# Invoke the model with a list of messages 
gpt4o_chat.invoke(messages)
```




    AIMessage(content='Hello! 👋 How can I help you today?', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 10, 'total_tokens': 21, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'qwen-plus', 'system_fingerprint': None, 'id': 'chatcmpl-59c1ed18-68e5-9f02-94d6-592be80dfa27', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--01a0406c-cb75-7982-aa82-a21f0359b770-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 10, 'output_tokens': 11, 'total_tokens': 21, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})



We get an `AIMessage` response.

我们得到一个 `AIMessage` 响应。

Also, note that we can just invoke a chat model with a string.

此外请注意，我们也可以直接用字符串调用聊天模型。

When a string is passed in as input, it is converted to a `HumanMessage` and then passed to the underlying model.

当传入字符串作为输入时，它会被自动转换为 `HumanMessage`，再传递给底层模型。



```python
gpt4o_chat.invoke("hello world")
```




    AIMessage(content="Hello! 👋 It's great to meet you. How can I assist you today?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 10, 'total_tokens': 28, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'qwen-plus', 'system_fingerprint': None, 'id': 'chatcmpl-c6afe24b-938a-9d92-b8f9-245f6ec6d09e', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--01a0406e-8608-71c3-9c6c-b5a4898200e8-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 10, 'output_tokens': 18, 'total_tokens': 28, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})




```python
gpt35_chat.invoke("hello world")
```




    AIMessage(content='Hello! 😊 How can I assist you today? Let me know if you have any questions or need help with something.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 25, 'prompt_tokens': 14, 'total_tokens': 39, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cache_write_tokens': None, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'qwen-turbo', 'system_fingerprint': None, 'id': 'chatcmpl-262ee9e8-b25a-97cd-ba19-95be671e92ee', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--01a0406e-198e-7960-b6d2-fa880bcc31df-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 14, 'output_tokens': 25, 'total_tokens': 39, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})



The interface is consistent across all chat models and models are typically initialized once at the start up each notebooks.

该接口在所有聊天模型中保持一致，且模型通常在每个笔记本启动时初始化一次。

So, you can easily switch between models without changing the downstream code if you have strong preference for another provider.

因此，如果你强烈偏好其他服务商，只需更换模型即可，无需修改下游代码。


## Search Tools 搜索工具

You'll also see [Tavily](https://tavily.com/) in the README, which is a search engine optimized for LLMs and RAG, aimed at efficient, quick, and persistent search results.

你还会在 `README` 中看到 [Tavily](https://tavily.com/)，这是一个专为 LLM 和 RAG 优化的搜索引擎，旨在提供高效、快速且持久的搜索结果。

As mentioned, it's easy to sign up and offers a generous free tier.

如前所述，注册十分便捷，且提供 generous 免费额度。

Some lessons (in Module 4) will use Tavily by default but, of course, other search tools can be used if you want to modify the code for yourself.

部分课程（模块 4 中）将默认使用 Tavily，但当然，你也可以根据自身需求修改代码，改用其他搜索工具。



```python
_set_env("TAVILY_API_KEY")
```


```python
from langchain_tavily import TavilySearch  # updated at 1.0

tavily_search = TavilySearch(max_results=3)

data = tavily_search.invoke({"query": "What is LangGraph?"})
search_docs = data.get("results", data)
```


```python
search_docs
```




    [{'url': 'https://www.ibm.com/think/topics/langgraph',
      'title': 'What is LangGraph?',
      'content': 'LangGraph, created by LangChain, is an open source AI agent framework designed to build, deploy and manage complex generative AI agent workflows. It provides a set of tools and libraries that enable users to create, run and optimize large language models (LLMs) in a scalable and efficient manner. At its core, LangGraph uses the power of graph-based architectures to model and manage the intricate relationships between various components of an AI agent workflow. [...] Agent systems: LangGraph provides a framework for building agent-based systems, which can be used in applications such as robotics, autonomous vehicles or video games.\n\nLLM applications: By using LangGraph’s capabilities, developers can build more sophisticated AI models that learn and improve over time. Norwegian Cruise Line uses LangGraph to compile, construct and refine guest-facing AI solutions. This capability allows for improved and personalized guest experiences. [...] LangGraph illuminates the processes within an AI workflow, allowing full transparency of the agent’s state. Within LangGraph, the “state” feature serves as a memory bank that records and tracks all the valuable information processed by the AI system. It’s similar to a digital notebook where the system captures and updates data as it moves through various stages of a workflow or graph analysis.',
      'score': 0.9580107,
      'raw_content': None,
      'id': '0e688c-00'},
     {'url': 'https://docs.langchain.com/oss/python/langgraph/overview',
      'title': 'LangGraph overview - Docs by LangChain',
      'content': 'Trusted by companies shaping the future of agents—including Klarna, Uber, J.P. Morgan, and more—LangGraph is a low-level orchestration framework and runtime for building, managing, and deploying long-running, stateful agents. LangGraph gives you fine-grained control to mix deterministic, hand-coded steps with LLM-driven agentic steps in the same graph, so you can build bespoke agents that behave exactly the way your application requires. LangGraph is very low-level, and focused entirely on [...] architectures for common LLM and tool-calling loops. LangGraph is focused on the underlying capabilities important for agent orchestration: durable execution, streaming, human-in-the-loop, and more. One of LangGraph’s core strengths is the ability to mix deterministic steps with LLM-driven agentic steps in a single graph. This lets you build bespoke workflows where parts of the logic are fully predictable and auditable while other parts are flexible and model-driven, giving you fine-grained [...] LangGraph is inspired by Pregel and Apache Beam. The public interface draws inspiration from NetworkX. LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain. \n\nConnect these docs to Claude, VSCode, and more via MCP for real-time answers.\n\nEdit this page on GitHub or file an issue.\n\nWas this page helpful?',
      'score': 0.95154464,
      'raw_content': None,
      'id': '97864c-01'},
     {'url': 'https://www.geeksforgeeks.org/machine-learning/what-is-langgraph',
      'title': 'What is LangGraph - GeeksforGeeks',
      'content': 'geeksforgeeks\n\nsearch icon\n\n Interview Prep\n\n DSA\n Practice Problems\n C\n C++\n Java\n Python\n JavaScript\n Data Science\n Machine Learning\n Courses\n Linux\n DevOps\n\n# What is LangGraph\n\nLast Updated : 14 Apr, 2026\n\nLangGraph is an open-source framework from LangChain designed to build and manage AI agent workflows using graph-based structures. It allows developers to define workflows as nodes and edges, making complex agent interactions more structured, scalable and easier to control. [...] langgraph: Framework for building graph-based AI workflows.\n langchain: Popular toolkit for LLM-powered AI applications.\n google-generativeai: Google’s API for Generative AI (Gemini models).\n\n Python  ````\n! pip install langgraph langchain google - generativeai\n```` \n\n### Step 2: Setup Gemini API [...] ## Building a Simple Chatbot with LangGraph\n\nLangGraph makes it easy to build structured, stateful applications like chatbots. In this example we’ll learn how to create a basic chatbot that can classify user input as either a greet, search query and respond accordingly.\n\n### Step 1: Install the Dependencies\n\nInstalls the required dependencies,',
      'score': 0.9467988,
      'raw_content': None,
      'id': '37df7e-02'}]


```python

```
