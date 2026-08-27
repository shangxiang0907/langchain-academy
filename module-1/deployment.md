[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/deployment.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239303-lesson-8-deployment)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/deployment.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239303-lesson-8-deployment)


# Deployment 部署

## Review  回顾

We built up to an agent with memory:

我们已构建出一个具备记忆能力的智能体：

* `act` - let the model call specific tools 
  - `act` — 允许模型调用特定工具

* `observe` - pass the tool output back to the model 
  - `observe` — 将工具输出传回模型

* `reason` - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)
  - `reason` — 让模型基于工具输出进行推理，以决定下一步操作（例如：调用另一工具或直接响应）

* `persist state` - use an in memory checkpointer to support long-running conversations with interruptions
  - `persist state` — 使用内存中的检查点器（checkpointer），支持带中断的长时间运行对话

## Goals 目标

Now, we'll cover how to actually deploy our agent locally to Studio and to `LangGraph Cloud`.

接下来，我们将介绍如何将智能体实际部署到本地 Studio 和 `LangGraph Cloud`。



```python
%%capture --no-stderr
%pip install --quiet -U langgraph_sdk langchain_core
```

## Concepts 概念

There are a few central concepts to understand -

需理解若干核心概念 —

`LangGraph` —

`LangGraph` —

- Python and JavaScript library 
  - Python 与 JavaScript 库

- Allows creation of agent workflows 
  - 支持创建智能体工作流

`LangGraph API` —

`LangGraph API` —

- Bundles the graph code 
  - 封装图（graph）代码

- Provides a task queue for managing asynchronous operations
  - 提供任务队列，用于管理异步操作

- Offers persistence for maintaining state across interactions
  - 提供持久化能力，以在多次交互间维持状态

`LangSmith Deployment` (formerly `LangGraph Cloud`) --

`LangSmith Deployment`（原 `LangGraph Cloud`）——

- Hosted service for the LangGraph API
  - 托管式 LangGraph API 服务

- Allows deployment of graphs from GitHub repositories
  - 支持从 GitHub 仓库部署图

- Also provides monitoring and tracing for deployed graphs
  - 还为已部署的图提供监控与追踪功能

- Accessible via a unique URL for each deployment
  - 每个部署均通过唯一 URL 访问

`LangSmith Studio` (formerly `LangGraph Studio`) --

`LangSmith Studio`（原 `LangGraph Studio`）——

- Integrated Development Environment (IDE) for LangGraph applications
  - 面向 LangGraph 应用的集成开发环境（IDE）

- Uses the API as its back-end, allowing real-time testing and exploration of graphs
  - 以后端 API 为基础，支持对图进行实时测试与探索

- Can be run locally or with cloud-deployment. See below.
  - 可本地运行，亦支持云部署。详见下文。

`LangGraph SDK` --

`LangGraph SDK` ——

- Python library for programmatically interacting with LangGraph graphs
  - 用于以编程方式与 LangGraph 图交互的 Python 库

- Provides a consistent interface for working with graphs, whether served locally or in the cloud
  - 无论图是本地托管还是云端托管，均提供统一接口

- Allows creation of clients, access to assistants, thread management, and execution of runs
  - 支持创建客户端、访问助手、线程管理及执行运行（run）

## Testing Locally 本地测试


## Studio

**⚠️ Notice**

⚠️ 注意

Since filming these videos, we've updated Studio so that it can now be run locally and accessed through your browser.

自录制本系列视频以来，我们已更新 Studio，使其现在可本地运行，并可通过浏览器访问。

This is the preferred way to run Studio instead of using the Desktop App shown in the video.

这是运行 Studio 的首选方式，而非视频中演示的桌面应用。

It is now called _LangSmith Studio_ instead of _LangGraph Studio_.

它现称为 _LangSmith Studio_，而非 _LangGraph Studio_。

Detailed setup instructions are available in the "Getting Setup" guide at the start of the course.

详细安装说明请参阅本课程开头的“开始设置”指南。

You can find a description of Studio [here](https://docs.langchain.com/langsmith/studio), and specific details for local deployment [here](https://docs.langchain.com/langsmith/quick-start-studio#local-development-server).

您可在此处查看 Studio 的说明文档 [此处](https://docs.langchain.com/langsmith/studio)，本地部署的具体细节请参见 [此处](https://docs.langchain.com/langsmith/quick-start-studio#local-development-server)。

To start the local development server, run the following command in your terminal in the `/studio` directory in this module:

要在本地启动开发服务器，请在本模块的 `/studio` 目录下于终端中运行以下命令：

```
langgraph dev
```

You should see the following output:

您应看到如下输出：

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Open your browser and navigate to the **Studio UI** URL shown above.

打开浏览器并导航至上方显示的 **Studio UI** URL。



```python
if 'google.colab' in str(get_ipython()):
    raise Exception("Unfortunately LangGraph Studio is currently not supported on Google Colab")
```


```python
from langgraph_sdk import get_client
```


```python
# This is the URL of the local development server
URL = "http://127.0.0.1:2024"
client = get_client(url=URL)

# Search all hosted graphs
assistants = await client.assistants.search()
```


```python
assistants[-3]
```




    {'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca',
     'graph_id': 'agent',
     'config': {},
     'metadata': {'created_by': 'system'},
     'name': 'agent',
     'created_at': '2025-03-04T22:57:28.424565+00:00',
     'updated_at': '2025-03-04T22:57:28.424565+00:00',
     'version': 1}




```python
# We create a thread for tracking the state of our run
thread = await client.threads.create()
```

Now, we can run our agent  [with `client.runs.stream`](https://docs.langchain.com/oss/python/langgraph/graph-api/#stream-and-astream) with:

现在，我们可以使用 [`client.runs.stream`](https://docs.langchain.com/oss/python/langgraph/graph-api/#stream-and-astream) 运行我们的智能体：

* The `thread_id`

* The `graph_id`

* The `input` 

* The `stream_mode`

We'll discuss streaming in depth in a future module.

我们将在后续模块中深入探讨流式处理（streaming）。

For now, just recognize that we are [streaming](https://docs.langchain.com/langsmith/streaming) the full value of the state after each step of the graph with `stream_mode="values"`.

目前只需了解：我们正以 `stream_mode="values"` 方式 [流式传输](https://docs.langchain.com/langsmith/streaming)，即在图每一步执行后，输出完整状态值。

The state is captured in the `chunk.data`.

状态值保存在 `chunk.data` 中。



```python
from langchain_core.messages import HumanMessage

# Input
input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}

# Stream
async for chunk in client.runs.stream(
        thread['thread_id'],
        "agent",
        input=input,
        stream_mode="values",
    ):
    if chunk.data and chunk.event != "metadata":
        print(chunk.data['messages'][-1])
```

    {'content': 'Multiply 3 by 2.', 'additional_kwargs': {'example': False, 'additional_kwargs': {}, 'response_metadata': {}}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'cdbd7bd8-c476-4ad4-8ab7-4ad9e3654267', 'example': False}
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_iIPryzZZxRtXozwwhVtFObNO', 'function': {'arguments': '{"a":3,"b":2}', 'name': 'multiply'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-06c7243c-426d-4c81-a113-f1335dda5fb2', 'example': False, 'tool_calls': [{'name': 'multiply', 'args': {'a': 3, 'b': 2}, 'id': 'call_iIPryzZZxRtXozwwhVtFObNO', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}
    {'content': '6', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'multiply', 'id': '988cb170-f6e6-43c1-82fd-309f519abe6d', 'tool_call_id': 'call_iIPryzZZxRtXozwwhVtFObNO', 'artifact': None, 'status': 'success'}
    {'content': 'The result of multiplying 3 by 2 is 6.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-7bda0aa0-6895-4250-9625-18419c5dc171', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}


## Testing with Cloud 使用云服务测试

We can deploy to Cloud via LangSmith, as outlined [here](https://docs.langchain.com/langsmith/deployment-quickstart#deploy-from-github-with-langgraph-cloud).

我们可通过 LangSmith 部署至云服务，具体步骤详见 [此处](https://docs.langchain.com/langsmith/deployment-quickstart#deploy-from-github-with-langgraph-cloud)。

### Create a New Repository on GitHub 在 GitHub 上新建仓库

* Go to your GitHub account
  - 前往您的 GitHub 账户

* Click on the "+" icon in the upper-right corner and select `"New repository"`
  - 点击右上角的 “+” 图标，选择 `“New repository”`

* Name your repository (e.g., `langchain-academy`)
  - 为您的仓库命名（例如：`langchain-academy`）

### Add Your GitHub Repository as a Remote 将您的 GitHub 仓库添加为远程仓库

* Go back to your terminal where you cloned `langchain-academy` at the start of this course
  - 返回本课程初始时克隆 `langchain-academy` 的终端

* Add your newly created GitHub repository as a remote
  - 将新创建的 GitHub 仓库添加为远程仓库

```
git remote add origin https://github.com/your-username/your-repo-name.git
```

* Push to it
  - 推送至该仓库

```
git push -u origin main
```

### Connect LangSmith to your GitHub Repository 将 LangSmith 连接到您的 GitHub 仓库

* Go to [LangSmith](hhttps://smith.langchain.com/)
  - 访问 [LangSmith](hhttps://smith.langchain.com/)

* Click on `deployments` tab on the left LangSmith panel
  - 点击 LangSmith 左侧面板中的 `deployments` 标签页

* Add `+ New Deployment`
  - 添加 `+ 新建部署`

* Then, select the Github repository (e.g., `langchain-academy`) that you just created for the course
  - 然后，选择你为本课程刚刚创建的 GitHub 仓库（例如 `langchain-academy`）

* Point the `LangGraph API config file` at one of the `studio` directories
  - 将 `LangGraph API 配置文件` 指向某个 `studio` 目录

* For example, for module-1 use: `module-1/studio/langgraph.json`
  - 例如，对于 module-1，请使用：`module-1/studio/langgraph.json`

* Set your API keys (e.g., you can just copy from your `module-1/studio/.env` file)
  - 设置你的 API 密钥（例如，可直接从 `module-1/studio/.env` 文件中复制）

![Screenshot 2024-09-03 at 11.35.12 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbad4fd61c93d48e5d0f47_deployment2.png)

### Work with your deployment 与你的部署交互

We can then interact with our deployment a few different ways:

随后，我们可通过几种不同方式与部署进行交互：

* With the SDK, as before.
  - 使用 SDK，方式与之前相同。

* With [LangGraph Studio](https://docs.langchain.com/langsmith/deployment-quickstart#3-test-your-application-in-studio).
  - 使用 [LangGraph Studio](https://docs.langchain.com/langsmith/deployment-quickstart#3-test-your-application-in-studio)。

![Screenshot 2024-08-23 at 10.59.36 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbad4fa159a09a51d601de_deployment3.png)

To use the SDK here in the notebook, simply ensure that `LANGSMITH_API_KEY` is set!

若要在本笔记本中使用 SDK，请确保已设置 `LANGSMITH_API_KEY`！



```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGSMITH_API_KEY")
```


```python
# Replace this with the URL of your deployed graph
URL = "https://langchain-academy-8011c561878d50b1883f7ed11b32d720.default.us.langgraph.app"
client = get_client(url=URL)

# Search all hosted graphs
assistants = await client.assistants.search()
```


```python
# Select the agent
agent = assistants[0]
```


```python
agent
```




    {'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca',
     'graph_id': 'agent',
     'created_at': '2024-08-23T17:58:02.722920+00:00',
     'updated_at': '2024-08-23T17:58:02.722920+00:00',
     'config': {},
     'metadata': {'created_by': 'system'}}




```python
from langchain_core.messages import HumanMessage

# We create a thread for tracking the state of our run
thread = await client.threads.create()

# Input
input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}

# Stream
async for chunk in client.runs.stream(
        thread['thread_id'],
        "agent",
        input=input,
        stream_mode="values",
    ):
    if chunk.data and chunk.event != "metadata":
        print(chunk.data['messages'][-1])
```

    {'content': 'Multiply 3 by 2.', 'additional_kwargs': {'example': False, 'additional_kwargs': {}, 'response_metadata': {}}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '8ea04559-f7d4-4c82-89d9-c60fb0502f21', 'example': False}
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_EQoolxFaaSVU8HrTnCmffLk7', 'function': {'arguments': '{"a":3,"b":2}', 'name': 'multiply'}, 'type': 'function'}]}, 'response_metadata': {'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27'}, 'type': 'ai', 'name': None, 'id': 'run-b0ea5ddd-e9ba-4242-bb8c-80eb52466c76', 'example': False, 'tool_calls': [{'name': 'multiply', 'args': {'a': 3, 'b': 2}, 'id': 'call_EQoolxFaaSVU8HrTnCmffLk7', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}
    {'content': '6', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'multiply', 'id': '1bf558e7-79ef-4f21-bb66-acafbd04677a', 'tool_call_id': 'call_EQoolxFaaSVU8HrTnCmffLk7', 'artifact': None, 'status': 'success'}
    {'content': '3 multiplied by 2 equals 6.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3aa7262c27'}, 'type': 'ai', 'name': None, 'id': 'run-ecc4b6ad-af15-4a85-a76c-de2ed0ed8ed9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}



```python

```
