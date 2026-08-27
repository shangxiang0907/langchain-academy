[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/router.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239412-lesson-5-router)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/router.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239412-lesson-5-router)


# Router 路由器

## Review 复习

We built a graph that uses `messages` as state and a chat model with bound tools.

我们构建了一个以 `messages` 为状态、并绑定了工具的聊天模型图。

We saw that the graph can:

我们看到该图可以：

* Return a tool call
  - 返回一个工具调用

* Return a natural language response
  - 返回一段自然语言响应

## Goals 目标

We can think of this as a router, where the chat model routes between a direct response or a tool call based upon the user input.

我们可以将此视为一个路由器，其中聊天模型根据用户输入，在直接响应与工具调用之间进行路由。

This is a simple example of an agent, where the LLM is directing the control flow either by calling a tool or just responding directly.

这是一个智能体（agent）的简单示例，其中 LLM 通过调用工具或直接响应来主导控制流。

![Screenshot 2024-08-21 at 9.24.09 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbac6543c3d4df239a4ed1_router1.png)

Let's extend our graph to work with either output!

让我们扩展该图，使其能处理任一输出！

For this, we can use two ideas:

为此，我们可以采用两个思路：

(1) Add a node that will call our tool.

（1）添加一个用于调用工具的节点。

(2) Add a conditional edge that will look at the chat model output, and route to our tool calling node or simply end if no tool call is performed.

（2）添加一条条件边，用于检查聊天模型的输出，并据此路由至工具调用节点；若未执行工具调用，则直接结束。



```python
%%capture --no-stderr
%pip install --quiet -U langchain_openai langchain_core langgraph langgraph-prebuilt
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


```python
import os
from langchain_openai import ChatOpenAI

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "qwen-plus"), base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
llm_with_tools = llm.bind_tools([multiply])
```

We use the  [built-in `ToolNode`](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.ToolNode) and simply pass a list of our tools to initialize it.

我们使用 [内置的 `ToolNode`](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.ToolNode)，只需传入工具列表即可完成初始化。

We use the [built-in `tools_condition`](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.tools_condition) as our conditional edge.

我们使用 [内置的 `tools_condition`](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.tools_condition) 作为条件边。



```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](router_files/router_6_0.png)
    



```python
from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Hello world.
    ==================================[1m Ai Message [0m==================================
    
    Hello! How can I assist you today?


Now, we can see that the graph runs the tool!

现在，我们可以看到该图成功运行了工具！

It responds with a `ToolMessage`.

它返回了一个 `ToolMessage`。


## Studio

**⚠️ Notice**

**⚠️ 注意**

Since filming these videos, we've updated Studio so that it can now be run locally and accessed through your browser.

自录制这些视频以来，我们已更新 Studio，使其现在可本地运行并通过浏览器访问。

This is the preferred way to run Studio instead of using the Desktop App shown in the video.

这是运行 Studio 的首选方式，而非视频中展示的桌面应用。

It is now called _LangSmith Studio_ instead of _LangGraph Studio_.

它现在被称为 _LangSmith Studio_，而非 _LangGraph Studio_。

Detailed setup instructions are available in the "Getting Setup" guide at the start of the course.

详细的安装说明请参阅本课程开头的“环境准备”指南。

You can find a description of Studio [here](https://docs.langchain.com/langsmith/studio), and specific details for local deployment [here](https://docs.langchain.com/langsmith/quick-start-studio#local-development-server).

您可在此处查看 Studio 的介绍：[链接](https://docs.langchain.com/langsmith/studio)，以及本地部署的具体细节：[链接](https://docs.langchain.com/langsmith/quick-start-studio#local-development-server)。

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

Open your browser and navigate to the Studio UI shown above.

打开浏览器，导航至上方所示的 Studio 用户界面。

Load the `router` in Studio, which uses `module-1/studio/router.py` set in `module-1/studio/langgraph.json`.

在 Studio 中加载 `router`，其对应文件为 `module-1/studio/router.py`，并在 `module-1/studio/langgraph.json` 中指定。



```python
if 'google.colab' in str(get_ipython()):
    raise Exception("Unfortunately LangGraph Studio is currently not supported on Google Colab")
```


```python

```
