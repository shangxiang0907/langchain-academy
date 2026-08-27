[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent-memory.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239417-lesson-7-agent-with-memory)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent-memory.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239417-lesson-7-agent-with-memory)


# Agent memory 代理记忆

## Review 回顾

Previously, we built an agent that can:

此前，我们构建了一个能够执行以下操作的代理：

* `act` - let the model call specific tools 
  - `act`（执行）——让模型调用特定工具

* `observe` - pass the tool output back to the model 
  - `observe`（观察）——将工具输出传回模型

* `reason` - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)
  - `reason`（推理）——让模型基于工具输出进行推理，以决定下一步操作（例如：调用另一工具，或直接响应）

![Screenshot 2024-08-21 at 12.45.32 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbab7453080e6802cd1703_agent-memory1.png)

## Goals 目标

Now, we're going extend our agent by introducing memory.

现在，我们将通过引入记忆来扩展我们的代理。



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

We'll use [LangSmith](https://docs.langchain.com/langsmith/home) for [tracing](https://docs.langchain.com/langsmith/observability-concepts).

我们将使用 [LangSmith](https://docs.langchain.com/langsmith/home) 进行 [追踪](https://docs.langchain.com/langsmith/observability-concepts)。



```python
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

This follows what we did previously.

这延续了我们此前的做法。



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

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "qwen-plus"), base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
llm_with_tools = llm.bind_tools(tools)
```


```python
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
```


```python
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Show
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
```


    
![jpeg](agent-memory_files/agent-memory_9_0.jpg)
    


## Memory 记忆

Let's run our agent, as before.

让我们像之前一样运行我们的代理。



```python
messages = [HumanMessage(content="Add 3 and 4.")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Add 3 and 4.
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      add (call_zZ4JPASfUinchT8wOqg9hCZO)
     Call ID: call_zZ4JPASfUinchT8wOqg9hCZO
      Args:
        a: 3
        b: 4
    =================================[1m Tool Message [0m=================================
    Name: add
    
    7
    ==================================[1m Ai Message [0m==================================
    
    The sum of 3 and 4 is 7.


Now, let's multiply by 2!

现在，让我们将其乘以 2！



```python
messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Multiply that by 2.
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      multiply (call_prnkuG7OYQtbrtVQmH2d3Nl7)
     Call ID: call_prnkuG7OYQtbrtVQmH2d3Nl7
      Args:
        a: 2
        b: 2
    =================================[1m Tool Message [0m=================================
    Name: multiply
    
    4
    ==================================[1m Ai Message [0m==================================
    
    The result of multiplying 2 by 2 is 4.


We don't retain memory of 7 from our initial chat!

我们并未保留初始对话中得到的数字 7！

This is because [state is transient](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9291220) to a single graph execution.

这是因为 [状态是瞬态的](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9291220)，仅存在于单次图执行过程中。

Of course, this limits our ability to have multi-turn conversations with interruptions.

当然，这限制了我们开展含中断的多轮对话的能力。

We can use [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) to address this!

我们可以使用 [持久化](https://docs.langchain.com/oss/python/langgraph/persistence) 来解决这一问题！

LangGraph can use a checkpointer to automatically save the graph state after each step.

LangGraph 可借助检查点器（checkpointer）在每一步后自动保存图状态。

This built-in persistence layer gives us memory, allowing LangGraph to pick up from the last state update.

这一内置持久化层为我们提供了记忆能力，使 LangGraph 能够从上一次状态更新处继续执行。

One of the easiest checkpointers to use is the `MemorySaver`, an in-memory key-value store for Graph state.

最易使用的检查点器之一是 `MemorySaver`，它是一个用于图状态的内存内键值存储。

All we need to do is simply compile the graph with a checkpointer, and our graph has memory!

我们只需在编译图时传入一个检查点器，该图便具备了记忆能力！



```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)
```

When we use memory, we need to specify a `thread_id`.

使用记忆时，我们需要指定一个 `thread_id`。

This `thread_id` will store our collection of graph states.

该 `thread_id` 将用于存储我们的一组图状态。

Here is a cartoon:

下图是一个示意图：

* The checkpointer write the state at every step of the graph
  - 检查点器在图的每一步都写入状态

* These checkpoints are saved in a thread 
  - 这些检查点被保存在一个线程（thread）中

* We can access that thread in the future using the `thread_id`
  - 未来我们可通过 `thread_id` 访问该线程

![state.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e0e9f526b41a4ed9e2d28b_agent-memory2.png)



```python
# Specify a thread
config = {"configurable": {"thread_id": "1"}}

# Specify an input
messages = [HumanMessage(content="Add 3 and 4.")]

# Run
messages = react_graph_memory.invoke({"messages": messages},config)
for m in messages['messages']:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Add 3 and 4.
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      add (call_MSupVAgej4PShIZs7NXOE6En)
     Call ID: call_MSupVAgej4PShIZs7NXOE6En
      Args:
        a: 3
        b: 4
    =================================[1m Tool Message [0m=================================
    Name: add
    
    7
    ==================================[1m Ai Message [0m==================================
    
    The sum of 3 and 4 is 7.


If we pass the same `thread_id`, then we can proceed from from the previously logged state checkpoint!

如果我们传入相同的 `thread_id`，便可从先前记录的状态检查点处继续执行！

In this case, the above conversation is captured in the thread.

本例中，上述对话已被捕获在线程中。

The `HumanMessage` we pass (`"Multiply that by 2."`) is appended to the above conversation.

我们传入的 `HumanMessage`（即 `"Multiply that by 2."`）将追加到上述对话之后。

So, the model now know that `that` refers to the `The sum of 3 and 4 is 7.`.

因此，模型现在知道 `that` 指代的是 `The sum of 3 and 4 is 7.`。



```python
messages = [HumanMessage(content="Multiply that by 2.")]
messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Add 3 and 4.
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      add (call_MSupVAgej4PShIZs7NXOE6En)
     Call ID: call_MSupVAgej4PShIZs7NXOE6En
      Args:
        a: 3
        b: 4
    =================================[1m Tool Message [0m=================================
    Name: add
    
    7
    ==================================[1m Ai Message [0m==================================
    
    The sum of 3 and 4 is 7.
    ================================[1m Human Message [0m=================================
    
    Multiply that by 2.
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      multiply (call_fWN7lnSZZm82tAg7RGeuWusO)
     Call ID: call_fWN7lnSZZm82tAg7RGeuWusO
      Args:
        a: 7
        b: 2
    =================================[1m Tool Message [0m=================================
    Name: multiply
    
    14
    ==================================[1m Ai Message [0m==================================
    
    The result of multiplying 7 by 2 is 14.


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

详细的设置说明请参阅本课程开头的“开始设置”指南。

You can find a description of Studio [here](https://docs.langchain.com/langsmith/studio), and specific details for local deployment [here](https://docs.langchain.com/langsmith/quick-start-studio#local-development-server).

您可在此处查阅 Studio 的介绍 [此处](https://docs.langchain.com/langsmith/studio)，以及本地部署的具体细节 [此处](https://docs.langchain.com/langsmith/quick-start-studio#local-development-server)。

To start the local development server, run the following command in your terminal in the `/studio` directory in this module:

要启动本地开发服务器，请在本模块的 `/studio` 目录下于终端中运行以下命令：

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

Load the `agent` in Studio, which uses `module-1/studio/agent.py` set in `module-1/studio/langgraph.json`.

在 Studio 中加载 `agent`，其对应 `module-1/studio/agent.py`，并在 `module-1/studio/langgraph.json` 中配置。
