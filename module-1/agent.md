[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239232-lesson-6-agent)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/agent.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239232-lesson-6-agent)


# Agent 智能体（Agent）

## Review 复习

We built a router.

我们构建了一个路由器。

* Our chat model will decide to make a tool call or not based upon the user input
  - 我们的聊天模型将根据用户输入决定是否调用工具

* We use a conditional edge to route to a node that will call our tool or simply end
  - 我们使用条件边，将流程路由至调用工具的节点，或直接结束

![Screenshot 2024-08-21 at 12.44.33 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbac0ba0bd34b541c448cc_agent1.png)

## Goals 目标

Now, we can extend this into a generic agent architecture.

现在，我们可以将该结构扩展为一种通用的智能体架构。

In the above router, we invoked the model and, if it chose to call a tool, we returned a `ToolMessage` to the user.

在上述路由器中，我们调用了模型；若模型选择调用工具，则向用户返回一个 `ToolMessage`。

But, what if we simply pass that `ToolMessage` *back to the model*?

但如果我们将该 `ToolMessage` *重新传回模型* 呢？

We can let it either (1) call another tool or (2) respond directly.

模型便可选择：（1）调用另一个工具，或（2）直接响应。

This is the intuition behind [ReAct](https://react-lm.github.io/), a general agent architecture.

这正是 [ReAct](https://react-lm.github.io/) 这一通用智能体架构背后的核心直觉。

* `act` - let the model call specific tools 
  - `act`（执行）—— 让模型调用特定工具

* `observe` - pass the tool output back to the model 
  - `observe`（观察）—— 将工具输出传回模型

* `reason` - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)
  - `reason`（推理）—— 让模型基于工具输出进行推理，以决定下一步操作（例如：调用另一工具，或直接响应）

This [general purpose architecture](https://blog.langchain.com/planning-for-agents/) can be applied to many types of tools.

这种[通用架构](https://blog.langchain.com/planning-for-agents/)可应用于多种类型的工具。



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

Here, we'll use [LangSmith](https://docs.langchain.com/langsmith/home) for [tracing](https://docs.langchain.com/langsmith/observability-concepts).

此处，我们将使用 [LangSmith](https://docs.langchain.com/langsmith/home) 实现[追踪（tracing）](https://docs.langchain.com/langsmith/observability-concepts)。

We'll log to a project, `langchain-academy`.

我们将日志记录到项目 `langchain-academy` 中。



```python
_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
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

# For this ipynb we set parallel tool calling to false as math generally is done sequentially, and this time we have 3 tools that can do math
# the OpenAI model specifically defaults to parallel tool calling for efficiency, see https://python.langchain.com/docs/how_to/tool_calling_parallel/
# play around with it and see how the model behaves with math equations!
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
```

Let's create our LLM and prompt it with the overall desired agent behavior.

让我们创建 LLM，并通过提示词设定其整体期望的智能体行为。



```python
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
```

As before, we use `MessagesState` and define a `Tools` node with our list of tools.

与之前一样，我们使用 `MessagesState`，并基于工具列表定义一个 `Tools` 节点。

The `Assistant` node is just our model with bound tools.

`Assistant` 节点即绑定了工具的模型。

We create a graph with `Assistant` and `Tools` nodes.

我们创建一个包含 `Assistant` 和 `Tools` 节点的图。

We add `tools_condition` edge, which routes to `End` or to `Tools` based on  whether the `Assistant` calls a tool.

我们添加 `tools_condition` 边，该边依据 `Assistant` 是否调用工具，将流程路由至 `End` 或 `Tools`。

Now, we add one new step:

现在，我们新增一个步骤：

We connect the `Tools` node *back* to the `Assistant`, forming a loop.

我们将 `Tools` 节点 *反向连接* 至 `Assistant`，从而构成一个循环。

* After the `assistant` node executes, `tools_condition` checks if the model's output is a tool call.
  - `assistant` 节点执行完毕后，`tools_condition` 检查模型输出是否为工具调用。

* If it is a tool call, the flow is directed to the `tools` node.
  - 若是工具调用，则流程被导向 `tools` 节点。

* The `tools` node connects back to `assistant`.
  - `tools` 节点再连接回 `assistant`。

* This loop continues as long as the model decides to call tools.
  - 只要模型持续决定调用工具，该循环便持续运行。

* If the model response is not a tool call, the flow is directed to END, terminating the process.
  - 若模型响应并非工具调用，则流程被导向 END，终止整个过程。



```python
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
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


    
![jpeg](agent_files/agent_10_0.jpg)
    



```python
messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
messages = react_graph.invoke({"messages": messages})
```


```python
for m in messages['messages']:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Add 3 and 4. Multiply the output by 2. Divide the output by 5
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      add (call_i8zDfMTdvmIG34w4VBA3m93Z)
     Call ID: call_i8zDfMTdvmIG34w4VBA3m93Z
      Args:
        a: 3
        b: 4
    =================================[1m Tool Message [0m=================================
    Name: add
    
    7
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      multiply (call_nE62D40lrGQC7b67nVOzqGYY)
     Call ID: call_nE62D40lrGQC7b67nVOzqGYY
      Args:
        a: 7
        b: 2
    =================================[1m Tool Message [0m=================================
    Name: multiply
    
    14
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      divide (call_6Q9SjxD2VnYJqEBXFt7O1moe)
     Call ID: call_6Q9SjxD2VnYJqEBXFt7O1moe
      Args:
        a: 14
        b: 5
    =================================[1m Tool Message [0m=================================
    Name: divide
    
    2.8
    ==================================[1m Ai Message [0m==================================
    
    The final result after performing the operations \( (3 + 4) \times 2 \div 5 \) is 2.8.


## LangSmith

We can look at traces in LangSmith.

我们可在 LangSmith 中查看追踪记录。
