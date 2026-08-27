[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/simple-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58238187-lesson-2-simple-graph)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-1/simple-graph.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58238187-lesson-2-simple-graph)


# The Simplest Graph 最简单的图

Let's build a simple graph with 3 nodes and one conditional edge.

我们来构建一个包含 3 个节点和一条条件边的简单图。

![Screenshot 2024-08-20 at 3.11.22 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dba5f465f6e9a2482ad935_simple-graph1.png)



```python
%%capture --no-stderr
%pip install --quiet -U langgraph
```

## State 状态

First, define the [State](https://docs.langchain.com/oss/python/langgraph/graph-api#state) of the graph.

首先，定义图的 [State](https://docs.langchain.com/oss/python/langgraph/graph-api#state)。

The State schema serves as the input schema for all Nodes and Edges in the graph.

状态（State）模式作为图中所有节点（Nodes）和边（Edges）的输入模式。

Let's use the `TypedDict` class from python's `typing` module as our schema, which provides type hints for the keys.

我们使用 Python `typing` 模块中的 `TypedDict` 类作为我们的模式，它为键提供类型提示。



```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

## Nodes 节点

[Nodes](https://docs.langchain.com/oss/python/langgraph/graph-api/#nodes) are just python functions.

[节点（Nodes）](https://docs.langchain.com/oss/python/langgraph/graph-api/#nodes) 就是普通的 Python 函数。

The first positional argument is the state, as defined above.

第一个位置参数是上文定义的状态（state）。

Because the state is a `TypedDict` with schema as defined above, each node can access the key, `graph_state`, with `state['graph_state']`.

由于该状态是一个 `TypedDict`，其模式如上所定义，因此每个节点均可通过 `state['graph_state']` 访问键 `graph_state`。

Each node returns a new value of the state key `graph_state`.

每个节点返回状态键 `graph_state` 的新值。

By default, the new value returned by each node [will override](https://docs.langchain.com/oss/python/langgraph/graph-api/#reducers) the prior state value.

默认情况下，每个节点返回的新值将[覆盖](https://docs.langchain.com/oss/python/langgraph/graph-api/#reducers)之前的状态值。



```python
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}
```

## Edges 边

[Edges](https://docs.langchain.com/oss/python/langgraph/graph-api/#edges) connect the nodes.

[边（Edges）](https://docs.langchain.com/oss/python/langgraph/graph-api/#edges) 连接各个节点。

Normal Edges are used if you want to *always* go from, for example, `node_1` to `node_2`.

若希望*始终*从例如 `node_1` 跳转到 `node_2`，则使用普通边（Normal Edges）。

[Conditional Edges](https://docs.langchain.com/oss/python/langgraph/graph-api/#conditional-edges) are used if you want to *optionally* route between nodes.

若希望*可选地*在节点之间路由，则使用[条件边（Conditional Edges）](https://docs.langchain.com/oss/python/langgraph/graph-api/#conditional-edges)。

Conditional edges are implemented as functions that return the next node to visit based on some logic.

条件边以函数形式实现，该函数根据某些逻辑返回下一个要访问的节点。



```python
import random
from typing import Literal

def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"
```

## Graph Construction 图的构建

Now, we build the graph from our components defined above.

现在，我们基于上文定义的组件来构建图。

The [StateGraph class](https://docs.langchain.com/oss/python/langgraph/graph-api/#stategraph) is the graph class that we can use.

[StateGraph 类](https://docs.langchain.com/oss/python/langgraph/graph-api/#stategraph) 是我们可用的图类。

First, we initialize a StateGraph with the `State` class we defined above.

首先，我们使用上文定义的 `State` 类初始化一个 StateGraph。

Then, we add our nodes and edges.

然后，我们添加节点和边。

We use the  [`START` Node, a special node](https://docs.langchain.com/oss/python/langgraph/graph-api/#start-node) that sends user input to the graph, to indicate where to start our graph.

我们使用 [`START` 节点（一种特殊节点）](https://docs.langchain.com/oss/python/langgraph/graph-api/#start-node)，它将用户输入发送至图中，以指明图的起始位置。

The [`END` Node](https://docs.langchain.com/oss/python/langgraph/graph-api/#end-node) is a special node that represents a terminal node.

[`END` 节点](https://docs.langchain.com/oss/python/langgraph/graph-api/#end-node) 是一种表示终止节点的特殊节点。

Finally, we [compile our graph](https://docs.langchain.com/oss/python/langgraph/graph-api/#compiling-your-graph) to perform a few basic checks on the graph structure.

最后，我们[编译图](https://docs.langchain.com/oss/python/langgraph/graph-api/#compiling-your-graph)，以对图结构执行若干基本检查。

We can visualize the graph as a [Mermaid diagram](https://github.com/mermaid-js/mermaid).

我们可以将图可视化为一张 [Mermaid 图](https://github.com/mermaid-js/mermaid)。



```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![jpeg](simple-graph_files/simple-graph_10_0.jpg)
    


## Graph Invocation 图的调用

The compiled graph implements the [runnable](https://reference.langchain.com/python/langchain_core/runnables/?h=runnables) protocol.

已编译的图实现了 [runnable](https://reference.langchain.com/python/langchain_core/runnables/?h=runnables) 协议。

This provides a standard way to execute LangChain components.

这为执行 LangChain 组件提供了标准方式。

`invoke` is one of the standard methods in this interface.

`invoke` 是该接口中的标准方法之一。

The input is a dictionary `{"graph_state": "Hi, this is lance."}`, which sets the initial value for our graph state dict.

输入是一个字典 `{"graph_state": "Hi, this is lance."}`，用于设置图状态字典的初始值。

When `invoke` is called, the graph starts execution from the `START` node.

当调用 `invoke` 时，图从 `START` 节点开始执行。

It progresses through the defined nodes (`node_1`, `node_2`, `node_3`) in order.

它按顺序遍历已定义的节点（`node_1`、`node_2`、`node_3`）。

The conditional edge will traverse from node `1` to node `2` or `3` using a 50/50 decision rule.

条件边将依据 50/50 决策规则，从节点 `1` 跳转至节点 `2` 或 `3`。

Each node function receives the current state and returns a new value, which overrides the graph state.

每个节点函数接收当前状态，并返回一个新值，该值将覆盖图状态。

The execution continues until it reaches the `END` node.

执行持续进行，直至到达 `END` 节点。



```python
graph.invoke({"graph_state" : "Hi, this is Lance."})
```

    ---Node 1---
    ---Node 3---





    {'graph_state': 'Hi, this is Lance. I am sad!'}



`invoke` runs the entire graph synchronously.

`invoke` 同步运行整个图。

This waits for each step to complete before moving to the next.

它会等待每一步完成后再进入下一步。

It returns the final state of the graph after all nodes have executed.

它返回所有节点执行完毕后的图最终状态。

In this case, it returns the state after `node_3` has completed:

本例中，它返回 `node_3` 执行完毕后的状态：

```
{'graph_state': 'Hi, this is Lance. I am sad!'}
```



```python

```
