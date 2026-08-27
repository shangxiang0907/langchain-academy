[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/multiple-schemas.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239434-lesson-3-multiple-schemas)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/multiple-schemas.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239434-lesson-3-multiple-schemas)


# Multiple Schemas 多种 Schema

## Review 复习

We just covered state schema and reducers.

我们刚刚介绍了状态 Schema 和规约器（reducers）。

Typically, all graph nodes communicate with a single schema.

通常，所有图节点都通过单一 Schema 进行通信。

Also, this single schema contains the graph's input and output keys / channels.

此外，该单一 Schema 包含图的输入与输出键（即通道）。

## Goals 目标

But, there are cases where we may want a bit more control over this:

但某些情况下，我们可能希望对此拥有更多控制权：

* Internal nodes may pass information that is *not required* in the graph's input / output.
  - 内部节点可能传递一些*并非图输入／输出所必需*的信息。

* We may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.
  - 我们还可能希望为图使用不同的输入／输出 Schema。例如，输出可能仅包含一个相关的输出键。

We'll discuss a few ways to customize graphs with multiple schemas.

我们将讨论几种使用多种 Schema 自定义图的方法。



```python
%%capture --no-stderr
%pip install --quiet -U langgraph
```

## Private State 私有状态

First, let's cover the case of passing [private state](https://docs.langchain.com/oss/python/langgraph/use-graph-api#pass-private-state-between-nodes) between nodes.

首先，让我们介绍在节点之间传递[私有状态](https://docs.langchain.com/oss/python/langgraph/use-graph-api#pass-private-state-between-nodes)的情形。

This is useful for anything needed as part of the intermediate working logic of the graph, but not relevant for the overall graph input or output.

这适用于图中间工作逻辑所需、但与整体图输入或输出无关的任何内容。

We'll define an `OverallState` and a `PrivateState`.

我们将定义一个 `OverallState` 和一个 `PrivateState`。

`node_2` uses `PrivateState` as input, but writes out to `OverallState`.

`node_2` 以 `PrivateState` 作为输入，但写入到 `OverallState`。



```python
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

class OverallState(TypedDict):
    foo: int

class PrivateState(TypedDict):
    baz: int

def node_1(state: OverallState) -> PrivateState:
    print("---Node 1---")
    return {"baz": state['foo'] + 1}

def node_2(state: PrivateState) -> OverallState:
    print("---Node 2---")
    return {"foo": state['baz'] + 1}

# Build graph
builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# Add
graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](multiple-schemas_files/multiple-schemas_4_0.png)
    



```python
graph.invoke({"foo" : 1})
```

    ---Node 1---
    ---Node 2---





    {'foo': 3}



`baz` is only included in `PrivateState`.

`baz` 仅包含在 `PrivateState` 中。

`node_2` uses `PrivateState` as input, but writes out to `OverallState`.

`node_2` 以 `PrivateState` 作为输入，但写入到 `OverallState`。

So, we can see that `baz` is excluded from the graph output because it is not in `OverallState`.

因此，我们可以看到 `baz` 被排除在图输出之外，因为它不在 `OverallState` 中。


## Input / Output Schema 输入／输出 Schema

By default, `StateGraph` takes in a single schema and all nodes are expected to communicate with that schema.

默认情况下，`StateGraph` 接收单一 Schema，且所有节点均需按该 Schema 进行通信。

However, it is also possible to [define explicit input and output schemas for a graph](https://docs.langchain.com/oss/python/langgraph/use-graph-api#define-input-and-output-schemas).

不过，也可以[为图明确定义输入和输出 Schema](https://docs.langchain.com/oss/python/langgraph/use-graph-api#define-input-and-output-schemas)。

In these cases, we often define an "internal" schema that contains *all* keys relevant to graph operations.

在这些情形下，我们通常定义一个“内部” Schema，其中包含*所有*与图操作相关的键。

But we use specific `input` and `output` schemas to constrain the input and output.

但我们使用特定的 `input` 和 `output` Schema 来约束图的输入与输出。

First, let's just run the graph with a single schema.

首先，我们仅用单一 Schema 运行该图。



```python
class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: OverallState):
    return {"answer": "bye", "notes": "... his name is Lance"}

def answer_node(state: OverallState):
    return {"answer": "bye Lance"}

graph = StateGraph(OverallState)
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)
graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](multiple-schemas_files/multiple-schemas_8_0.png)
    


Notice that the output of invoke contains all keys in `OverallState`.

注意，`invoke` 的输出包含 `OverallState` 中的所有键。



```python
graph.invoke({"question":"hi"})
```




    {'question': 'hi', 'answer': 'bye Lance', 'notes': '... his name is Lance'}



Now, let's use a specific `input` and `output` schema with our graph.

现在，让我们为图使用特定的 `input` 和 `output` Schema。

Here, `input` / `output` schemas perform *filtering* on what keys are permitted on the input and output of the graph.

此处，`input`／`output` Schema 对图输入与输出所允许的键执行*过滤*。

In addition, we can use a type hint `state: InputState` to specify the input schema of each of our nodes.

此外，我们可以使用类型提示 `state: InputState` 来指定每个节点的输入 Schema。

This is important when the graph is using multiple schemas.

当图使用多种 Schema 时，这一点尤为重要。

We use type hints below to, for example, show that the output of `answer_node` will be filtered to `OutputState`.

我们在下方使用类型提示，例如表明 `answer_node` 的输出将被过滤为 `OutputState`。



```python
class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: InputState):
    return {"answer": "bye", "notes": "... his is name is Lance"}

def answer_node(state: OverallState) -> OutputState:
    return {"answer": "bye Lance"}

graph = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
graph.add_node("answer_node", answer_node)
graph.add_node("thinking_node", thinking_node)
graph.add_edge(START, "thinking_node")
graph.add_edge("thinking_node", "answer_node")
graph.add_edge("answer_node", END)

graph = graph.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

graph.invoke({"question":"hi"})
```


    
![png](multiple-schemas_files/multiple-schemas_12_0.png)
    





    {'question': 'hi', 'answer': 'bye Lance', 'notes': '... his is name is Lance'}



We can see the `output` schema constrains the output to only the `answer` key.

我们可以看到 `output` Schema 将输出约束为仅 `answer` 键。
