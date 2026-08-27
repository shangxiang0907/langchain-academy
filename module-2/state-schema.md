[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/state-schema.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239426-lesson-1-state-schema)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/state-schema.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239426-lesson-1-state-schema)


# State Schema  状态模式

## Review 回顾

In module 1, we laid the foundations!

在模块 1 中，我们打下了基础！

We built up to an agent that can:

我们构建了一个具备以下能力的智能体：

* `act` - let the model call specific tools 
  - `act`（执行）——让模型调用特定工具

* `observe` - pass the tool output back to the model 
  - `observe`（观察）——将工具输出传回模型

* `reason` - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)
  - `reason`（推理）——让模型基于工具输出进行推理，以决定下一步操作（例如，调用另一个工具或直接响应）

* `persist state` - use an in memory checkpointer to support long-running conversations with interruptions
  - `persist state`（持久化状态）——使用内存中的检查点器（checkpointer），支持带有中断的长时间运行对话

And, we showed how to serve it locally in LangGraph Studio or deploy it with LangGraph Cloud.

此外，我们还演示了如何在 LangGraph Studio 中本地运行该智能体，或通过 LangGraph Cloud 部署它。

## Goals 学习目标

In this module, we're going to build a deeper understanding of both state and memory.

在本模块中，我们将深入理解状态与记忆。

First, let's review a few different ways to define your state schema.

首先，让我们回顾几种定义状态模式的不同方式。



```python
%%capture --no-stderr
%pip install --quiet -U langgraph
```

## Schema 模式

When we define a LangGraph `StateGraph`, we use a [state schema](https://docs.langchain.com/oss/python/langgraph/graph-api/#state).

当我们定义一个 LangGraph `StateGraph` 时，需使用 [状态模式](https://docs.langchain.com/oss/python/langgraph/graph-api/#state)。

The state schema represents the structure and types of data that our graph will use.

状态模式表示图将使用的数据结构与类型。

All nodes are expected to communicate with that schema.

所有节点都应依据该模式进行通信。

LangGraph offers flexibility in how you define your state schema, accommodating various Python [types](https://docs.python.org/3/library/stdtypes.html#type-objects) and validation approaches!

LangGraph 在状态模式的定义方式上提供了灵活性，支持多种 Python [类型](https://docs.python.org/3/library/stdtypes.html#type-objects)及验证方法！

## TypedDict

As we mentioned in Module 1, we can use the `TypedDict` class from python's `typing` module.

如模块 1 所述，我们可以使用 Python `typing` 模块中的 `TypedDict` 类。

It allows you to specify keys and their corresponding value types.

它允许你指定键及其对应值的类型。

But, note that these are type hints.

但请注意，这些仅为类型提示。

They can be used by static type checkers (like [mypy](https://github.com/python/mypy)) or IDEs to catch potential type-related errors before the code is run.

它们可被静态类型检查器（如 [mypy](https://github.com/python/mypy)）或 IDE 用于在代码运行前捕获潜在的类型相关错误。

But they are not enforced at runtime!

但它们在运行时并不强制执行！



```python
from typing_extensions import TypedDict

class TypedDictState(TypedDict):
    foo: str
    bar: str
```

For more specific value constraints, you can use things like the `Literal` type hint.

若需更具体的值约束，可使用 `Literal` 等类型提示。

Here, `mood` can only be either "happy" or "sad".

此处，`mood` 只能是 "happy" 或 "sad"。



```python
from typing import Literal

class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy","sad"]
```

We can use our defined state class (e.g., here `TypedDictState`) in LangGraph by simply passing it to `StateGraph`.

我们可在 LangGraph 中通过将已定义的状态类（例如此处的 `TypedDictState`）直接传入 `StateGraph` 来使用它。

And, we can think about each state key as just a "channel" in our graph.

同时，我们可以将每个状态键视作图中的一个“通道”。

As discussed in Module 1, we overwrite the value of a specified key or "channel" in each node.

如模块 1 所述，我们在每个节点中覆写指定键（即“通道”）的值。



```python
import random
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

def node_1(state):
    print("---Node 1---")
    return {"name": state['name'] + " is ... "}

def node_2(state):
    print("---Node 2---")
    return {"mood": "happy"}

def node_3(state):
    print("---Node 3---")
    return {"mood": "sad"}

def decide_mood(state) -> Literal["node_2", "node_3"]:
        
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"

# Build graph
builder = StateGraph(TypedDictState)
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


    
![jpeg](state-schema_files/state-schema_8_0.jpg)
    


Because our state is a dict, we simply invoke the graph with a dict to set an initial value of the `name` key in our state.

由于我们的状态是一个字典，只需传入一个字典即可为状态中的 `name` 键设置初始值。



```python
graph.invoke({"name":"Lance"})
```

    ---Node 1---
    ---Node 2---





    {'name': 'Lance is ... ', 'mood': 'happy'}



## Dataclass 数据类（Dataclass）

Python's [dataclasses](https://docs.python.org/3/library/dataclasses.html) provide [another way to define structured data](https://www.datacamp.com/tutorial/python-data-classes).

Python 的 [dataclasses](https://docs.python.org/3/library/dataclasses.html) 提供了 [另一种定义结构化数据的方式](https://www.datacamp.com/tutorial/python-data-classes)。

Dataclasses offer a concise syntax for creating classes that are primarily used to store data.

数据类提供了一种简洁语法，用于创建主要用途为存储数据的类。



```python
from dataclasses import dataclass

@dataclass
class DataclassState:
    name: str
    mood: Literal["happy","sad"]
```

To access the keys of a `dataclass`, we just need to modify the subscripting used in `node_1`:

要访问 `dataclass` 的键，我们只需修改 `node_1` 中使用的下标访问方式：

* We use `state.name` for the `dataclass` state rather than `state["name"]` for the `TypedDict` above
  - 对于 `dataclass` 状态，我们使用 `state.name`；而对于上方的 `TypedDict` 状态，则使用 `state["name"]`

You'll notice something a bit odd: in each node, we still return a dictionary to perform the state updates.

你会注意到一个略显奇怪的现象：在每个节点中，我们仍返回一个字典来执行状态更新。

This is possible because LangGraph stores each key of your state object separately.

这是可行的，因为 LangGraph 将状态对象的每个键单独存储。

The object returned by the node only needs to have keys (attributes) that match those in the state!

节点所返回的对象只需包含与状态中匹配的键（属性）即可！

In this case, the `dataclass` has key `name` so we can update it by passing a dict from our node, just as we did when state was a `TypedDict`.

本例中，`dataclass` 具有键 `name`，因此我们可通过节点返回字典来更新它，这与状态为 `TypedDict` 时的操作完全一致。



```python
def node_1(state):
    print("---Node 1---")
    return {"name": state.name + " is ... "}

# Build graph
builder = StateGraph(DataclassState)
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


    
![jpeg](state-schema_files/state-schema_14_0.jpg)
    


We invoke with a `dataclass` to set the initial values of each key / channel in our state!

我们通过传入一个 `dataclass` 实例，来为状态中的每个键／通道设置初始值！



```python
graph.invoke(DataclassState(name="Lance",mood="sad"))
```

    ---Node 1---
    ---Node 3---





    {'name': 'Lance is ... ', 'mood': 'sad'}



## Pydantic

As mentioned, `TypedDict` and `dataclasses` provide type hints but they don't enforce types at runtime.

如前所述，`TypedDict` 和 `dataclasses` 仅提供类型提示，而不在运行时强制执行类型。

This means you could potentially assign invalid values without raising an error!

这意味着你可能在不引发错误的情况下赋给变量非法值！

For example, we can set `mood` to `mad` even though our type hint specifies `mood: list[Literal["happy","sad"]]`.

例如，尽管我们的类型提示声明为 `mood: list[Literal["happy","sad"]]`，我们仍可将 `mood` 设为 `mad`。



```python
dataclass_instance = DataclassState(name="Lance", mood="mad")
```

[Pydantic](https://docs.pydantic.dev/latest/api/base_model/) is a data validation and settings management library using Python type annotations.

[Pydantic](https://docs.pydantic.dev/latest/api/base_model/) 是一个利用 Python 类型注解实现数据验证和配置管理的库。

It's particularly well-suited [for defining state schemas in LangGraph](https://docs.langchain.com/oss/python/langgraph/use-graph-api#use-pydantic-models-for-graph-state) due to its validation capabilities.

得益于其强大的验证能力，Pydantic 特别适合 [在 LangGraph 中定义状态模式](https://docs.langchain.com/oss/python/langgraph/use-graph-api#use-pydantic-models-for-graph-state)。

Pydantic can perform validation to check whether data conforms to the specified types and constraints at runtime.

Pydantic 可在运行时执行验证，以检查数据是否符合指定的类型与约束条件。



```python
from pydantic import BaseModel, field_validator, ValidationError

class PydanticState(BaseModel):
    name: str
    mood: str # "happy" or "sad" 

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        # Ensure the mood is either "happy" or "sad"
        if value not in ["happy", "sad"]:
            raise ValueError("Each mood must be either 'happy' or 'sad'")
        return value

try:
    state = PydanticState(name="John Doe", mood="mad")
except ValidationError as e:
    print("Validation Error:", e)
```

    Validation Error: 1 validation error for PydanticState
    mood
      Input should be 'happy' or 'sad' [type=literal_error, input_value='mad', input_type=str]
        For further information visit https://errors.pydantic.dev/2.8/v/literal_error


We can use `PydanticState` in our graph seamlessly.

我们可以无缝地在图中使用 `PydanticState`。



```python
# Build graph
builder = StateGraph(PydanticState)
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


    
![jpeg](state-schema_files/state-schema_22_0.jpg)
    



```python
graph.invoke(PydanticState(name="Lance",mood="sad"))
```

    ---Node 1---
    ---Node 3---





    {'name': 'Lance is ... ', 'mood': 'sad'}




```python

```
