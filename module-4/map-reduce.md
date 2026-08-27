[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/map-reduce.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239947-lesson-3-map-reduce)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/map-reduce.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239947-lesson-3-map-reduce)


# Map-reduce Map-Reduce（映射-归约）

## Review 复习

We're building up to a multi-agent research assistant that ties together all of the modules from this course.

我们正在构建一个多功能智能体研究助手，该助手将整合本课程所有模块的内容。

To build this multi-agent assistant, we've been introducing a few LangGraph controllability topics.

为构建这一多智能体助手，我们已陆续介绍了若干 LangGraph 可控性主题。

We just covered parallelization and sub-graphs.

我们刚刚讲解了并行化与子图。

## Goals 目标

Now, we're going to cover [map reduce](https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api).

接下来，我们将学习 [map reduce](https://docs.langchain.com/oss/python/langgraph/use-graph-api#map-reduce-and-the-send-api)。



```python
%%capture --no-stderr
%pip install -U langchain_openai langgraph
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

## Problem 问题

Map-reduce operations are essential for efficient task decomposition and parallel processing.

Map-reduce 操作对于高效任务分解与并行处理至关重要。

It has two phases:

它包含两个阶段：

(1) `Map` - Break a task into smaller sub-tasks, processing each sub-task in parallel.

（1）`Map` —— 将一项任务拆分为若干更小的子任务，并对每个子任务进行并行处理。

(2) `Reduce` - Aggregate the results across all of the completed, parallelized sub-tasks.

（2）`Reduce` —— 汇总所有已完成的并行化子任务的结果。

Let's design a system that will do two things:

让我们设计一个具备以下两项功能的系统：

(1) `Map` - Create a set of jokes about a topic.

（1）`Map` —— 为某一主题生成一组笑话。

(2) `Reduce` - Pick the best joke from the list.

（2）`Reduce` —— 从该列表中选出最佳笑话。

We'll use an LLM to do the job generation and selection.

我们将使用 LLM 完成任务生成与筛选工作。



```python
import os
from langchain_openai import ChatOpenAI

# Prompts we will use
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

# LLM
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "qwen-plus"), base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), temperature=0) 
```

## State 状态

### Parallelizing joke generation 并行化笑话生成

First, let's define the entry point of the graph that will:

首先，定义图的入口点，其功能包括：

* Take a user input topic
  - 接收用户输入的主题

* Produce a list of joke topics from it
  - 从中生成一组笑话主题

* Send each joke topic to our above joke generation node
  - 将每个笑话主题发送至上述笑话生成节点

Our state has a `jokes` key, which will accumulate jokes from parallelized joke generation

我们的状态中有一个 `jokes` 键，用于累积来自并行化笑话生成的结果



```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel

class Subjects(BaseModel):
    subjects: list[str]

class BestJoke(BaseModel):
    id: int
    
class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str
```

Generate subjects for jokes.

生成笑话主题。



```python
def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}
```

Here is the magic: we use the  [Send](https://docs.langchain.com/oss/python/langgraph/graph-api/#send) to create a joke for each subject.

此处的关键在于：我们使用 [Send](https://docs.langchain.com/oss/python/langgraph/graph-api/#send) 为每个主题生成一则笑话。

This is very useful!

这非常实用！

It can automatically parallelize joke generation for any number of subjects.

它能自动为任意数量的主题并行化笑话生成。

* `generate_joke`: the name of the node in the graph
  - `generate_joke`：图中节点的名称

* `{"subject": s`}: the state to send
  - `{"subject": s}`：待发送的状态

`Send` allow you to pass any state that you want to `generate_joke`!

`Send` 允许你向 `generate_joke` 传递任意所需状态！

It does not have to align with `OverallState`.

该状态无需与 `OverallState` 对齐。

In this case, `generate_joke` is using its own internal state, and we can populate this via `Send`.

本例中，`generate_joke` 使用其自身的内部状态，而我们可通过 `Send` 填充该状态。



```python
from langgraph.types import Send
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
```

### Joke generation (map) 笑话生成（map）

Now, we just define a node that will create our jokes, `generate_joke`!

现在，我们只需定义一个用于生成笑话的节点 `generate_joke`！

We write them back out to `jokes` in `OverallState`!

我们将生成的笑话写回 `OverallState` 中的 `jokes` 字段！

This key has a reducer that will combine lists.

该字段配备了一个用于合并列表的归约器（reducer）。



```python
class JokeState(TypedDict):
    subject: str

class Joke(BaseModel):
    joke: str

def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}
```

### Best joke selection (reduce) 最佳笑话筛选（reduce）

Now, we add logic to pick the best joke.

接下来，我们添加逻辑以筛选出最佳笑话。



```python
def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}
```

## Compile 编译



```python
from IPython.display import Image
from langgraph.graph import END, StateGraph, START

# Construct the graph: here we put everything together to construct our graph
graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)

# Compile the graph
app = graph.compile()
Image(app.get_graph().draw_mermaid_png())
```




    
![jpeg](map-reduce_files/map-reduce_19_0.jpg)
    




```python
# Call the graph: here we call it to generate a list of jokes
for s in app.stream({"topic": "animals"}):
    print(s)
```

    {'generate_topics': {'subjects': ['mammals', 'reptiles', 'birds']}}
    {'generate_joke': {'jokes': ["Why don't mammals ever get lost? Because they always follow their 'instincts'!"]}}
    {'generate_joke': {'jokes': ["Why don't alligators like fast food? Because they can't catch it!"]}}
    {'generate_joke': {'jokes': ["Why do birds fly south for the winter? Because it's too far to walk!"]}}
    {'best_joke': {'best_selected_joke': "Why don't alligators like fast food? Because they can't catch it!"}}


## Studio

**⚠️ Notice**

⚠️ 注意

Since filming these videos, we've updated Studio so that it can now be run locally and accessed through your browser.

自录制这些视频以来，我们已更新 Studio，使其支持本地运行并通过浏览器访问。

This is the preferred way to run Studio instead of using the Desktop App shown in the video.

推荐采用此方式运行 Studio，而非视频中演示的桌面应用。

It is now called _LangSmith Studio_ instead of _LangGraph Studio_.

它现被称为 _LangSmith Studio_，而非 _LangGraph Studio_。

Detailed setup instructions are available in the "Getting Setup" guide at the start of the course.

详细安装说明请参阅本课程开篇的“环境准备”指南。

You can find a description of Studio [here](https://docs.langchain.com/langsmith/studio), and specific details for local deployment [here](https://docs.langchain.com/langsmith/quick-start-studio#local-development-server).

您可在此处查阅 Studio 的介绍 [链接](https://docs.langchain.com/langsmith/studio)，以及本地部署的具体细节 [链接](https://docs.langchain.com/langsmith/quick-start-studio#local-development-server)。

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

Let's load the above graph in the Studio UI, which uses `module-4/studio/map_reduce.py` set in `module-4/studio/langgraph.json`.

让我们在 Studio UI 中加载上述图，其对应文件为 `module-4/studio/map_reduce.py`，并在 `module-4/studio/langgraph.json` 中指定。



```python

```
