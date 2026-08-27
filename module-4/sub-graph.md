[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-4/sub-graph.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239937-lesson-2-sub-graphs)


# Sub-graphs 子图

## Review 复习

We're building up to a multi-agent research assistant that ties together all of the modules from this course.

我们正在构建一个多功能智能体研究助手，该助手将本课程所有模块整合在一起。

We just covered parallelization, which is one important LangGraph controllability topic.

我们刚刚学习了并行化，这是 LangGraph 可控性的一个重要主题。

## Goals 目标

Now, we're [going to cover sub-graphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs).

接下来，我们将[学习子图](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)。

## State 状态

Sub-graphs allow you to create and manage different states in different parts of your graph.

子图允许你在图的不同部分创建和管理不同的状态。

This is particularly useful for multi-agent systems, with teams of agents that each have their own state.

这对多智能体系统尤其有用——例如由多个智能体组成的团队，每个智能体都拥有自己的状态。

Let's consider a toy example:

我们来看一个简化示例：

* I have a system that accepts logs
  - 我有一个接收日志的系统

* It performs two separate sub-tasks by different agents (summarize logs, find failure modes)
  - 它由不同智能体执行两项独立的子任务（摘要日志、识别故障模式）

* I want to perform these two operations in two different sub-graphs.
  - 我希望在这两个不同的子图中分别执行这两项操作。

The most critical thing to understand is how the graphs communicate!

最关键的是要理解图之间如何通信！

In short, communication is **done with over-lapping keys**:

简而言之，通信是通过**重叠的键（keys）**实现的：

* The sub-graphs can access `docs` from the parent
  - 子图可访问父图中的 `docs`

* The parent can access `summary/failure_report` from the sub-graphs
  - 父图可访问子图中的 `summary/failure_report`

![subgraph.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbb1abf89f2d847ee6f1ff_sub-graph1.png)

## Input 输入

Let's define a schema for the logs that will be input to our graph.

我们来定义将输入到图中的日志的结构。



```python
%%capture --no-stderr
%pip install -U  langgraph
```

We'll use [LangSmith](https://docs.langchain.com/langsmith/home) for [tracing](https://docs.langchain.com/langsmith/observability-concepts).

我们将使用 [LangSmith](https://docs.langchain.com/langsmith/home) 进行[追踪（tracing）](https://docs.langchain.com/langsmith/observability-concepts)。



```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```


```python
from operator import add
from typing_extensions import TypedDict
from typing import List, Optional, Annotated

# The structure of the logs
class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]
```

## Sub graphs 子图

Here is the failure analysis sub-graph, which uses `FailureAnalysisState`.

以下是故障分析子图，它使用 `FailureAnalysisState`。



```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Failure Analysis Sub-graph
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]

def get_failures(state):
    """ Get logs that contain a failure """
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}

def generate_summary(state):
    """ Generate summary of failures """
    failures = state["failures"]
    # Add fxn: fa_summary = summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {"fa_summary": fa_summary, "processed_logs": [f"failure-analysis-on-log-{failure['id']}" for failure in failures]}

fa_builder = StateGraph(state_schema=FailureAnalysisState,output_schema=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)

graph = fa_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](sub-graph_files/sub-graph_7_0.png)
    


Here is the question summarization sub-grap, which uses `QuestionSummarizationState`.

以下是问题摘要子图，它使用 `QuestionSummarizationState`。



```python
# Summarization subgraph
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]

def generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    # Add fxn: summary = summarize(generate_summary)
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {"qs_summary": summary, "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]}

def send_to_slack(state):
    qs_summary = state["qs_summary"]
    # Add fxn: report = report_generation(qs_summary)
    report = "foo bar baz"
    return {"report": report}

qs_builder = StateGraph(QuestionSummarizationState,output_schema=QuestionSummarizationOutputState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)

graph = qs_builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](sub-graph_files/sub-graph_9_0.png)
    


## Adding sub graphs to our parent graph 将子图添加到父图

Now, we can bring it all together.

现在，我们可以将所有内容整合起来。

We create our parent graph with `EntryGraphState`.

我们使用 `EntryGraphState` 创建父图。

And we add our sub-graphs as nodes!

并将我们的子图作为节点添加进去！

```
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())
```



```python
# Entry Graph
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: Annotated[List[Log], add] # This will be USED BY in BOTH sub-graphs
    fa_summary: str # This will only be generated in the FA sub-graph
    report: str # This will only be generated in the QS sub-graph
    processed_logs:  Annotated[List[int], add] # This will be generated in BOTH sub-graphs
```

But, why does `cleaned_logs` have a reducer if it only goes *into* each sub-graph as an input?

但为什么 `cleaned_logs` 需要一个归约器（reducer），而它仅作为输入传入每个子图？

It is not modified.

它并未被修改。

```
cleaned_logs: Annotated[List[Log], add] # This will be USED BY in BOTH sub-graphs
```

This is because the output state of the subgraphs will contain **all keys**, even if they are unmodified.

这是因为子图的输出状态将包含**所有键**，即使它们未被修改。

The sub-graphs are run in parallel.

子图是并行运行的。

Because the parallel sub-graphs return the same key, it needs to have a reducer like `operator.add` to combine the incoming values from each sub-graph.

由于并行子图返回相同的键，因此需要一个类似 `operator.add` 的归约器，以合并各子图传入的值。

But, we can work around this by using another concept we talked about before.

但我们可以借助之前讨论过的另一个概念来规避此问题。

We can simply create an output state schema for each sub-graph and ensure that the output state schema contains different keys to publish as output.

我们可以为每个子图简单地创建一个输出状态结构，并确保该输出状态结构包含不同的键，以便作为输出发布。

We don't actually need each sub-graph to output `cleaned_logs`.

我们实际上并不需要每个子图都输出 `cleaned_logs`。



```python
# Entry Graph
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str # This will only be generated in the FA sub-graph
    report: str # This will only be generated in the QS sub-graph
    processed_logs:  Annotated[List[int], add] # This will be generated in BOTH sub-graphs

def clean_logs(state):
    # Get logs
    raw_logs = state["raw_logs"]
    # Data cleaning raw_logs -> docs 
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}

entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("clean_logs", clean_logs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

entry_builder.add_edge(START, "clean_logs")
entry_builder.add_edge("clean_logs", "failure_analysis")
entry_builder.add_edge("clean_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()

from IPython.display import Image, display

# Setting xray to 1 will show the internal structure of the nested graph
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```


    
![png](sub-graph_files/sub-graph_13_0.png)
    



```python
# Dummy logs
question_answer = Log(
    id="1",
    question="How can I import ChatOllama?",
    answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
)

question_answer_feedback = Log(
    id="2",
    question="How can I use Chroma vector store?",
    answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
    grade=0,
    grader="Document Relevance Recall",
    feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
)

raw_logs = [question_answer,question_answer_feedback]
graph.invoke({"raw_logs": raw_logs})
```




    {'raw_logs': [{'id': '1',
       'question': 'How can I import ChatOllama?',
       'answer': "To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'"},
      {'id': '2',
       'question': 'How can I use Chroma vector store?',
       'answer': 'To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).',
       'grade': 0,
       'grader': 'Document Relevance Recall',
       'feedback': 'The retrieved documents discuss vector stores in general, but not Chroma specifically'}],
     'cleaned_logs': [{'id': '1',
       'question': 'How can I import ChatOllama?',
       'answer': "To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'"},
      {'id': '2',
       'question': 'How can I use Chroma vector store?',
       'answer': 'To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).',
       'grade': 0,
       'grader': 'Document Relevance Recall',
       'feedback': 'The retrieved documents discuss vector stores in general, but not Chroma specifically'}],
     'fa_summary': 'Poor quality retrieval of Chroma documentation.',
     'report': 'foo bar baz',
     'processed_logs': ['failure-analysis-on-log-2',
      'summary-on-log-1',
      'summary-on-log-2']}




```python

```

## LangSmith

Let's look at the LangSmith trace:

我们来看 LangSmith 追踪结果：

https://smith.langchain.com/public/f8f86f61-1b30-48cf-b055-3734dfceadf2/r

https://smith.langchain.com/public/f8f86f61-1b30-48cf-b055-3734dfceadf2/r
