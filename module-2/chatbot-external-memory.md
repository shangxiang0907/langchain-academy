[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/chatbot-external-memory.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239440-lesson-6-chatbot-w-summarizing-messages-and-external-memory)

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-2/chatbot-external-memory.ipynb) [![在 LangChain 学院中打开](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/58239440-lesson-6-chatbot-w-summarizing-messages-and-external-memory)


# Chatbot with message summarization & external DB memory 具备消息摘要与外部数据库记忆的聊天机器人

## Review 回顾

We've covered how to customize graph state schema and reducer.

我们已介绍如何自定义图状态模式（state schema）与规约器（reducer）。

We've also shown a number of tricks for trimming or filtering messages in graph state.

我们还展示了多种用于修剪或过滤图状态中消息的技巧。

We've used these concepts in a Chatbot with memory that produces a running summary of the conversation.

我们已在具备记忆功能的聊天机器人中应用了这些概念，该机器人可生成对话的持续摘要。

## Goals 目标

But, what if we want our Chatbot to have memory that persists indefinitely?

但如果我们希望聊天机器人拥有永久持久化的记忆，该怎么办？

Now, we'll introduce some more advanced checkpointers that support external databases.

接下来，我们将介绍一些更高级的支持外部数据库的检查点器（checkpointer）。

Here, we'll show how to use [Sqlite as a checkpointer](https://docs.langchain.com/oss/python/langgraph/persistence#checkpointer-libraries), but other checkpointers, such as  Postgres are available!

此处，我们将演示如何使用 [Sqlite 作为检查点器](https://docs.langchain.com/oss/python/langgraph/persistence#checkpointer-libraries)，但其他检查点器（例如 Postgres）也可用！



```python
%%capture --no-stderr
%pip install --quiet -U langgraph-checkpoint-sqlite langchain_core langgraph langchain_openai
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

## Sqlite

A good starting point here is the [SqliteSaver checkpointer](https://docs.langchain.com/oss/python/langgraph/persistence#checkpointer-libraries).

一个良好的起点是 [SqliteSaver 检查点器](https://docs.langchain.com/oss/python/langgraph/persistence#checkpointer-libraries)。

Sqlite is a [small, fast, highly popular](https://x.com/karpathy/status/1819490455664685297) SQL database.

Sqlite 是一种 [轻量、快速且广受欢迎](https://x.com/karpathy/status/1819490455664685297) 的 SQL 数据库。

If we supply `":memory:"` it creates an in-memory Sqlite database.

若传入 `":memory:"`，它将创建一个内存中的 Sqlite 数据库。



```python
import sqlite3
# In memory
conn = sqlite3.connect(":memory:", check_same_thread = False)
```

But, if we supply a db path, then it will create a database for us!

但若提供数据库路径，则会自动为我们创建一个数据库！



```python
# pull file if it doesn't exist and connect to local db
!mkdir -p state_db && [ ! -f state_db/example.db ] && wget -P state_db https://github.com/langchain-ai/langchain-academy/raw/main/module-2/state_db/example.db

db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
```


```python
# Here is our checkpointer 
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver(conn)
```

Let's re-define our chatbot.

让我们重新定义聊天机器人。



```python
from typing_extensions import Literal
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from langgraph.graph import END
from langgraph.graph import MessagesState


model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "qwen-plus"), base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),temperature=0)

class State(MessagesState):
    summary: str

# Define the logic to call the model
def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    response = model.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Determine whether to end or summarize the conversation
def should_continue(state: State)-> Literal ["summarize_conversation",END]:
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END
```

Now, we just re-compile with our sqlite checkpointer.

现在，只需使用我们的 sqlite 检查点器重新编译即可。



```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
graph = workflow.compile(checkpointer=memory)
display(Image(graph.get_graph().draw_mermaid_png()))
```

Now, we can invoke the graph several times.

现在，我们可以多次调用该图。



```python
# Create a thread
config = {"configurable": {"thread_id": "1"}}

# Start conversation
input_message = HumanMessage(content="hi! I'm Lance")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="i like the 49ers!")
output = graph.invoke({"messages": [input_message]}, config) 
for m in output['messages'][-1:]:
    m.pretty_print()
```

Let's confirm that our state is saved locally.

让我们确认状态已本地保存。



```python
config = {"configurable": {"thread_id": "1"}}
graph_state = graph.get_state(config)
graph_state
```

### Persisting state 持久化状态

Using database like Sqlite means state is persisted!

使用 Sqlite 等数据库意味着状态将被持久化！

For example, we can re-start the notebook kernel and see that we can still load from Sqlite DB on disk.

例如，我们可以重启笔记本内核，并仍能从磁盘上的 Sqlite 数据库加载状态。



```python
# Create a thread
config = {"configurable": {"thread_id": "1"}}
graph_state = graph.get_state(config)
graph_state
```

## Studio

**⚠️ Notice**

**⚠️ 注意**

Since filming these videos, we've updated Studio so that it can now be run locally and accessed through your browser.

自录制这些视频以来，我们已更新 Studio，使其现在可本地运行并通过浏览器访问。

This is the preferred way to run Studio instead of using the Desktop App shown in the video.

这是运行 Studio 的首选方式，而非视频中展示的桌面应用程序。

It is now called _LangSmith Studio_ instead of _LangGraph Studio_.

它现在被称为 _LangSmith Studio_，而非 _LangGraph Studio_。

Detailed setup instructions are available in the "Getting Setup" guide at the start of the course.

详细的安装说明请参阅本课程开头的“入门设置”指南。

You can find a description of Studio [here](https://docs.langchain.com/langsmith/studio), and specific details for local deployment [here](https://docs.langchain.com/langsmith/quick-start-studio#local-development-server).

您可在此处查看 Studio 的说明：[链接](https://docs.langchain.com/langsmith/studio)，以及本地部署的具体细节：[链接](https://docs.langchain.com/langsmith/quick-start-studio#local-development-server)。

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

打开您的浏览器并导航至上方显示的 **Studio UI** URL。

Load the `chatbot` in Studio, which uses `module-2/studio/chatbot.py` set in `module-2/studio/langgraph.json`.

在 Studio 中加载 `chatbot`，其对应 `module-2/studio/chatbot.py` 文件，并由 `module-2/studio/langgraph.json` 中指定。





```python

```
