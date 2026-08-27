# Chatbot with Memory 带记忆功能的聊天机器人

## Review 复习

[Memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC10410470/) is a cognitive function that allows people to store, retrieve, and use information to understand their present and future.

[记忆](https://pmc.ncbi.nlm.nih.gov/articles/PMC10410470/) 是一种认知功能，使人们能够存储、检索和使用信息，以理解当下与未来。

There are [various long-term memory types](https://docs.langchain.com/oss/python/concepts/memory#memory-types) that can be used in AI applications.

存在多种[长期记忆类型](https://docs.langchain.com/oss/python/concepts/memory#memory-types)，可用于人工智能应用。

## Goals 目标

Here, we'll introduce the [LangGraph Memory Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) as a way to save and retrieve long-term memories.

此处，我们将介绍 [LangGraph Memory Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore)，作为一种保存与检索长期记忆的方式。

We'll build a chatbot that uses both `short-term (within-thread)` and `long-term (across-thread)` memory.

我们将构建一个同时使用 `短期（线程内）` 和 `长期（跨线程）` 记忆的聊天机器人。

We'll focus on long-term [semantic memory](https://docs.langchain.com/oss/python/concepts/memory#semantic-memory), which will be facts about the user.

我们将重点关注长期[语义记忆](https://docs.langchain.com/oss/python/concepts/memory#semantic-memory)，即关于用户的事实性信息。

These long-term memories will be used to create a personalized chatbot that can remember facts about the user.

这些长期记忆将用于创建一个个性化聊天机器人，使其能够记住有关用户的事实。

It will save memory ["in the hot path"](https://docs.langchain.com/oss/python/concepts/memory#writing-memories), as the user is chatting with it.

它将在用户与其聊天时，[“在热路径中”](https://docs.langchain.com/oss/python/concepts/memory#writing-memories) 保存记忆。



```python
%%capture --no-stderr
%pip install -U langchain_openai langgraph langchain_core
```

We'll use [LangSmith](https://docs.langchain.com/langsmith/home) for [tracing](https://docs.langchain.com/langsmith/observability-concepts).

我们将使用 [LangSmith](https://docs.langchain.com/langsmith/home) 进行[追踪](https://docs.langchain.com/langsmith/observability-concepts)。



```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

## Introduction to the LangGraph Store LangGraph Store 简介

The  [LangGraph Memory Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) provides a way to store and retrieve information *across threads* in LangGraph.

[LangGraph Memory Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) 提供了一种在 LangGraph 中*跨线程*存储与检索信息的方式。

This is an  [open source base class](https://blog.langchain.com/launching-long-term-memory-support-in-langgraph/) for persistent `key-value` stores.

这是一个[开源基类](https://blog.langchain.com/launching-long-term-memory-support-in-langgraph/)，用于持久化的 `键值（key-value）` 存储。



```python
import uuid
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
```

When storing objects (e.g., memories) in the [Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore), we provide:

向 [Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) 中存储对象（例如记忆）时，我们需要提供：

- The `namespace` for the object, a tuple (similar to directories)
  - 对象的 `命名空间（namespace）`，为一个元组（类似于目录结构）

- the object `key` (similar to filenames)
  - 对象的 `键（key）`（类似于文件名）

- the object `value` (similar to file contents)
  - 对象的 `值（value）`（类似于文件内容）

We use the [put](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.put) method to save an object to the store by `namespace` and `key`.

我们使用 [put](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.put) 方法，按 `命名空间` 和 `键` 将对象保存至存储中。

![langgraph_store.png](memory_store_files/6281b4e3-4930-467e-83ce-ba1aa837ca16.png)



```python
# Namespace for the memory to save
user_id = "1"
namespace_for_memory = (user_id, "memories")

# Save a memory to namespace as key and value
key = str(uuid.uuid4())

# The value needs to be a dictionary  
value = {"food_preference" : "I like pizza"}

# Save the memory
in_memory_store.put(namespace_for_memory, key, value)
```

We use [search](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.search) to retrieve objects from the store by `namespace`.

我们使用 [search](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.search) 方法，按 `命名空间` 从存储中检索对象。

This returns a list.

该方法返回一个列表。



```python
# Search 
memories = in_memory_store.search(namespace_for_memory)
type(memories)
```




    list




```python
# Metatdata 
memories[0].dict()
```




    {'value': {'food_preference': 'I like pizza'},
     'key': 'a754b8c5-e8b7-40ec-834b-c426a9a7c7cc',
     'namespace': ['1', 'memories'],
     'created_at': '2024-11-04T22:48:16.727572+00:00',
     'updated_at': '2024-11-04T22:48:16.727574+00:00'}




```python
# The key, value
print(memories[0].key, memories[0].value)
```

    a754b8c5-e8b7-40ec-834b-c426a9a7c7cc {'food_preference': 'I like pizza'}


We can also use [get](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.get) to retrieve an object by `namespace` and `key`.

我们还可以使用 [get](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.get) 方法，按 `命名空间` 和 `键` 检索单个对象。



```python
# Get the memory by namespace and key
memory = in_memory_store.get(namespace_for_memory, key)
memory.dict()
```




    {'value': {'food_preference': 'I like pizza'},
     'key': 'a754b8c5-e8b7-40ec-834b-c426a9a7c7cc',
     'namespace': ['1', 'memories'],
     'created_at': '2024-11-04T22:48:16.727572+00:00',
     'updated_at': '2024-11-04T22:48:16.727574+00:00'}



## Chatbot with long-term memory 带长期记忆的聊天机器人

We want a chatbot that [has two types of memory](https://docs.google.com/presentation/d/181mvjlgsnxudQI6S3ritg9sooNyu4AcLLFH1UK0kIuk/edit#slide=id.g30eb3c8cf10_0_156):

我们希望构建一个[具备两种记忆类型](https://docs.google.com/presentation/d/181mvjlgsnxudQI6S3ritg9sooNyu4AcLLFH1UK0kIuk/edit#slide=id.g30eb3c8cf10_0_156)的聊天机器人：

1. `Short-term (within-thread) memory`: Chatbot can persist conversational history and / or allow interruptions in a chat session.
  - `短期（线程内）记忆`：聊天机器人可在一次聊天会话中持续保存对话历史，或支持中断操作。

2. `Long-term (cross-thread) memory`: Chatbot can remember information about a specific user *across all chat sessions*.
  - `长期（跨线程）记忆`：聊天机器人可跨所有聊天会话，记住特定用户的有关信息。



```python
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))
_set_env("OPENAI_API_KEY")
```

For `short-term memory`, we'll use a [checkpointer](https://docs.langchain.com/oss/python/langgraph/persistence#checkpointer-libraries).

对于 `短期记忆`，我们将使用 [检查点器（checkpointer）](https://docs.langchain.com/oss/python/langgraph/persistence#checkpointer-libraries)。

See Module 2 and our [conceptual docs](https://docs.langchain.com/oss/python/langgraph/persistence) for more on checkpointers, but in summary:

更多关于检查点器的内容，请参阅模块 2 及我们的[概念文档](https://docs.langchain.com/oss/python/langgraph/persistence)，简而言之：

* They write the graph state at each step to a thread.
  - 它们在每一步都将图状态写入线程。

* They persist the chat history in the thread.
  - 它们在线程中持久化聊天历史。

* They allow the graph to be interrupted and / or resumed from any step in the thread.
  - 它们允许图从线程中的任意步骤被中断和/或恢复。

And, for `long-term memory`, we'll use the [LangGraph Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) as introduced above.

而对于 `长期记忆`，我们将使用上文介绍的 [LangGraph Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore)。



```python
# Chat model 
import os
from langchain_openai import ChatOpenAI

# Initialize the LLM
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "qwen-plus"), base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), temperature=0) 
```

The chat history will be saved to short-term memory using the checkpointer.

聊天历史将通过检查点器保存至短期记忆。

The chatbot will reflect on the chat history.

聊天机器人将对聊天历史进行反思。

It will then create and save a memory to the [LangGraph Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore).

然后，它将创建一条记忆并保存至 [LangGraph Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore)。

This memory is accessible in future chat sessions to personalize the chatbot's responses.

该记忆可在未来的聊天会话中被访问，从而实现聊天机器人响应的个性化。



```python
from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_INSTRUCTION = """"You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memory from the store and use it to personalize the chatbot's response."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    # Extract the actual memory content if it exists and add a prefix
    if existing_memory:
        # Value is a dictionary with a memory key
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)
    
    # Respond using memory as well as the chat history
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Reflect on the chat history and save a memory to the store."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")
        
    # Extract the memory
    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # Format the memory in the system prompt
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'])

    # Overwrite the existing memory in the store 
    key = "user_memory"

    # Write value as a dictionary with a memory key
    store.put(namespace, key, {"memory": new_memory.content})

# Define the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)

# Store for long-term (across-thread) memory
across_thread_memory = InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = MemorySaver()

# Compile the graph with the checkpointer fir and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

# View
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```


    
![jpeg](memory_store_files/memory_store_19_0.jpg)
    


When we interact with the chatbot, we supply two things:

当我们与聊天机器人交互时，需提供两样东西：

1. `Short-term (within-thread) memory`: A `thread ID` for persisting the chat history.
  - `短期（线程内）记忆`：一个用于持久化聊天历史的 `线程 ID`。

2. `Long-term (cross-thread) memory`: A `user ID` to namespace long-term memories to the user.
  - `长期（跨线程）记忆`：一个用于将长期记忆按用户命名空间隔离的 `用户 ID`。

Let's see how these work together in practice.

让我们在实践中看看它们如何协同工作。



```python
# We supply a thread ID for short-term (within-thread) memory
# We supply a user ID for long-term (across-thread) memory 
config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# User input 
input_messages = [HumanMessage(content="Hi, my name is Lance")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Hi, my name is Lance
    ==================================[1m Ai Message [0m==================================
    
    Hello, Lance! It's nice to meet you. How can I assist you today?



```python
# User input 
input_messages = [HumanMessage(content="I like to bike around San Francisco")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    I like to bike around San Francisco
    ==================================[1m Ai Message [0m==================================
    
    That sounds like a great way to explore the city, Lance! San Francisco has some beautiful routes and views. Do you have a favorite trail or area you like to bike in?


We're using the `MemorySaver` checkpointer for within-thread memory.

我们正在使用 `MemorySaver` 检查点器来处理线程内记忆。

This saves the chat history to the thread.

该检查点器将聊天历史保存至线程。

We can look at the chat history saved to the thread.

我们可以查看已保存至线程的聊天历史。



```python
thread = {"configurable": {"thread_id": "1"}}
state = graph.get_state(thread).values
for m in state["messages"]: 
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Hi, my name is Lance
    ==================================[1m Ai Message [0m==================================
    
    Hello, Lance! It's nice to meet you. How can I assist you today?
    ================================[1m Human Message [0m=================================
    
    I like to bike around San Francisco
    ==================================[1m Ai Message [0m==================================
    
    That sounds like a great way to explore the city, Lance! San Francisco has some beautiful routes and views. Do you have a favorite trail or area you like to bike in?


Recall that we compiled the graph with our the store:

请回顾一下，我们已使用该存储编译了图：

```python
across_thread_memory = InMemoryStore()
```

And, we added a node to the graph (`write_memory`) that reflects on the chat history and saves a memory to the store.

并且，我们在图中添加了一个节点（`write_memory`），用于反思聊天历史并将记忆保存至存储。

We can to see if the memory was saved to the store.

我们可以验证该记忆是否已成功保存至存储。



```python
# Namespace for the memory to save
user_id = "1"
namespace = ("memory", user_id)
existing_memory = across_thread_memory.get(namespace, "user_memory")
existing_memory.dict()
```




    {'value': {'memory': "**Updated User Information:**\n- User's name is Lance.\n- Likes to bike around San Francisco."},
     'key': 'user_memory',
     'namespace': ['memory', '1'],
     'created_at': '2024-11-05T00:12:17.383918+00:00',
     'updated_at': '2024-11-05T00:12:25.469528+00:00'}



Now, let's kick off a *new thread* with the *same user ID*.

现在，让我们用*相同的用户 ID*启动一个*新线程*。

We should see that the chatbot remembered the user's profile and used it to personalize the response.

我们应该能看到聊天机器人记住了用户的档案，并据此个性化其响应。



```python
# We supply a user ID for across-thread memory as well as a new thread ID
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

# User input 
input_messages = [HumanMessage(content="Hi! Where would you recommend that I go biking?")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Hi! Where would you recommend that I go biking?
    ==================================[1m Ai Message [0m==================================
    
    Hi Lance! Since you enjoy biking around San Francisco, there are some fantastic routes you might love. Here are a few recommendations:
    
    1. **Golden Gate Park**: This is a classic choice with plenty of trails and beautiful scenery. You can explore the park's many attractions, like the Conservatory of Flowers and the Japanese Tea Garden.
    
    2. **The Embarcadero**: A ride along the Embarcadero offers stunning views of the Bay Bridge and the waterfront. It's a great way to experience the city's vibrant atmosphere.
    
    3. **Marin Headlands**: If you're up for a bit of a challenge, biking across the Golden Gate Bridge to the Marin Headlands offers breathtaking views of the city and the Pacific Ocean.
    
    4. **Presidio**: This area has a network of trails with varying difficulty levels, and you can enjoy views of the Golden Gate Bridge and the bay.
    
    5. **Twin Peaks**: For a more challenging ride, head up to Twin Peaks. The climb is worth it for the panoramic views of the city.
    
    Let me know if you want more details on any of these routes!



```python
# User input 
input_messages = [HumanMessage(content="Great, are there any bakeries nearby that I can check out? I like a croissant after biking.")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Great, are there any bakeries nearby that I can check out? I like a croissant after biking.
    ==================================[1m Ai Message [0m==================================
    
    Absolutely, Lance! Here are a few bakeries in San Francisco where you can enjoy a delicious croissant after your ride:
    
    1. **Tartine Bakery**: Located in the Mission District, Tartine is famous for its pastries, and their croissants are a must-try.
    
    2. **Arsicault Bakery**: This bakery in the Richmond District has been praised for its buttery, flaky croissants. It's a bit of a detour, but worth it!
    
    3. **b. Patisserie**: Situated in Lower Pacific Heights, b. Patisserie offers a variety of pastries, and their croissants are particularly popular.
    
    4. **Le Marais Bakery**: With locations in the Marina and Castro, Le Marais offers a charming French bakery experience with excellent croissants.
    
    5. **Neighbor Bakehouse**: Located in the Dogpatch, this bakery is known for its creative pastries, including some fantastic croissants.
    
    These spots should provide a delightful treat after your biking adventures. Enjoy your ride and your croissant!


## Viewing traces in LangSmith 在 LangSmith 中查看追踪记录

We can see that the memories are retrieved from the store and supplied as part of the system prompt, as expected:

我们可以看到，记忆已按预期从存储中检索，并作为系统提示的一部分提供：

https://smith.langchain.com/public/10268d64-82ff-434e-ac02-4afa5cc15432/r

https://smith.langchain.com/public/10268d64-82ff-434e-ac02-4afa5cc15432/r

## Studio

We can also interact with our chatbot in Studio.

我们还可以在 Studio 中与我们的聊天机器人进行交互。

![Screenshot 2024-10-28 at 10.08.27 AM.png](memory_store_files/afa216f7-4b67-4783-82af-c319e0f512ac.png)
