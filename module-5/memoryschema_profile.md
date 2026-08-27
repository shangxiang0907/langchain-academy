# Chatbot with Profile Schema  使用用户档案模式的聊天机器人

## Review 回顾

We introduced the [LangGraph Memory Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) as a way to save and retrieve long-term memories.

我们介绍了 [LangGraph Memory Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore)，作为一种保存和检索长期记忆的方法。

We built a simple chatbot that uses both `short-term (within-thread)` and `long-term (across-thread)` memory.

我们构建了一个简单的聊天机器人，它同时使用 `短期（线程内）` 和 `长期（跨线程）` 记忆。

It saved long-term [semantic memory](https://docs.langchain.com/oss/python/concepts/memory#semantic-memory) (facts about the user) ["in the hot path"](https://docs.langchain.com/oss/python/concepts/memory#writing-memories), as the user is chatting with it.

它在用户与其聊天时，将长期 [语义记忆](https://docs.langchain.com/oss/python/concepts/memory#semantic-memory)（关于用户的事实）[“在热路径中”](https://docs.langchain.com/oss/python/concepts/memory#writing-memories) 进行保存。

## Goals 目标

Our chatbot saved memories as a string.

我们的聊天机器人将记忆保存为字符串。

In practice, we often want memories to have a structure.

在实践中，我们通常希望记忆具有结构。

For example, memories can be a [single, continuously updated schema](https://docs.langchain.com/oss/python/concepts/memory#profile).

例如，记忆可以是 [单个、持续更新的模式](https://docs.langchain.com/oss/python/concepts/memory#profile)。

In our case, we want this to be a single user profile.

在本例中，我们希望它是一个单一的用户档案。

We'll extend our chatbot to save semantic memories to a single [user profile](https://docs.langchain.com/oss/python/concepts/memory#profile).

我们将扩展聊天机器人，以将语义记忆保存到单一的 [用户档案](https://docs.langchain.com/oss/python/concepts/memory#profile) 中。

We'll also introduce a library, [Trustcall](https://github.com/hinthornw/trustcall), to update this schema with new information.

我们还将引入一个名为 [Trustcall](https://github.com/hinthornw/trustcall) 的库，用于根据新信息更新该模式。



```python
%%capture --no-stderr
%pip install -U langchain_openai langgraph trustcall langchain_core
```


```python
import os, getpass

def _set_env(var: str):
    # Check if the variable is set in the OS environment
    env_value = os.environ.get(var)
    if not env_value:
        # If not set, prompt the user for input
        env_value = getpass.getpass(f"{var}: ")
    
    # Set the environment variable for the current process
    os.environ[var] = env_value

_set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
```

## Defining a user profile schema 定义用户档案模式

Python has many different types for [structured data](https://docs.langchain.com/oss/python/langchain/models#structured-outputs), such as TypedDict, Dictionaries, JSON, and [Pydantic](https://docs.pydantic.dev/latest/).

Python 提供了多种用于 [结构化数据](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) 的类型，例如 TypedDict、字典、JSON 和 [Pydantic](https://docs.pydantic.dev/latest/)。

Let's start by using TypedDict to define a user profile schema.

让我们首先使用 TypedDict 来定义用户档案模式。



```python
from typing import TypedDict, List

class UserProfile(TypedDict):
    """User profile schema with typed fields"""
    user_name: str  # The user's preferred name
    interests: List[str]  # A list of the user's interests
```

## Saving a schema to the store 将模式保存至存储

The  [LangGraph Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) accepts any Python dictionary as the `value`.

[LangGraph Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) 接受任意 Python 字典作为 `value`。



```python
# TypedDict instance
user_profile: UserProfile = {
    "user_name": "Lance",
    "interests": ["biking", "technology", "coffee"]
}
user_profile
```




    {'user_name': 'Lance', 'interests': ['biking', 'technology', 'coffee']}



We use the [put](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.put) method to save the TypedDict to the store.

我们使用 [put](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.put) 方法将 TypedDict 保存至存储。



```python
import uuid
from langgraph.store.memory import InMemoryStore

# Initialize the in-memory store
in_memory_store = InMemoryStore()

# Namespace for the memory to save
user_id = "1"
namespace_for_memory = (user_id, "memory")

# Save a memory to namespace as key and value
key = "user_profile"
value = user_profile
in_memory_store.put(namespace_for_memory, key, value)
```

We use [search](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.search) to retrieve objects from the store by namespace.

我们使用 [search](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.search) 方法按命名空间从存储中检索对象。



```python
# Search 
for m in in_memory_store.search(namespace_for_memory):
    print(m.dict())
```

    {'value': {'user_name': 'Lance', 'interests': ['biking', 'technology', 'coffee']}, 'key': 'user_profile', 'namespace': ['1', 'memory'], 'created_at': '2024-11-04T23:37:34.871675+00:00', 'updated_at': '2024-11-04T23:37:34.871680+00:00'}


We can also use [get](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.get) to retrieve a specific object by namespace and key.

我们还可以使用 [get](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.get) 方法按命名空间和键检索特定对象。



```python
# Get the memory by namespace and key
profile = in_memory_store.get(namespace_for_memory, "user_profile")
profile.value
```




    {'user_name': 'Lance', 'interests': ['biking', 'technology', 'coffee']}



## Chatbot with profile schema 使用用户档案模式的聊天机器人

Now we know how to specify a schema for the memories and save it to the store.

现在我们已了解如何为记忆指定模式并将其保存至存储。

Now, how do we actually *create* memories with this particular schema?

那么，我们究竟该如何 *创建* 符合该特定模式的记忆呢？

In our chatbot, we [want to create memories from a user chat](https://docs.langchain.com/oss/python/concepts/memory#profile).

在我们的聊天机器人中，我们 [希望从用户聊天中创建记忆](https://docs.langchain.com/oss/python/concepts/memory#profile)。

This is where the concept of [structured outputs](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) is useful.

此时，[结构化输出](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) 的概念便非常有用。

LangChain's [chat model](https://docs.langchain.com/oss/python/langchain/models) interface has a [`with_structured_output`](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) method to enforce structured output.

LangChain 的 [聊天模型](https://docs.langchain.com/oss/python/langchain/models) 接口提供 [`with_structured_output`](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) 方法，以强制输出符合结构。

This is useful when we want to enforce that the output conforms to a schema, and it parses the output for us.

当我们希望确保输出符合某一模式，并由系统自动解析输出时，该方法非常有用。



```python
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))
_set_env("OPENAI_API_KEY")
```

Let's pass the `UserProfile` schema we created to the `with_structured_output` method.

让我们将之前创建的 `UserProfile` 模式传入 `with_structured_output` 方法。

We can then invoke the chat model with a list of [messages](https://docs.langchain.com/oss/python/langchain/messages) and get a structured output that conforms to our schema.

然后，我们可以使用一组 [消息](https://docs.langchain.com/oss/python/langchain/messages) 调用聊天模型，并获得符合我们模式的结构化输出。



```python
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
import os
from langchain_openai import ChatOpenAI

# Initialize the model
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "qwen-plus"), base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), temperature=0)

# Bind schema to model
model_with_structure = model.with_structured_output(UserProfile)

# Invoke the model to produce structured output that matches the schema
structured_output = model_with_structure.invoke([HumanMessage("My name is Lance, I like to bike.")])
structured_output
```




    {'user_name': 'Lance', 'interests': ['biking']}



Now, let's use this with our chatbot.

现在，让我们将此功能应用于我们的聊天机器人。

This only requires minor changes to the `write_memory` function.

这仅需对 `write_memory` 函数做少量修改。

We use `model_with_structure`, as defined above, to produce a profile that matches our schema.

我们使用上方定义的 `model_with_structure` 来生成符合我们模式的档案。



```python
from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_INSTRUCTION = """Create or update a user profile memory based on the user's chat history. 
This will be saved for long-term memory. If there is an existing memory, simply update it. 
Here is the existing memory (it may be empty): {memory}"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memory from the store and use it to personalize the chatbot's response."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Format the memories for the system prompt
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"
        )
    else:
        formatted_memory = None

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

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

    # Format the memories for the system prompt
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"
        )
    else:
        formatted_memory = None
        
    # Format the existing memory in the instruction
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=formatted_memory)

    # Invoke the model to produce structured output that matches the schema
    new_memory = model_with_structure.invoke([SystemMessage(content=system_msg)]+state['messages'])

    # Overwrite the existing use profile memory
    key = "user_memory"
    store.put(namespace, key, new_memory)

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


    
![jpeg](memoryschema_profile_files/memoryschema_profile_18_0.jpg)
    



```python
# We supply a thread ID for short-term (within-thread) memory
# We supply a user ID for long-term (across-thread) memory 
config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# User input 
input_messages = [HumanMessage(content="Hi, my name is Lance and I like to bike around San Francisco and eat at bakeries.")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Hi, my name is Lance and I like to bike around San Francisco and eat at bakeries.
    ==================================[1m Ai Message [0m==================================
    
    Hi Lance! It's great to meet you. Biking around San Francisco sounds like a fantastic way to explore the city, and there are so many amazing bakeries to try. Do you have any favorite bakeries or biking routes in the city?


Let's check the memory in the store.

让我们检查存储中的记忆。

We can see that the memory is a dictionary that matches our schema.

我们可以看到，该记忆是一个符合我们模式的字典。



```python
# Namespace for the memory to save
user_id = "1"
namespace = ("memory", user_id)
existing_memory = across_thread_memory.get(namespace, "user_memory")
existing_memory.value
```




    {'user_name': 'Lance', 'interests': ['biking', 'bakeries', 'San Francisco']}



## When can this fail? 何时会失败？

[`with_structured_output`](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) is very useful, but what happens if we're working with a more complex schema?

[`with_structured_output`](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) 非常有用，但若处理更复杂的模式，会发生什么情况？

[Here's](https://github.com/hinthornw/trustcall?tab=readme-ov-file#complex-schema) an example of a more complex schema, which we'll test below.

[此处](https://github.com/hinthornw/trustcall?tab=readme-ov-file#complex-schema) 展示了一个更复杂模式的示例，我们将在下方测试该示例。

This is a [Pydantic](https://docs.pydantic.dev/latest/) model that describes a user's preferences for communication and trust fall.

这是一个 [Pydantic](https://docs.pydantic.dev/latest/) 模型，用于描述用户在沟通与信任崩塌方面的偏好。



```python
from typing import List, Optional

class OutputFormat(BaseModel):
    preference: str
    sentence_preference_revealed: str

class TelegramPreferences(BaseModel):
    preferred_encoding: Optional[List[OutputFormat]] = None
    favorite_telegram_operators: Optional[List[OutputFormat]] = None
    preferred_telegram_paper: Optional[List[OutputFormat]] = None

class MorseCode(BaseModel):
    preferred_key_type: Optional[List[OutputFormat]] = None
    favorite_morse_abbreviations: Optional[List[OutputFormat]] = None

class Semaphore(BaseModel):
    preferred_flag_color: Optional[List[OutputFormat]] = None
    semaphore_skill_level: Optional[List[OutputFormat]] = None

class TrustFallPreferences(BaseModel):
    preferred_fall_height: Optional[List[OutputFormat]] = None
    trust_level: Optional[List[OutputFormat]] = None
    preferred_catching_technique: Optional[List[OutputFormat]] = None

class CommunicationPreferences(BaseModel):
    telegram: TelegramPreferences
    morse_code: MorseCode
    semaphore: Semaphore

class UserPreferences(BaseModel):
    communication_preferences: CommunicationPreferences
    trust_fall_preferences: TrustFallPreferences

class TelegramAndTrustFallPreferences(BaseModel):
    pertinent_user_preferences: UserPreferences
```

Now, let's try extraction of this schema using the `with_structured_output` method.

现在，让我们尝试使用 `with_structured_output` 方法提取该模式。



```python
from pydantic import ValidationError

# Bind schema to model
model_with_structure = model.with_structured_output(TelegramAndTrustFallPreferences)

# Conversation
conversation = """Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir."""

# Invoke the model
try:
    model_with_structure.invoke(f"""Extract the preferences from the following conversation:
    <convo>
    {conversation}
    </convo>""")
except ValidationError as e:
    print(e)
```

    1 validation error for TelegramAndTrustFallPreferences
    pertinent_user_preferences.communication_preferences.semaphore
      Input should be a valid dictionary or instance of Semaphore [type=model_type, input_value=None, input_type=NoneType]
        For further information visit https://errors.pydantic.dev/2.9/v/model_type


If we naively extract more complex schemas, even using high capacity model like `gpt-4o`, it is prone to failure.

即使使用 `gpt-4o` 等高容量模型，若直接提取更复杂的模式，也极易失败。


## Trustcall for creating and updating profile schemas 用于创建和更新档案模式的 Trustcall

As we can see, working with schemas can be tricky.

如我们所见，处理模式可能颇具挑战性。

Complex schemas can be difficult to extract.

复杂模式难以准确提取。

In addition, updating even simple schemas can pose challenges.

此外，即使是简单模式的更新也可能带来挑战。

Consider our above chatbot.

以我们上面的聊天机器人为例。

We regenerated the profile schema *from scratch* each time we chose to save a new memory.

每次选择保存新记忆时，我们都需从头重新生成整个档案模式。

This is inefficient, potentially wasting model tokens if the schema contains a lot of information to re-generate each time.

这种方式效率低下，若模式包含大量信息，则每次重新生成都可能浪费模型 token。

Worse, we may loose information when regenerating the profile from scratch.

更严重的是，在从头重新生成档案时，我们可能会丢失信息。

Addressing these problems is the motivation for [TrustCall](https://github.com/hinthornw/trustcall)!

解决上述问题，正是 [TrustCall](https://github.com/hinthornw/trustcall) 的设计初衷！

This is an open-source library for updating JSON schemas developed by one [Will Fu-Hinthorn](https://github.com/hinthornw) on the LangChain team.

这是一个开源库，由 LangChain 团队成员 [Will Fu-Hinthorn](https://github.com/hinthornw) 开发，专用于更新 JSON 模式。

It's motivated by exactly these challenges while working on memory.

其开发动机正是我们在处理记忆时所面临的这些挑战。

Let's first show simple usage of extraction with TrustCall on this list of [messages](https://docs.langchain.com/oss/python/langchain/messages).

我们首先展示 TrustCall 在该组 [消息](https://docs.langchain.com/oss/python/langchain/messages) 上进行提取的简单用法。



```python
# Conversation
conversation = [HumanMessage(content="Hi, I'm Lance."), 
                AIMessage(content="Nice to meet you, Lance."), 
                HumanMessage(content="I really like biking around San Francisco.")]
```

We use `create_extractor`, passing in the model as well as our schema as a [tool](https://docs.langchain.com/oss/python/langchain/tools).

我们使用 `create_extractor`，传入模型以及作为 [工具](https://docs.langchain.com/oss/python/langchain/tools) 的模式。

With TrustCall, can supply supply the schema in various ways.

借助 TrustCall，我们可以以多种方式提供模式。

For example, we can pass a JSON object / Python dictionary or Pydantic model.

例如，我们可以传入 JSON 对象 / Python 字典或 Pydantic 模型。

Under the hood, TrustCall uses [tool calling](https://docs.langchain.com/oss/python/langchain/models#tool-calling) to produce [structured output](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) from an input list of [messages](https://docs.langchain.com/oss/python/langchain/messages).

TrustCall 底层使用 [工具调用](https://docs.langchain.com/oss/python/langchain/models#tool-calling)，从输入的 [消息](https://docs.langchain.com/oss/python/langchain/messages) 列表中生成 [结构化输出](https://docs.langchain.com/oss/python/langchain/models#structured-outputs)。

To force Trustcall to produce structured output, we can include the schema name in the `tool_choice` argument.

为强制 Trustcall 生成结构化输出，我们可在 `tool_choice` 参数中包含模式名称。

We can invoke the extractor with  the above conversation.

我们可以使用上述对话调用该提取器。



```python
from trustcall import create_extractor

# Schema 
class UserProfile(BaseModel):
    """User profile schema with typed fields"""
    user_name: str = Field(description="The user's preferred name")
    interests: List[str] = Field(description="A list of the user's interests")

# Initialize the model
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "qwen-plus"), base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), temperature=0)

# Create the extractor
trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile"
)

# Instruction
system_msg = "Extract the user profile from the following conversation"

# Invoke the extractor
result = trustcall_extractor.invoke({"messages": [SystemMessage(content=system_msg)]+conversation})
```

When we invoke the extractor, we get a few things:

当我们调用提取器时，会得到以下几项内容：

* `messages`: The list of `AIMessages` that contain the tool calls. 
  - `messages`：包含工具调用的 `AIMessages` 列表。

* `responses`: The resulting parsed tool calls that match our schema.
  - `responses`：匹配我们模式的已解析工具调用结果。

* `response_metadata`: Applicable if updating existing tool calls. It says which of the responses correspond to which of the existing objects.
  - `response_metadata`：仅在更新现有工具调用时适用，用于说明各响应分别对应哪些现有对象。



```python
for m in result["messages"]: 
    m.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      UserProfile (call_spGGUsoaUFXU7oOrUNCASzfL)
     Call ID: call_spGGUsoaUFXU7oOrUNCASzfL
      Args:
        user_name: Lance
        interests: ['biking around San Francisco']



```python
schema = result["responses"]
schema
```




    [UserProfile(user_name='Lance', interests=['biking around San Francisco'])]




```python
schema[0].model_dump()
```




    {'user_name': 'Lance', 'interests': ['biking around San Francisco']}




```python
result["response_metadata"]
```




    [{'id': 'call_spGGUsoaUFXU7oOrUNCASzfL'}]



Let's see how we can use it to *update* the profile.

让我们看看如何使用它来*更新*个人资料。

For updating, TrustCall takes a set of messages as well as the existing schema.

对于更新操作，TrustCall 接收一组消息以及现有模式。

The central idea is that it prompts the model to produce a [JSON Patch](https://jsonpatch.com/) to update only the relevant parts of the schema.

其核心思想是提示模型生成一个 [JSON Patch](https://jsonpatch.com/)，仅更新模式中相关部分。

This is less error-prone than naively overwriting the entire schema.

这比简单地覆盖整个模式更不易出错。

It's also more efficient since the model only needs to generate the parts of the schema that have changed.

同时这也更高效，因为模型只需生成模式中已发生变化的部分。

We can save the existing schema as a dict.

我们可以将现有模式保存为字典（dict）。

We can use `model_dump()` to serialize a Pydantic model instance into a dict.

可使用 `model_dump()` 将 Pydantic 模型实例序列化为字典。

We pass it to the `"existing"` argument along with the schema name, `UserProfile`.

我们将该字典连同模式名称 `UserProfile` 一起传入 `"existing"` 参数。



```python
# Update the conversation
updated_conversation = [HumanMessage(content="Hi, I'm Lance."), 
                        AIMessage(content="Nice to meet you, Lance."), 
                        HumanMessage(content="I really like biking around San Francisco."),
                        AIMessage(content="San Francisco is a great city! Where do you go after biking?"),
                        HumanMessage(content="I really like to go to a bakery after biking."),]

# Update the instruction
system_msg = f"""Update the memory (JSON doc) to incorporate new information from the following conversation"""

# Invoke the extractor with the updated instruction and existing profile with the corresponding tool name (UserProfile)
result = trustcall_extractor.invoke({"messages": [SystemMessage(content=system_msg)]+updated_conversation}, 
                                    {"existing": {"UserProfile": schema[0].model_dump()}})  
```


```python
for m in result["messages"]: 
    m.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      UserProfile (call_WeZl0ACfQStxblim0ps8LNKT)
     Call ID: call_WeZl0ACfQStxblim0ps8LNKT
      Args:
        user_name: Lance
        interests: ['biking', 'visiting bakeries']



```python
result["response_metadata"]
```




    [{'id': 'call_WeZl0ACfQStxblim0ps8LNKT'}]




```python
updated_schema = result["responses"][0]
updated_schema.model_dump()
```




    {'user_name': 'Lance', 'interests': ['biking', 'visiting bakeries']}



LangSmith trace:

LangSmith 追踪记录：

https://smith.langchain.com/public/229eae22-1edb-44c6-93e6-489124a43968/r

https://smith.langchain.com/public/229eae22-1edb-44c6-93e6-489124a43968/r

Now, let's also test Trustcall on the [challenging schema](https://github.com/hinthornw/trustcall?tab=readme-ov-file#complex-schema) that we saw earlier.

接下来，我们也在之前看到的 [复杂模式](https://github.com/hinthornw/trustcall?tab=readme-ov-file#complex-schema) 上测试 TrustCall。



```python
bound = create_extractor(
    model,
    tools=[TelegramAndTrustFallPreferences],
    tool_choice="TelegramAndTrustFallPreferences",
)

# Conversation
conversation = """Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir."""

result = bound.invoke(
    f"""Extract the preferences from the following conversation:
<convo>
{conversation}
</convo>"""
)

# Extract the preferences
result["responses"][0]
```




    TelegramAndTrustFallPreferences(pertinent_user_preferences=UserPreferences(communication_preferences=CommunicationPreferences(telegram=TelegramPreferences(preferred_encoding=[OutputFormat(preference='standard encoding', sentence_preference_revealed='standard encoding')], favorite_telegram_operators=None, preferred_telegram_paper=[OutputFormat(preference='Daredevil', sentence_preference_revealed='Daredevil')]), morse_code=MorseCode(preferred_key_type=[OutputFormat(preference='straight key', sentence_preference_revealed='straight key')], favorite_morse_abbreviations=None), semaphore=Semaphore(preferred_flag_color=None, semaphore_skill_level=None)), trust_fall_preferences=TrustFallPreferences(preferred_fall_height=[OutputFormat(preference='higher', sentence_preference_revealed='higher')], trust_level=None, preferred_catching_technique=[OutputFormat(preference='diamond formation', sentence_preference_revealed='diamond formation')])))



Trace:

追踪记录：

https://smith.langchain.com/public/5cd23009-3e05-4b00-99f0-c66ee3edd06e/r

https://smith.langchain.com/public/5cd23009-3e05-4b00-99f0-c66ee3edd06e/r

For more examples, you can see an overview video [here](https://www.youtube.com/watch?v=-H4s0jQi-QY).

更多示例请参阅此处的概览视频：[链接](https://www.youtube.com/watch?v=-H4s0jQi-QY)。


## Chatbot with profile schema updating 支持个人资料模式更新的聊天机器人

Now, let's bring Trustcall into our chatbot to create *and update* a memory profile.

现在，让我们将 TrustCall 集成到我们的聊天机器人中，以实现个人资料记忆的*创建与更新*。



```python
from IPython.display import Image, display

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

# Initialize the model
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "qwen-plus"), base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), temperature=0)

# Schema 
class UserProfile(BaseModel):
    """ Profile of a user """
    user_name: str = Field(description="The user's preferred name")
    user_location: str = Field(description="The user's location")
    interests: list = Field(description="A list of the user's interests")

# Create the extractor
trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile", # Enforces use of the UserProfile tool
)

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Extraction instruction
TRUSTCALL_INSTRUCTION = """Create or update the memory (JSON doc) to incorporate information from the following conversation:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """Load memory from the store and use it to personalize the chatbot's response."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Format the memories for the system prompt
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Location: {memory_dict.get('user_location', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"      
        )
    else:
        formatted_memory = None

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

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
        
    # Get the profile as the value from the list, and convert it to a JSON doc
    existing_profile = {"UserProfile": existing_memory.value} if existing_memory else None
    
    # Invoke the extractor
    result = trustcall_extractor.invoke({"messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)]+state["messages"], "existing": existing_profile})
    
    # Get the updated profile as a JSON object
    updated_profile = result["responses"][0].model_dump()

    # Save the updated profile
    key = "user_memory"
    store.put(namespace, key, updated_profile)

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


    
![jpeg](memoryschema_profile_files/memoryschema_profile_45_0.jpg)
    



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
    
    That sounds like a great way to explore the city! San Francisco has some beautiful routes and views. Do you have any favorite trails or spots you like to visit while biking?



```python
# Namespace for the memory to save
user_id = "1"
namespace = ("memory", user_id)
existing_memory = across_thread_memory.get(namespace, "user_memory")
existing_memory.dict()
```




    {'value': {'user_name': 'Lance',
      'user_location': 'San Francisco',
      'interests': ['biking']},
     'key': 'user_memory',
     'namespace': ['memory', '1'],
     'created_at': '2024-11-04T23:51:17.662428+00:00',
     'updated_at': '2024-11-04T23:51:41.697652+00:00'}




```python
# The user profile saved as a JSON object
existing_memory.value
```




    {'user_name': 'Lance',
     'user_location': 'San Francisco',
     'interests': ['biking']}




```python
# User input 
input_messages = [HumanMessage(content="I also enjoy going to bakeries")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    I also enjoy going to bakeries
    ==================================[1m Ai Message [0m==================================
    
    Biking and visiting bakeries sounds like a delightful combination! San Francisco has some fantastic bakeries. Do you have any favorites, or are you looking for new recommendations to try out?


Continue the conversation in a new thread.

在新线程中继续对话。



```python
# We supply a thread ID for short-term (within-thread) memory
# We supply a user ID for long-term (across-thread) memory 
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

# User input 
input_messages = [HumanMessage(content="What bakeries do you recommend for me?")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    What bakeries do you recommend for me?
    ==================================[1m Ai Message [0m==================================
    
    Since you're in San Francisco and enjoy going to bakeries, here are a few recommendations you might like:
    
    1. **Tartine Bakery** - Known for its delicious bread and pastries, it's a must-visit for any bakery enthusiast.
    2. **B. Patisserie** - Offers a delightful selection of French pastries, including their famous kouign-amann.
    3. **Arsicault Bakery** - Renowned for its croissants, which have been praised as some of the best in the country.
    4. **Craftsman and Wolves** - Known for their inventive pastries and the "Rebel Within," a savory muffin with a soft-cooked egg inside.
    5. **Mr. Holmes Bakehouse** - Famous for their cruffins and other creative pastries.
    
    These spots should offer a great variety of treats for you to enjoy. Happy bakery hopping!


Trace:

追踪记录：

https://smith.langchain.com/public/f45bdaf0-6963-4c19-8ec9-f4b7fe0f68ad/r

https://smith.langchain.com/public/f45bdaf0-6963-4c19-8ec9-f4b7fe0f68ad/r

## Studio

![Screenshot 2024-10-30 at 11.26.31 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6732d0437060f1754ea79908_Screenshot%202024-11-11%20at%207.48.53%E2%80%AFPM.png)



```python

```
