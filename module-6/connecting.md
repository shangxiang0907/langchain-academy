# Connecting to a LangGraph Platform Deployment 连接到 LangGraph 平台部署

## Deployment Creation 部署创建

We just created a <!--[~deployment~](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-to-do-a-self-hosted-deployment-of-langgraph) --> [deployment](https://docs.langchain.com/langsmith/self_hosted_data_plane) for the `task_maistro` app from module 5.

我们刚刚为模块 5 中的 `task_maistro` 应用创建了一个 <!--[~deployment~](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-to-do-a-self-hosted-deployment-of-langgraph) --> [部署](https://docs.langchain.com/langsmith/self_hosted_data_plane)。

* We used the [the LangGraph CLI](https://docs.langchain.com/langsmith/cli) to build a Docker image for the LangGraph Server with our `task_maistro` graph.
  - 我们使用 [LangGraph CLI](https://docs.langchain.com/langsmith/cli) 为 LangGraph Server 构建了包含 `task_maistro` 图的 Docker 镜像。

* We used the provided `docker-compose.yml` file to create three separate containers based on the services defined: 
  - 我们使用提供的 `docker-compose.yml` 文件，基于所定义的服务创建了三个独立容器：

    * `langgraph-redis`: Creates a new container using the official Redis image.
      - `langgraph-redis`：使用官方 Redis 镜像创建一个新容器。

    * `langgraph-postgres`: Creates a new container using the official Postgres image.
      - `langgraph-postgres`：使用官方 Postgres 镜像创建一个新容器。

    * `langgraph-api`: Creates a new container using our pre-built `task_maistro` Docker image.
      - `langgraph-api`：使用我们预先构建的 `task_maistro` Docker 镜像创建一个新容器。

```
$ cd module-6/deployment
$ docker compose up
```

Once running, we can access the deployment through:

运行后，我们可通过以下方式访问该部署：

* API: http://localhost:8123
  - API：http://localhost:8123

* Docs: http://localhost:8123/docs
  - 文档：http://localhost:8123/docs

* LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123
  - LangGraph Studio：https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123

![langgraph-platform-high-level.png](connecting_files/3a5ede4f-7a62-4e05-9e44-301465ca8555.png)

## Using the API   使用 API

LangGraph Server exposes [many API endpoints](https://github.com/langchain-ai/agent-protocol) for interacting with the deployed agent.

LangGraph Server 暴露了 [多个 API 端点](https://github.com/langchain-ai/agent-protocol)，用于与已部署的智能体交互。

We can group [these endpoints into a few common agent needs](https://github.com/langchain-ai/agent-protocol):

我们可以将 [这些端点按常见智能体需求分组](https://github.com/langchain-ai/agent-protocol)：

* **Runs**: Atomic agent executions
  - **Runs（运行）**：原子化智能体执行

* **Threads**: Multi-turn interactions or human in the loop
  - **Threads（线程）**：多轮交互或人工介入

* **Store**: Long-term memory
  - **Store（存储）**：长期记忆

We can test requests directly [in the API docs](http://localhost:8123/docs#tag/thread-runs).

我们可直接在 [API 文档](http://localhost:8123/docs#tag/thread-runs) 中测试请求。


## SDK

The [LangGraph SDKs](https://docs.langchain.com/langsmith/sdk) (Python and JS) provide a developer-friendly interface to interact with the LangGraph Server API presented above.

[LangGraph SDK](https://docs.langchain.com/langsmith/sdk)（Python 和 JS 版本）为上述 LangGraph Server API 提供了开发者友好的接口。



```python
%%capture --no-stderr
%pip install -U langgraph_sdk
```


```python
from langgraph_sdk import get_client

# Connect via SDK
url_for_cli_deployment = "http://localhost:8123"
client = get_client(url=url_for_cli_deployment)
```

## Remote Graph 远程图

If you are working in the LangGraph library, [Remote Graph](https://docs.langchain.com/langsmith/use-remote-graph) is also a useful way to connect directly to the graph.

若你在 LangGraph 库中工作，[远程图](https://docs.langchain.com/langsmith/use-remote-graph) 也是一种直接连接图的实用方式。



```python
%%capture --no-stderr
%pip install -U langchain_openai langgraph langchain_core
```


```python
from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import convert_to_messages
from langchain_core.messages import HumanMessage, SystemMessage

# Connect via remote graph
url = "http://localhost:8123"
graph_name = "task_maistro" 
remote_graph = RemoteGraph(graph_name, url=url)
```

## Runs Runs（运行）

A "run" represents a [single execution](https://github.com/langchain-ai/agent-protocol?tab=readme-ov-file#runs-atomic-agent-executions) of your graph.

一次“运行”代表你的图的 [单次执行](https://github.com/langchain-ai/agent-protocol?tab=readme-ov-file#runs-atomic-agent-executions)。

Each time a client makes a request:

每次客户端发起请求时：

1. The HTTP worker generates a unique run ID
  - HTTP 工作者生成唯一运行 ID

2. This run and its results are stored in PostgreSQL
  - 该运行及其结果被存储在 PostgreSQL 中

3. You can query these runs to:
  - 你可以查询这些运行以：

   - Check their status
     - 检查其状态

   - Get their results
     - 获取其结果

   - Track execution history
     - 追踪执行历史

You can see a full set of How To guides for various types of runs [here](https://langchain-ai.github.io/langgraph/how-tos/#runs).

各种类型运行的完整操作指南请参见 [此处](https://langchain-ai.github.io/langgraph/how-tos/#runs)。

Let's looks at a few of the interesting things we can do with runs.

让我们来看几个关于运行的有趣功能。

### Background Runs 后台运行

The LangGraph server supports two types of runs:

LangGraph 服务器支持两种运行类型：

* `Fire and forget` - Launch a run in the background, but don’t wait for it to finish
  - `即发即弃（Fire and forget）` —— 在后台启动一次运行，但不等待其完成

* `Waiting on a reply (blocking or polling)` - Launch a run and wait/stream its output
  - `等待回复（阻塞式或轮询式）` —— 启动一次运行并等待/流式传输其输出

Background runs and polling are quite useful when working with long-running agents.

后台运行和轮询在处理长时间运行的智能体时非常有用。

Let's [see](https://docs.langchain.com/langsmith/background-run) how this works:

让我们 [查看](https://docs.langchain.com/langsmith/background-run) 其工作原理：



```python
# Create a thread
thread = await client.threads.create()
thread
```




    {'thread_id': '7f71c0dd-768b-4e53-8349-42bdd10e7caf',
     'created_at': '2024-11-14T19:36:08.459457+00:00',
     'updated_at': '2024-11-14T19:36:08.459457+00:00',
     'metadata': {},
     'status': 'idle',
     'config': {},
     'values': None}




```python
# Check any existing runs on a thread
thread = await client.threads.create()
runs = await client.runs.list(thread["thread_id"])
print(runs)
```

    []



```python
# Ensure we've created some ToDos and saved them to my user_id
user_input = "Add a ToDo to finish booking travel to Hong Kong by end of next week. Also, add a ToDo to call parents back about Thanksgiving plans."
config = {"configurable": {"user_id": "Test"}}
graph_name = "task_maistro" 
run = await client.runs.create(thread["thread_id"], graph_name, input={"messages": [HumanMessage(content=user_input)]}, config=config)
```


```python
# Kick off a new thread and a new run
thread = await client.threads.create()
user_input = "Give me a summary of all ToDos."
config = {"configurable": {"user_id": "Test"}}
graph_name = "task_maistro" 
run = await client.runs.create(thread["thread_id"], graph_name, input={"messages": [HumanMessage(content=user_input)]}, config=config)
```


```python
# Check the run status
print(await client.runs.get(thread["thread_id"], run["run_id"]))
```

    {'run_id': '1efa2c00-63e4-6f4a-9c5b-ca3f5f9bff07', 'thread_id': '641c195a-9e31-4250-a729-6b742c089df8', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21', 'created_at': '2024-11-14T19:38:29.394777+00:00', 'updated_at': '2024-11-14T19:38:29.394777+00:00', 'metadata': {}, 'status': 'pending', 'kwargs': {'input': {'messages': [{'id': None, 'name': None, 'type': 'human', 'content': 'Give me a summary of all ToDos.', 'example': False, 'additional_kwargs': {}, 'response_metadata': {}}]}, 'config': {'metadata': {'created_by': 'system'}, 'configurable': {'run_id': '1efa2c00-63e4-6f4a-9c5b-ca3f5f9bff07', 'user_id': 'Test', 'graph_id': 'task_maistro', 'thread_id': '641c195a-9e31-4250-a729-6b742c089df8', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21'}}, 'webhook': None, 'subgraphs': False, 'temporary': False, 'stream_mode': ['values'], 'feedback_keys': None, 'interrupt_after': None, 'interrupt_before': None}, 'multitask_strategy': 'reject'}


We can see that it has `'status': 'pending'` because it is still running.

我们可以看到其 `'status': 'pending'`（状态为‘待处理’），因为它仍在运行中。

What if we want to wait until the run completes, making it a blocking run?

如果我们希望等待运行完成，使其变为阻塞式运行，该怎么办？

We can use `client.runs.join` to wait until the run completes.

我们可以使用 `client.runs.join` 等待运行完成。

This ensures that no new runs are started until the current run completes on the thread.

这确保在当前线程上的运行完成前，不会启动新的运行。



```python
# Wait until the run completes
await client.runs.join(thread["thread_id"], run["run_id"])
print(await client.runs.get(thread["thread_id"], run["run_id"]))
```

    {'run_id': '1efa2c00-63e4-6f4a-9c5b-ca3f5f9bff07', 'thread_id': '641c195a-9e31-4250-a729-6b742c089df8', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21', 'created_at': '2024-11-14T19:38:29.394777+00:00', 'updated_at': '2024-11-14T19:38:29.394777+00:00', 'metadata': {}, 'status': 'success', 'kwargs': {'input': {'messages': [{'id': None, 'name': None, 'type': 'human', 'content': 'Give me a summary of all ToDos.', 'example': False, 'additional_kwargs': {}, 'response_metadata': {}}]}, 'config': {'metadata': {'created_by': 'system'}, 'configurable': {'run_id': '1efa2c00-63e4-6f4a-9c5b-ca3f5f9bff07', 'user_id': 'Test', 'graph_id': 'task_maistro', 'thread_id': '641c195a-9e31-4250-a729-6b742c089df8', 'assistant_id': 'ea4ebafa-a81d-5063-a5fa-67c755d98a21'}}, 'webhook': None, 'subgraphs': False, 'temporary': False, 'stream_mode': ['values'], 'feedback_keys': None, 'interrupt_after': None, 'interrupt_before': None}, 'multitask_strategy': 'reject'}


Now the run has `'status': 'success'` because it has completed.

现在运行状态为 `'status': 'success'`（成功），因为它已完成。


### Streaming Runs 流式运行

Each time a client makes a streaming request:

每次客户端发起流式请求时：

1. The HTTP worker generates a unique run ID
  - HTTP 工作者生成唯一运行 ID

2. The Queue worker begins work on the run 
  - 队列工作者开始处理该运行

3. During execution, the Queue worker publishes update to Redis
  - 执行期间，队列工作者向 Redis 发布更新

4. The HTTP worker subscribes to updates from Redis for ths run, and returns them to the client 
  - HTTP 工作者订阅该运行在 Redis 中的更新，并将其返回给客户端

This enabled streaming!

这实现了流式传输！

We've covered [streaming](https://langchain-ai.github.io/langgraph/how-tos/#streaming_1) in previous modules, but let's pick one method -- streaming tokens -- to highlight.

我们在之前的模块中已涵盖 [流式传输](https://langchain-ai.github.io/langgraph/how-tos/#streaming_1)，但让我们选取其中一种方法——流式传输 token——加以说明。

Streaming tokens back to the client is especially useful when working with production agents that may take a while to complete.

将 token 流式传输回客户端，在处理可能耗时较长的生产环境智能体时尤为有用。

We [stream tokens](https://docs.langchain.com/langsmith/streaming) using `stream_mode="messages-tuple"`.

我们使用 `stream_mode="messages-tuple"` [流式传输 token](https://docs.langchain.com/langsmith/streaming)。



```python
user_input = "What ToDo should I focus on first."
async for chunk in client.runs.stream(thread["thread_id"], 
                                      graph_name, 
                                      input={"messages": [HumanMessage(content=user_input)]},
                                      config=config,
                                      stream_mode="messages-tuple"):

    if chunk.event == "messages":
        print("".join(data_item['content'] for data_item in chunk.data if 'content' in data_item), end="", flush=True)
```

    You might want to focus on "Call parents back about Thanksgiving plans" first. It has a shorter estimated time to complete (15 minutes) and doesn't have a specific deadline, so it could be a quick task to check off your list. Once that's done, you can dedicate more time to "Finish booking travel to Hong Kong," which is more time-consuming and has a deadline.

## Threads Threads（线程）

Whereas a run is only a single execution of the graph, a thread supports *multi-turn* interactions.

与仅表示图单次执行的运行不同，线程支持*多轮*交互。

When the client makes a graph execution execution with a `thread_id`, the server will save all [checkpoints](https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints) (steps) in the run to the thread in the Postgres database.

当客户端使用 `thread_id` 发起图执行请求时，服务器会将运行中的所有 [检查点](https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints)（步骤）保存至 PostgreSQL 数据库中的该线程。

The server allows us <!-- to  [~check the status of created threads~](https://langchain-ai.github.io/langgraph/cloud/how-tos/check_thread_status/) -->
a variety of ways to [work with threads](https://reference.langchain.com/python/langsmith/deployment/sdk/#langgraph_sdk.client.ThreadsClient).

服务器允许我们 <!-- 通过 [~check the status of created threads~](https://langchain-ai.github.io/langgraph/cloud/how-tos/check_thread_status/) --> 以多种方式 [操作线程](https://reference.langchain.com/python/langsmith/deployment/sdk/#langgraph_sdk.client.ThreadsClient)。

### Check thread state 检查线程状态

For example, we can easily access the state [checkpoints](https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints) saved to any specific thread.

例如，我们可以轻松访问保存到任意特定线程的状态[检查点](https://docs.langchain.com/oss/python/langgraph/persistence#checkpoints)。



```python
thread_state = await client.threads.get_state(thread['thread_id'])
for m in convert_to_messages(thread_state['values']['messages']):
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Give me a summary of all ToDos.
    ==================================[1m Ai Message [0m==================================
    
    Here's a summary of your current ToDo list:
    
    1. **Task:** Finish booking travel to Hong Kong
       - **Status:** Not started
       - **Deadline:** November 22, 2024
       - **Solutions:** 
         - Check flight prices on Skyscanner
         - Book hotel through Booking.com
         - Arrange airport transfer
       - **Estimated Time to Complete:** 120 minutes
    
    2. **Task:** Call parents back about Thanksgiving plans
       - **Status:** Not started
       - **Deadline:** None
       - **Solutions:** 
         - Check calendar for availability
         - Discuss travel arrangements
         - Confirm dinner plans
       - **Estimated Time to Complete:** 15 minutes
    
    Let me know if there's anything else you'd like to do with your ToDo list!
    ================================[1m Human Message [0m=================================
    
    What ToDo should I focus on first.
    ==================================[1m Ai Message [0m==================================
    
    You might want to focus on "Call parents back about Thanksgiving plans" first. It has a shorter estimated time to complete (15 minutes) and doesn't have a specific deadline, so it could be a quick task to check off your list. Once that's done, you can dedicate more time to "Finish booking travel to Hong Kong," which is more time-consuming and has a deadline.


### Copy threads 复制线程

We can also [copy](https://docs.langchain.com/langsmith/use-threads#copy-thread) (i.e. "fork") an existing thread.

我们还可以[复制](https://docs.langchain.com/langsmith/use-threads#copy-thread)（即“分叉”）一个现有线程。

This will keep the existing thread's history, but allow us to create independent runs that do not affect the original thread.

这将保留现有线程的历史记录，同时允许我们创建独立运行，且不会影响原始线程。



```python
# Copy the thread
copied_thread = await client.threads.copy(thread['thread_id'])
```


```python
# Check the state of the copied thread
copied_thread_state = await client.threads.get_state(copied_thread['thread_id'])
for m in convert_to_messages(copied_thread_state['values']['messages']):
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Give me a summary of all ToDos.
    ==================================[1m Ai Message [0m==================================
    
    Here's a summary of your current ToDo list:
    
    1. **Task:** Finish booking travel to Hong Kong
       - **Status:** Not started
       - **Deadline:** November 22, 2024
       - **Solutions:** 
         - Check flight prices on Skyscanner
         - Book hotel through Booking.com
         - Arrange airport transfer
       - **Estimated Time to Complete:** 120 minutes
    
    2. **Task:** Call parents back about Thanksgiving plans
       - **Status:** Not started
       - **Deadline:** None
       - **Solutions:** 
         - Check calendar for availability
         - Discuss travel arrangements
         - Confirm dinner plans
       - **Estimated Time to Complete:** 15 minutes
    
    Let me know if there's anything else you'd like to do with your ToDo list!
    ================================[1m Human Message [0m=================================
    
    What ToDo should I focus on first.
    ==================================[1m Ai Message [0m==================================
    
    You might want to focus on "Call parents back about Thanksgiving plans" first. It has a shorter estimated time to complete (15 minutes) and doesn't have a specific deadline, so it could be a quick task to check off your list. Once that's done, you can dedicate more time to "Finish booking travel to Hong Kong," which is more time-consuming and has a deadline.


### Human in the loop 人工介入循环

We covered [Human in the loop](https://docs.langchain.com/langsmith/add-human-in-the-loop) in Module 3, and the server supports all Human in the loop features that we discussed.

我们在第 3 模块中已介绍过[人工介入循环](https://docs.langchain.com/langsmith/add-human-in-the-loop)，服务器支持我们讨论过的全部人工介入循环功能。

As an example, [we can search, edit, and continue graph execution](https://docs.langchain.com/oss/python/langgraph/persistence#capabilities) from any prior checkpoint.

例如，[我们可以从任意先前检查点搜索、编辑并继续图执行](https://docs.langchain.com/oss/python/langgraph/persistence#capabilities)。



```python
# Get the history of the thread
states = await client.threads.get_history(thread['thread_id'])

# Pick a state update to fork
to_fork = states[-2]
to_fork['values']
```




    {'messages': [{'content': 'Give me a summary of all ToDos.',
       'additional_kwargs': {'example': False,
        'additional_kwargs': {},
        'response_metadata': {}},
       'response_metadata': {},
       'type': 'human',
       'name': None,
       'id': '3680da45-e3a5-4a47-b5b1-4fd4d3e8baf9',
       'example': False}]}




```python
to_fork['values']['messages'][0]['id']
```




    '3680da45-e3a5-4a47-b5b1-4fd4d3e8baf9'




```python
to_fork['next']
```




    ['task_mAIstro']




```python
to_fork['checkpoint_id']
```




    '1efa2c00-6609-67ff-8000-491b1dcf8129'



Let's edit the state.

让我们编辑状态。

Remember how our reducer on `messages` works:

还记得我们作用于 `messages` 的规约器（reducer）是如何工作的吗？

* It will append, unless we supply a message ID.
  - 除非提供消息 ID，否则它将追加消息。

* We supply the message ID to overwrite the message, rather than appending to state!
  - 我们提供消息 ID 是为了覆盖该消息，而非向状态追加！



```python
forked_input = {"messages": HumanMessage(content="Give me a summary of all ToDos that need to be done in the next week.",
                                         id=to_fork['values']['messages'][0]['id'])}

# Update the state, creating a new checkpoint in the thread
forked_config = await client.threads.update_state(
    thread["thread_id"],
    forked_input,
    checkpoint_id=to_fork['checkpoint_id']
)
```


```python
# Run the graph from the new checkpoint in the thread
async for chunk in client.runs.stream(thread["thread_id"], 
                                      graph_name, 
                                      input=None,
                                      config=config,
                                      checkpoint_id=forked_config['checkpoint_id'],
                                      stream_mode="messages-tuple"):

    if chunk.event == "messages":
        print("".join(data_item['content'] for data_item in chunk.data if 'content' in data_item), end="", flush=True)
```

    Here's a summary of your ToDos that need to be done in the next week:
    
    1. **Finish booking travel to Hong Kong**
       - **Status:** Not started
       - **Deadline:** November 22, 2024
       - **Solutions:** 
         - Check flight prices on Skyscanner
         - Book hotel through Booking.com
         - Arrange airport transfer
       - **Estimated Time to Complete:** 120 minutes
    
    It looks like this task is due soon, so you might want to prioritize it. Let me know if there's anything else you need help with!

## Across-thread memory 跨线程内存

In module 5, we covered how the [LangGraph memory `store`](https://docs.langchain.com/oss/python/langgraph/persistence#memory-store) can be used to save information across threads.

在第 5 模块中，我们介绍了如何使用 [LangGraph 内存 `store`](https://docs.langchain.com/oss/python/langgraph/persistence#memory-store) 在多个线程间保存信息。

Our deployed graph, `task_maistro`, uses the `store` to save information -- such as ToDos -- namespaced to the `user_id`.

我们部署的图 `task_maistro` 使用 `store` 保存信息（例如待办事项 ToDos），这些信息按 `user_id` 命名空间隔离。

Our deployment includes a Postgres database, which stores these long-term (across-thread) memories.

我们的部署包含一个 PostgreSQL 数据库，用于存储这些长期（跨线程）内存。

There are several methods available [for interacting with the store](https://reference.langchain.com/python/langsmith/deployment/sdk/#langgraph_sdk.client.StoreClient) in our deployment using the LangGraph SDK.

在我们的部署中，可通过 LangGraph SDK 使用多种方法[与 store 交互](https://reference.langchain.com/python/langsmith/deployment/sdk/#langgraph_sdk.client.StoreClient)。

### Search items 搜索条目

The `task_maistro` graph uses the `store` to save ToDos namespaced by default to (`todo`, `todo_category`, `user_id`).

`task_maistro` 图使用 `store` 保存待办事项（ToDos），默认按 (`todo`, `todo_category`, `user_id`) 命名空间隔离。

The `todo_category` is by default set to `general` (as you can see in `deployment/configuration.py`).

`todo_category` 默认设为 `general`（如 `deployment/configuration.py` 中所示）。

We can simply supply this tuple to search for all ToDos.

我们只需提供该元组即可搜索所有待办事项（ToDos）。



```python
items = await client.store.search_items(
    ("todo", "general", "Test"),
    limit=5,
    offset=0
)
items['items']
```




    [{'value': {'task': 'Finish booking travel to Hong Kong',
       'status': 'not started',
       'deadline': '2024-11-22T23:59:59',
       'solutions': ['Check flight prices on Skyscanner',
        'Book hotel through Booking.com',
        'Arrange airport transfer'],
       'time_to_complete': 120},
      'key': '18524803-c182-49de-9b10-08ccb0a06843',
      'namespace': ['todo', 'general', 'Test'],
      'created_at': '2024-11-14T19:37:41.664827+00:00',
      'updated_at': '2024-11-14T19:37:41.664827+00:00'},
     {'value': {'task': 'Call parents back about Thanksgiving plans',
       'status': 'not started',
       'deadline': None,
       'solutions': ['Check calendar for availability',
        'Discuss travel arrangements',
        'Confirm dinner plans'],
       'time_to_complete': 15},
      'key': '375d9596-edf8-4de2-985b-bacdc623d6ef',
      'namespace': ['todo', 'general', 'Test'],
      'created_at': '2024-11-14T19:37:41.664827+00:00',
      'updated_at': '2024-11-14T19:37:41.664827+00:00'}]



### Add items 添加条目

In our graph, we call `put` to add items to the store.

在我们的图中，我们调用 `put` 将条目添加至 store。

We can use [put](https://reference.langchain.com/python/langsmith/deployment/sdk/#langgraph_sdk.client.StoreClient.put_item) with the SDK if we want to directly add items to the store outside our graph.

如果我们希望在图外部直接向 store 添加条目，可使用 SDK 的 [put](https://reference.langchain.com/python/langsmith/deployment/sdk/#langgraph_sdk.client.StoreClient.put_item) 方法。



```python
from uuid import uuid4
await client.store.put_item(
    ("testing", "Test"),
    key=str(uuid4()),
    value={"todo": "Test SDK put_item"},
)
```


```python
items = await client.store.search_items(
    ("testing", "Test"),
    limit=5,
    offset=0
)
items['items']
```




    [{'value': {'todo': 'Test SDK put_item'},
      'key': '3de441ba-8c79-4beb-8f52-00e4dcba68d4',
      'namespace': ['testing', 'Test'],
      'created_at': '2024-11-14T19:56:30.452808+00:00',
      'updated_at': '2024-11-14T19:56:30.452808+00:00'},
     {'value': {'todo': 'Test SDK put_item'},
      'key': '09b9a869-4406-47c5-a635-4716bd79a8b3',
      'namespace': ['testing', 'Test'],
      'created_at': '2024-11-14T19:53:24.812558+00:00',
      'updated_at': '2024-11-14T19:53:24.812558+00:00'}]



### Delete items 删除条目

We can use the SDK to [delete items](https://reference.langchain.com/python/langsmith/deployment/sdk/#langgraph_sdk.client.StoreClient.delete_item) from the store by key.

我们可使用 SDK 通过键[从 store 删除条目](https://reference.langchain.com/python/langsmith/deployment/sdk/#langgraph_sdk.client.StoreClient.delete_item)。



```python
[item['key'] for item in items['items']]
```




    ['3de441ba-8c79-4beb-8f52-00e4dcba68d4',
     '09b9a869-4406-47c5-a635-4716bd79a8b3']




```python
await client.store.delete_item(
       ("testing", "Test"),
        key='3de441ba-8c79-4beb-8f52-00e4dcba68d4',
    )
```


```python
items = await client.store.search_items(
    ("testing", "Test"),
    limit=5,
    offset=0
)
items['items']
```




    [{'value': {'todo': 'Test SDK put_item'},
      'key': '09b9a869-4406-47c5-a635-4716bd79a8b3',
      'namespace': ['testing', 'Test'],
      'created_at': '2024-11-14T19:53:24.812558+00:00',
      'updated_at': '2024-11-14T19:53:24.812558+00:00'}]




