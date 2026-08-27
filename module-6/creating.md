# Creating a deployment 创建部署

Let's create a deployment of the `task_maistro` app that we created in module 5.

让我们为我们在模块 5 中创建的 `task_maistro` 应用创建一个部署。

## Code structure 代码结构

[The following information should be provided](https://docs.langchain.com/langsmith/application-structure) to create a LangGraph Platform deployment:

[以下信息应被提供](https://docs.langchain.com/langsmith/application-structure)，以创建 LangGraph 平台部署：

* A [LangGraph API Configuration file](https://docs.langchain.com/langsmith/application-structure#configuration-file) - `langgraph.json`
  - 一个 [LangGraph API 配置文件](https://docs.langchain.com/langsmith/application-structure#configuration-file) — `langgraph.json`

* The graphs that implement the logic of the application - e.g., `task_maistro.py`
  - 实现应用逻辑的图（graph）——例如，`task_maistro.py`

* A file that specifies dependencies required to run the application - `requirements.txt`
  - 指定运行应用所需依赖项的文件 — `requirements.txt`

* Supply environment variables needed for the application to run - `.env` or `docker-compose.yml`
  - 提供应用运行所需的环境变量 — `.env` 或 `docker-compose.yml`

We have this already in the `module-6/deployment` directory!

这些文件我们已在 `module-6/deployment` 目录中准备就绪！

## CLI 命令行界面（CLI）

The [LangGraph CLI](https://docs.langchain.com/langsmith/cli) is a command-line interface for creating a LangGraph Platform deployment.

[LangGraph CLI](https://docs.langchain.com/langsmith/cli) 是用于创建 LangGraph 平台部署的命令行接口。



```python
%%capture --no-stderr
%pip install -U langgraph-cli
```

To create a <!-- [~self-hosted deployment~](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-to-do-a-self-hosted-deployment-of-langgraph) --> [self-hosted deployment](https://docs.langchain.com/langsmith/self_hosted_data_plane), we'll follow a few steps.

为创建 [自托管部署](https://docs.langchain.com/langsmith/self_hosted_data_plane)，我们将按几个步骤操作。

### Build Docker Image for LangGraph Server 为 LangGraph 服务器构建 Docker 镜像

We first use the langgraph CLI to create a Docker image for the [LangGraph Server](https://docs.google.com/presentation/d/18MwIaNR2m4Oba6roK_2VQcBE_8Jq_SI7VHTXJdl7raU/edit#slide=id.g313fb160676_0_32).

我们首先使用 langgraph CLI 为 [LangGraph 服务器](https://docs.google.com/presentation/d/18MwIaNR2m4Oba6roK_2VQcBE_8Jq_SI7VHTXJdl7raU/edit#slide=id.g313fb160676_0_32) 创建一个 Docker 镜像。

This will package our graph and dependencies into a Docker image.

该操作会将我们的图及其依赖项打包进一个 Docker 镜像。

A Docker image is a template for a Docker container that contains the code and dependencies required to run the application.

Docker 镜像是 Docker 容器的模板，其中包含运行应用所需的代码和依赖项。

Ensure that [Docker](https://docs.docker.com/engine/install/) is installed and then run the following command to create the Docker image, `my-image`:

请确保已安装 [Docker](https://docs.docker.com/engine/install/)，然后运行以下命令创建名为 `my-image` 的 Docker 镜像：

```
$ cd module-6/deployment
$ langgraph build -t my-image
```

### Set Up Redis and PostgreSQL 设置 Redis 和 PostgreSQL

If you already have Redis and PostgreSQL running (e.g., locally or on other servers), then create and run the LangGraph Server container  [by itself](https://docs.langchain.com/langsmith/deploy-hybrid#running-the-application-locally) with the URIs for Redis and PostgreSQL:

如果你已有正在运行的 Redis 和 PostgreSQL（例如，在本地或其他服务器上），则可 [单独](https://docs.langchain.com/langsmith/deploy-hybrid#running-the-application-locally) 创建并运行 LangGraph 服务器容器，并传入 Redis 和 PostgreSQL 的 URI。

```
docker run \
    --env-file .env \
    -p 8123:8000 \
    -e REDIS_URI="foo" \
    -e DATABASE_URI="bar" \
    -e LANGSMITH_API_KEY="baz" \
    my-image
```

Alternatively, you can use the provided `docker-compose.yml` file to create three separate containers based on the services defined:

或者，你可以使用提供的 `docker-compose.yml` 文件，基于所定义的服务创建三个独立的容器：

* `langgraph-redis`: Creates a new container using the official Redis image.
  - `langgraph-redis`：使用官方 Redis 镜像创建新容器。

* `langgraph-postgres`: Creates a new container using the official Postgres image.
  - `langgraph-postgres`：使用官方 Postgres 镜像创建新容器。

* `langgraph-api`: Creates a new container using your pre-built image.
  - `langgraph-api`：使用你预先构建的镜像创建新容器。

Simply copy the `docker-compose-example.yml` and add the following environment variables to run the deployed `task_maistro` app:

只需复制 `docker-compose-example.yml`，并添加以下环境变量，即可运行已部署的 `task_maistro` 应用：

* `IMAGE_NAME` (e.g., `my-image`) 

* `LANGSMITH_API_KEY`

* `OPENAI_API_KEY`

Then, <!-- [~launch the deployment~](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#using-docker-compose) [launch the deployment](https://docs.langchain.com/langsmith/self_hosted_data_plane): --> launch the deployment!

然后，[启动部署](https://docs.langchain.com/langsmith/self_hosted_data_plane)！

```
$ cd module-6/deployment
$ docker compose up
```
