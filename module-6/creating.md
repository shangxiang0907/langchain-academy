# Creating a deployment

Let's create a deployment of the `task_maistro` app that we created in module 5.

## Code structure

[The following information should be provided](https://docs.langchain.com/langsmith/application-structure) to create a LangGraph Platform deployment:

* A [LangGraph API Configuration file](https://docs.langchain.com/langsmith/application-structure#configuration-file) - `langgraph.json`
* The graphs that implement the logic of the application - e.g., `task_maistro.py`
* A file that specifies dependencies required to run the application - `requirements.txt`
* Supply environment variables needed for the application to run - `.env` or `docker-compose.yml`

We have this already in the `module-6/deployment` directory! 

## CLI

The [LangGraph CLI](https://docs.langchain.com/langsmith/cli) is a command-line interface for creating a LangGraph Platform deployment.


```python
%%capture --no-stderr
%pip install -U langgraph-cli
```

To create a <!-- [~self-hosted deployment~](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#how-to-do-a-self-hosted-deployment-of-langgraph) --> [self-hosted deployment](https://docs.langchain.com/langsmith/self_hosted_data_plane), we'll follow a few steps. 

### Build Docker Image for LangGraph Server

We first use the langgraph CLI to create a Docker image for the [LangGraph Server](https://docs.google.com/presentation/d/18MwIaNR2m4Oba6roK_2VQcBE_8Jq_SI7VHTXJdl7raU/edit#slide=id.g313fb160676_0_32).

This will package our graph and dependencies into a Docker image.

A Docker image is a template for a Docker container that contains the code and dependencies required to run the application.

Ensure that [Docker](https://docs.docker.com/engine/install/) is installed and then run the following command to create the Docker image, `my-image`:

```
$ cd module-6/deployment
$ langgraph build -t my-image
```

### Set Up Redis and PostgreSQL

If you already have Redis and PostgreSQL running (e.g., locally or on other servers), then create and run the LangGraph Server container  [by itself](https://docs.langchain.com/langsmith/deploy-hybrid#running-the-application-locally) with the URIs for Redis and PostgreSQL:

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

* `langgraph-redis`: Creates a new container using the official Redis image.
* `langgraph-postgres`: Creates a new container using the official Postgres image.
* `langgraph-api`: Creates a new container using your pre-built image.

Simply copy the `docker-compose-example.yml` and add the following environment variables to run the deployed `task_maistro` app:

* `IMAGE_NAME` (e.g., `my-image`) 
* `LANGSMITH_API_KEY`
* `OPENAI_API_KEY`

Then, <!-- [~launch the deployment~](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted/#using-docker-compose) [launch the deployment](https://docs.langchain.com/langsmith/self_hosted_data_plane): --> launch the deployment!

```
$ cd module-6/deployment
$ docker compose up
```


