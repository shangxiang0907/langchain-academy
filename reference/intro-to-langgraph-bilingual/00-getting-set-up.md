---
title: "Getting Set Up / 环境设置"
source: "https://academy.langchain.com/courses/take/intro-to-langgraph/texts/58238105-getting-set-up"
saved_at: "2026-08-22"
---

# Getting Set Up

**中文：环境设置**

![Course image](https://files.cdn.thinkific.com/file_uploads/967498/images/d6a/87b/e15/Course_Banner.png)

## Introduction

**中文：简介**

Welcome to LangChain Academy’s Introduction to LangGraph course!

欢迎来到 LangChain Academy 的 LangGraph 入门课程！

This course is divided into six modules, starting with the basics and gradually progressing to more advanced topics.

本课程分为六个模块，从基础知识开始，逐步深入到更高级的主题。

Each module includes video lessons to walk you through key concepts, along with corresponding notebooks.

每个模块都包含讲解关键概念的视频课程，以及与之对应的笔记本。

You’ll also find a **studio** subdirectory in each module, featuring a set of relevant graphs that we’ll explore using the LangGraph API and Studio.

每个模块中还包含一个 **studio** 子目录，其中提供了一组相关的图；我们将使用 LangGraph API 和 Studio 对它们进行探索。

---

## Setup

**中文：设置**

Here’s our recommended setup to get started with the course.

下面是我们建议的课程入门设置。

We'll be using the set of notebooks located [here](https://github.com/langchain-ai/langchain-academy).

我们将使用位于[此处](https://github.com/langchain-ai/langchain-academy)的一组笔记本。

If you prefer to download the notebooks manually, you can access a ZIP file [here](https://github.com/langchain-ai/langchain-academy/archive/refs/heads/main.zip).

如果你希望手动下载这些笔记本，可以在[此处](https://github.com/langchain-ai/langchain-academy/archive/refs/heads/main.zip)获取 ZIP 文件。

Each module will also include links to the corresponding notebooks, with additional options to access them in Google Colab for a quick start.

每个模块还会提供相应笔记本的链接，并额外提供在 Google Colab 中打开的选项，方便你快速开始。

### Python version

**中文：Python 版本**

Please make sure you're using a Python version higher than 3.11 but lower than 3.14, specifically 3.11, 3.12, or 3.13.

请确保你使用的 Python 版本高于 3.11 且低于 3.14，具体可使用 3.11、3.12 或 3.13。

```text
python3 --version
```

### Clone repo

**中文：克隆代码仓库**

```text
git clone https://github.com/langchain-ai/langchain-academy.git
$ cd langchain-academy
```

### Create an environment and install dependencies

**中文：创建环境并安装依赖项**

```text
$ python3 -m venv lc-academy-env
$ source lc-academy-env/bin/activate
$ pip install -r requirements.txt
```

### Running notebooks

**中文：运行笔记本**

If you don't have Jupyter set up, follow installation instructions [here](https://jupyter.org/install).

如果你尚未设置 Jupyter，请按照[此处](https://jupyter.org/install)的安装说明操作。

```text
$ jupyter notebook
```

### Sign up for LangSmith

**中文：注册 LangSmith**

Create a [LangSmith](https://docs.langchain.com/langsmith/create-account-api-key) account and API key.

创建一个 [LangSmith](https://docs.langchain.com/langsmith/create-account-api-key) 账户和 API 密钥。

You can reference LangSmith docs [here](https://docs.smith.langchain.com/).

你可以在[此处](https://docs.smith.langchain.com/)查看 LangSmith 文档。

Then, set

然后设置：

```text
LANGSMITH_API_KEY="your-key"
LANGSMITH_TRACING_V2=true
LANGSMITH_PROJECT="langchain-academy"
# If you are on the EU instance:
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
```

代码注释翻译：`# If you are on the EU instance:` 表示“如果你使用的是欧盟实例”。

in your environment.

在你的环境中完成上述设置。

### Set up OpenAI API key

**中文：设置 OpenAI API 密钥**

If you don’t have an OpenAI API key, you can sign up [here](https://openai.com/index/openai-api/).

如果你没有 OpenAI API 密钥，可以在[此处](https://openai.com/index/openai-api/)注册。

Note that OpenAI no longer has a free tier.

请注意，OpenAI 已不再提供免费层级。

You will need to add funds to your OpenAI account.

你需要为 OpenAI 账户充值。

You can use other models.

你也可以使用其他模型。

Your results may differ from those in the course video, but that is not unusual in LLM development.

你的结果可能与课程视频中的结果不同，但这在 LLM 开发中并不罕见。

Be sure to set up keys and alter invocations for any alternate models.

如果使用其他模型，请务必设置相应密钥并修改调用方式。

Then, set

然后设置：

```text
OPENAI_API_KEY
```

in your environment.

在你的环境中完成上述设置。

### Tavily for web search

**中文：使用 Tavily 进行网络搜索**

Tavily Search API is a search engine optimized for LLMs and RAG, aimed at efficient, quick, and persistent search results.

Tavily Search API 是一款针对 LLM 和 RAG 优化的搜索引擎，旨在高效、快速且持续地提供搜索结果。

You can sign up for an API key [here](https://tavily.com/).

你可以在[此处](https://tavily.com/)注册 API 密钥。

It’s easy to sign up and offers a generous free tier.

注册过程很简单，并且提供额度充足的免费层级。

Some lessons in Module 4 will use Tavily.

模块4中的部分课程会使用 Tavily。

Then, set

然后设置：

```text
TAVILY_API_KEY
```

in your environment.

在你的环境中完成上述设置。

### Set up LangSmith Studio

**中文：设置 LangSmith Studio**

### Note: LangSmith Studio was formerly LangGraph Studio

**中文：注意：LangSmith Studio 原名为 LangGraph Studio。**

- LangSmith Studio is a custom IDE for viewing and testing agents.
  - 中文：LangSmith Studio 是一个用于查看和测试智能体的定制 IDE。
- Studio can be run locally and opened in your browser on Mac, Windows, and Linux.
  - 中文：在 Mac、Windows 和 Linux 上，Studio 都可以在本地运行并通过浏览器打开。
- See documentation [here](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#local-development-server).
  - 中文：请在[此处](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#local-development-server)查看文档。
- Graphs for LangSmith Studio are in the module-x/studio/ folders.
  - 中文：LangSmith Studio 使用的图位于 `module-x/studio/` 文件夹中。
- To start the local development server, run the following command in your terminal in the /studio directory each module:
  - 中文：若要启动本地开发服务器，请在每个模块的 `/studio` 目录中，通过终端运行以下命令：

```text
langgraph dev
```

You should see the following output:

你应该会看到以下输出：

```text
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

打开浏览器并前往 Studio UI：`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

- To use Studio, you will need to create a .env file with the relevant API keys
  - 中文：若要使用 Studio，需要创建一个包含相关 API 密钥的 `.env` 文件。
- Run this from the command line to create these files for module 1 to 6, as an example:
  - 中文：以下命令演示了如何通过命令行为模块1至模块6创建这些文件：

```text
for i in {1..5}; do
  cp module-$i/studio/.env.example module-$i/studio/.env
  echo "OPENAI_API_KEY=\"$OPENAI_API_KEY\"" > module-$i/studio/.env
done
echo "TAVILY_API_KEY=\"$TAVILY_API_KEY\"" >> module-4/studio/.env
```

---

## Slack Community

**中文：Slack 社区**

If you're interested in connecting with other learners, you can join our Slack Community [here](https://www.langchain.com/join-community).

如果你希望与其他学习者交流，可以在[此处](https://www.langchain.com/join-community)加入我们的 Slack 社区。

If you have any questions, we encourage you to ask and explore in our community-run [Forum](https://forum.langchain.com/).

如果你有任何问题，我们鼓励你在社区运营的[论坛](https://forum.langchain.com/)中提问和探索。

We encourage you to [pay it forward](https://www.langchain.com/community-code) within the community and help other learners.

我们鼓励你在社区中[将善意传递下去](https://www.langchain.com/community-code)，帮助其他学习者。

Please note that our Slack Community is not an official support channel.

请注意，我们的 Slack 社区并非官方支持渠道。

If you are a customer and need product support, reach out to [support@langchain.dev](mailto:support@langchain.dev).

如果你是客户并需要产品支持，请联系 [support@langchain.dev](mailto:support@langchain.dev)。
