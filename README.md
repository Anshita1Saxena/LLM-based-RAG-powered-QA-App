# LLM-based-RAG-powered-QA-App

Here is the Anyscale reference blog which I used in order to develop this application:- [Anyscale Blog Link](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)

This repo implements a production-ready, scalable Retrieval Augmented Generation (RAG)-powered LLM-based Open Generative (or Extractive) context-aware Question-Answering (QA) App that:

1. Takes as input a new query (or question)
2. Implements vector similarity search within the embedding space by seeking relevant contexts corresponding to the incoming query in the vector database
3. Passes the relevant contexts as well as the input query to LLM
4. LLM then produces the answer to the input query while being aware of the relevant contexts related to the requested query.

This project also includes Fine-tuning a 20B parameters Large Language Model (LLM) in a multi-GPU cluster environment by leveraging the distributed training paradigm. Moreover, this repo develops scalable major ML workloads for contexts (load, embed, and index the contexts in the vector database) across multiple workers with different compute resources and serves the LLM App in a highly robust and scalable manner.

The architecture Flow for the app is shown below:
![Architecture_Flow](https://github.com/Anshita1Saxena/LLM-based-RAG-powered-QA-App/blob/main/images/architecture.png)

The component evaluations of the retrieval system and LLM (left) and Overall evaluation (right) are shown below:
![component_evaluations](https://github.com/Anshita1Saxena/LLM-based-RAG-powered-QA-App/blob/main/images/component_evaluation.png)

## Requirements
1. Python
2. Streamlit
3. PEFT (for Parameter-Efficient Fine-Tuning)
4. Accelerate
5. Ray (for distributed LLM Fine-Tuning)
6. Datasets
7. Transformers
8. PyTorch
9. Numpy
10. Scikit-Learn
11. Deta (To access Deta Vector Database)
12. LangChain
13. FastAPI (To serve production-ready LLM App)

## Data
Squad dataset is used to fine-tune `Eleuther AI's GPT-Neo 20B LLM` model, which comprises Title, Question, Answer, and Context for each of the `98.2k` dataset IDs.

## LLM Training and Serving
1. The Fine-Tuning process for GPT-Neo LLM model can be found in `finetune.py` file.
2. The code to create RAG-powered LLM Agent for QA task can be seen in `qa_agent.py` file.
3. To build the agent as production-ready API for QA task, it's worth delving deep into `serve.py` file.
4. To seek prospects of using Streamlit to deploy the LLM app, head to `streamlit.py` file.
5. All hyperparameters to control fine-tuning of the model are provided in the given `config.py` file.


