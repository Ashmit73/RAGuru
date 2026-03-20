# RAGuru — RAG Based Intelligent Q&A System

RAGuru is an intelligent document question-answering system built using
Retrieval Augmented Generation (RAG) pipeline with HuggingFace models.

---

##  How It Works
```
 Document Input
      ↓
 Text Chunking
      ↓
 HuggingFace Embeddings
      ↓
 FAISS Vector Store
      ↓
 Semantic Search + Retrieval
      ↓
 HuggingFace LLM
      ↓
 Accurate Answer
```



## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| LLM & Embeddings | HuggingFace Models |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| UI | Streamlit |
| Deployment | Streamlit + Ngrok |

---

## How to Run Locally
```bash
git clone https://github.com/Ashmit73/RAGuru
pip install -r requirements.txt
streamlit run app.py
```



##  Features

-  Upload any PDF/document
-  Ask questions in natural language
-  RAG pipeline for accurate context-aware answers
-  Powered by HuggingFace open-source models



## 👨‍💻 Author
**Ashmit Agrawal** — B.Tech CSE (AIML)
[![GitHub](https://img.shields.io/badge/GitHub-Ashmit73-black?style=flat&logo=github)](https://github.com/Ashmit73)




