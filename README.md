---
title: RAG Colosseum
emoji: 🏛️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RAG Colosseum

A FastAPI RAG service over a 50-passage knowledge base. It combines MiniLM embeddings with BM25 keyword matching and returns sentences copied from the passages.

`POST /query` with `{"question": "...", "top_k": 3}` returns `{"answer", "sources", "scores", "confidence"}`.
