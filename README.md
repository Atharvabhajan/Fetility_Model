# 🚀 AI-Powered E-Commerce Product Recommendation System

An **end-to-end intelligent recommendation platform** combining:

- 🤖 Machine Learning–based recommendation models
- 🧠 LLM-powered explanation engine
- ⚡ Production-grade FastAPI backend
- 🎨 Modern React + TailwindCSS dashboard

This system delivers **personalized recommendations**, explains **why an item is suggested**, and demonstrates **real-world recommender system design** used by platforms like **Amazon, Flipkart, and Myntra**.

---

## 🌟 Project Highlights

✔ End-to-end AI recommendation system  
✔ Multiple ML models for comparative analysis  
✔ Hybrid recommendation strategy (content + behavior)  
✔ LLM-generated natural language explanations  
✔ Clean, interactive dashboard  
✔ Strong ranking metrics (Hit@K, NDCG)  
✔ Fully documented architecture & pipeline  

---

## 🧠 Models Implemented (Comparative Study)

This project implements **four models**, all trained and evaluated using **consistent preprocessing, metrics, and output formats**.

| Model | Type | Purpose |
|------|------|---------|
| Content-Based Recommender | Embedding similarity | Core recommendation engine |
| Hybrid Recommender | Content + behavioral scoring | Improved personalization |
| Random Forest Classifier | ML baseline | Feature-based prediction |
| Logistic Regression | ML baseline | Interpretable linear model |

---

## 📊 Model Performance Summary

### Recommendation Models (Ranking Metrics)

| Model | Hit@10 | NDCG@10 |
|------|--------|---------|
| Content-Based | 0.9398 | 0.5781 |
| Hybrid (α = 0.2) | 0.9373 | 0.5777 |
| Hybrid (α = 0.1–0.3) | Excellent | Highly Stable |

---

### Classification Models (Predictive Metrics)

| Model | Accuracy | F1-Score | ROC-AUC |
|------|----------|----------|---------|
| Random Forest | High | Strong | Robust |
| Logistic Regression | Stable | Interpretable | Consistent |

---

## 🧠 Model Development Approach

### ① Data Understanding & Feature Engineering

We extracted meaningful **product-level features**, including:

- Product title
- Description / blurb
- Category & metadata
- Style and functional attributes

These were converted into **dense semantic embeddings** using **Sentence-BERT / MiniLM**.

These embeddings capture:
- Semantic similarity
- Style alignment
- Functional relevance

---

### ② Content-Based Similarity Model (Core Engine)

- Cosine similarity between product embeddings
- Top-K most relevant items returned per user
- Cold-start friendly and scalable

**Performance**
- Hit@10 = 0.9398
- NDCG@10 = 0.5781

---

### ③ Hybrid Recommendation Engine

To strengthen personalization, we introduced **behavioral signals**.

#### Hybrid Scoring Formula

