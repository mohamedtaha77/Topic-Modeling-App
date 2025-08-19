# 🧠 Topic Modeling on News — BBC Dataset

An interactive **topic modeling app** built with **Streamlit**.  
It discovers hidden themes in news articles using:

- **NMF (Non‑negative Matrix Factorization)** — *recommended on this dataset*
- **LDA (Latent Dirichlet Allocation)** — baseline for comparison

**Dataset:** [BBC News Summary (Kaggle)](https://www.kaggle.com/datasets/pariza/bbc-news-summary)  
**Live app:** https://topic-modeling-app-77.streamlit.app/

---

## 📌 Features

✅ Paste text **or upload a `.txt` file**  
✅ Choose **NMF (recommended)** or **LDA**  
✅ **Top words** for the predicted topic  
✅ **Top‑5 topic probabilities** (bar plot)  
✅ **Dominant category** label per topic (NMF) + **purity** percentage  
✅ **Similar‑article lookup** (shows training articles closest to your input)  
✅ Mini **metrics** row (coherence, K) to justify the recommendation  
✅ Clean, single‑page Streamlit UI

> NMF is recommended because it achieved **higher c_v coherence** on this dataset in the companion notebook and enables category‑aware analytics.

---

## 📁 Files Included / Expected

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI (this repo) |
| `requirements.txt` | App dependencies (Streamlit, scikit‑learn, gensim, etc.) |
| `README.md` | This file |
| `tfidf_vectorizer.joblib` | Fitted TF‑IDF vectorizer |
| `nmf_model.joblib` | Trained NMF model |
| `best_lda.model` | Trained LDA model (gensim) |
| `dictionary.dict` | Gensim dictionary used to train LDA |
| `documents_with_topics.csv` | Corpus with `text`, `clean_text`, `true_category`, and assigned `topic` (for similarity lookup) |
| `topic_to_category_nmf.json` | Mapping: topic → dominant category + purity  |
| `model_metrics.json` | Small JSON with `{NMF: {K, coherence_c_v}, LDA: {K, coherence_c_v}}` for the UI badge |

> Only `app.py` + `requirements.txt` are needed to **run the app**.  
> The other files are **artifacts exported from the notebook** to enable full functionality (prediction, keywords, similarity, and badges).

---

## 🚀 How to Run Locally

1) **Clone** and enter the project
```bash
git clone https://github.com/yourname/topic-modeling-news.git
cd topic-modeling-news
```

2) **(Optional) Create a virtual environment**
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3) **Install dependencies**
```bash
pip install -r requirements.txt
```

4) **Place model artifacts** in the project root:
```
tfidf_vectorizer.joblib
nmf_model.joblib
best_lda.model
dictionary.dict
documents_with_topics.csv
topic_to_category_nmf.json
model_metrics.json
```

5) **Run the app**
```bash
streamlit run app.py
```

---

## 🌍 Deployment (Streamlit Cloud)

- Go to https://streamlit.io/cloud  
- Connect your GitHub repo and pick the branch  
- Set **Main file**: `app.py`  
- Add a **Python version** and **requirements.txt** in app settings  
- Place artifact files in the app’s working directory (e.g., upload them as repo assets)

> First load may take longer while dependencies initialize.

---

## 🧪 Notebook Workflow (what was done)

- **Data Collection**  
  - Download the **BBC News Summary** from Kaggle and load all five categories: `business`, `entertainment`, `politics`, `sport`, `tech`.

- **Preprocessing**  
  - Lowercasing, URL removal, punctuation/digit stripping, token filtering, lemmatization (in the notebook).  
  - Build `clean_text` and `tokens` fields; drop empty rows.

- **Topic Modeling**  
  - **LDA**: build dictionary/corpus; sweep `K` over several values; compute **c_v coherence**; select best‑`K`.  
  - **NMF**: TF‑IDF features; train `NMF(n_components=K)`; get top words per topic.

- **Evaluation & Visualization**  
  - Coherence vs **K** plot (for LDA).  
  - **pyLDAvis** (LDA), **word clouds** (NMF).  
  - Category × Topic **crosstabs**, stacked bars, row‑normalized **heatmap**.  
  - Compute **dominant topic per category** and **dominant category per topic**.  
  - Derive **purity** = share of dominant topic within each category.

- **Exports (for the app)**  
  - `tfidf_vectorizer.joblib`, `nmf_model.joblib`  
  - `best_lda.model`, `dictionary.dict`  
  - `documents_with_topics.csv` (text, clean_text, true_category, topic)  
  - `topic_to_category_nmf.json` (topic → dominant_category, purity)  
  - `model_metrics.json` with coherence and K for both models

**Key takeaways:**  
- On this dataset, **NMF** achieved **higher coherence** and offers category‑aware analytics and similar‑article lookup in the UI.  
- **LDA** remains a solid baseline and is helpful for interactive exploration via pyLDAvis and top‑word lists.

---


## 🔒 License

MIT

