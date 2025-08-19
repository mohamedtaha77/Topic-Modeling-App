import os, re, string, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

# ---------- App ----------
st.set_page_config(page_title="Topic Modeling on News Articles", layout="wide")
WARN = []

# ---------- Paths ----------
BASE = Path(__file__).resolve().parent
def P(name: str) -> str:
    return str(BASE / name)

def sha16(name: str) -> str:
    try:
        return hashlib.sha256(open(P(name), "rb").read()).hexdigest()[:16]
    except Exception:
        return "n/a"

# ---------- Preprocess ----------
STOP = set("""
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
she her hers herself it its itself they them their theirs themselves what which who whom
this that these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about against
between into through during before after above below to from up down in out on off over
under again further then once here there when where why how all any both each few more most
other some such no nor not only own same so than too very s t can will just don should now
""".split())

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", "", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\d+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_for_vect(s: str) -> str:
    toks = [w for w in clean_text(s).split() if w.isalpha() and w not in STOP]
    return " ".join(toks)

def preprocess_tokens_for_lda(s: str):
    return [w for w in clean_text(s).split() if w.isalpha() and w not in STOP]

# ------------------------------
# Load artifacts
# ------------------------------
vect = nmf = terms = H = None
df_docs = None
X_corpus = None
VECT_OK = NMF_OK = DIM_OK = False

def try_transform(v) -> bool:
    try:
        v.transform(["sanity test"])
        return True
    except NotFittedError:
        return False
    except Exception:
        return False

try:
    vect = joblib.load(P("tfidf_vectorizer.joblib"))
    nmf  = joblib.load(P("nmf_model.joblib"))
    NMF_OK = nmf is not None

    VECT_OK = try_transform(vect)

    if not VECT_OK:
        try:
            _ = vect.get_feature_names_out() 
            try:
                idf_vals = vect.idf_
            except Exception:
                idf_vals = None
            if idf_vals is not None:
                if getattr(vect._tfidf, "idf_", None) is None:
                    vect._tfidf.idf_ = np.array(idf_vals)
                if getattr(vect._tfidf, "_idf_diag", None) is None:
                    vect._tfidf._idf_diag = sparse.spdiags(
                        vect._tfidf.idf_, 0, len(vect._tfidf.idf_), len(vect._tfidf.idf_)
                    )
            VECT_OK = try_transform(vect)
        except Exception as e:
            WARN.append(f"Vectorizer repair attempt 1 failed: {type(e).__name__}: {e}")
            VECT_OK = False

    csv_path = BASE / "documents_with_topics.csv"
    if not VECT_OK and csv_path.exists():
        try:
            df_docs = pd.read_csv(csv_path)
            if "clean_text" not in df_docs.columns:
                df_docs["clean_text"] = df_docs["text"].fillna("").map(preprocess_for_vect)
            rebuilt = TfidfVectorizer(stop_words="english",
                                      vocabulary=getattr(vect, "vocabulary_", None))
            rebuilt.fit(df_docs["clean_text"].fillna(""))
            vect = rebuilt
            VECT_OK = try_transform(vect)
        except Exception as e:
            WARN.append(f"Vectorizer repair attempt 2 (rebuild from CSV) failed: {type(e).__name__}: {e}")

    if VECT_OK:
        terms = np.array(vect.get_feature_names_out())
    if NMF_OK and VECT_OK:
        H = nmf.components_
        DIM_OK = (H.shape[1] == len(terms))
        if not DIM_OK:
            WARN.append("Vectorizer/NMF dimension mismatch. Re-export BOTH from the same run.")
    else:
        if not VECT_OK:
            WARN.append("Vectorizer is NOT fitted and could not be auto-repaired.")
        if not NMF_OK:
            WARN.append("NMF model not loaded.")

except Exception as e:
    WARN.append(f"NMF artifacts load error: {type(e).__name__}: {e}")

# ------------------------------
# LDA artifacts
# ------------------------------
lda = dictionary = None
try:
    from gensim import models, corpora
    lda = models.LdaModel.load(P("best_lda.model"))
    dictionary = corpora.Dictionary.load(P("dictionary.dict"))
except Exception:
    WARN.append("LDA artifacts missing or failed to load (OK if you only use NMF).")

# ------------------------------
# corpus for examples
# ------------------------------
csv_path = BASE / "documents_with_topics.csv"

# Load df_docs
if df_docs is None and csv_path.exists():
    try:
        df_docs = pd.read_csv(csv_path)
    except Exception as e:
        WARN.append(f"Could not read documents_with_topics.csv: {type(e).__name__}: {e}")
        df_docs = None

# Ensure clean_text exists
if df_docs is not None and "clean_text" not in df_docs.columns:
    try:
        df_docs["clean_text"] = df_docs["text"].fillna("").map(preprocess_for_vect)
    except Exception as e:
        WARN.append(f"Could not build clean_text for corpus: {type(e).__name__}: {e}")

# Build X_corpus
if df_docs is not None and VECT_OK:
    try:
        X_corpus = vect.transform(df_docs["clean_text"].fillna(""))
    except Exception as e:
        WARN.append(f"Could not transform corpus for similarity: {type(e).__name__}: {e}")
elif df_docs is None:
    WARN.append("documents_with_topics.csv not found — similar-article lookup will be hidden.")
elif not VECT_OK:
    WARN.append("Vectorizer not ready — cannot compute similarities to corpus.")

# ------------------------------
# Dominant category per topic (NMF)
# ------------------------------
TOPIC2CAT, TOPIC2PURITY = {}, {}

meta_path = BASE / "topic_to_category_nmf.json"
if meta_path.exists():
    try:
        topic_meta_df = pd.read_json(P("topic_to_category_nmf.json")).T
        try:
            topic_meta_df.index = topic_meta_df.index.astype(int)
        except Exception:
            pass
        TOPIC2CAT = topic_meta_df["dominant_category"].to_dict()
        TOPIC2PURITY = topic_meta_df["purity"].to_dict()
    except Exception as e:
        WARN.append(f"Could not read topic_to_category_nmf.json: {type(e).__name__}: {e}")

if (not TOPIC2CAT) and (df_docs is not None) and ("topic" in df_docs.columns) and ("true_category" in df_docs.columns):
    try:
        ct_topics = pd.crosstab(df_docs["topic"], df_docs["true_category"])
        TOPIC2CAT = ct_topics.idxmax(axis=1).to_dict()
        TOPIC2PURITY = (ct_topics.max(axis=1) / ct_topics.sum(axis=1)).to_dict()
    except Exception as e:
        WARN.append(f"Could not compute dominant categories: {type(e).__name__}: {e}")

# ------------------------------
# model metrics
# ------------------------------
NMF_COH = LDA_COH = None
NMF_K = LDA_K = None
try:
    import json
    with open(P("model_metrics.json"), "r") as f:
        mm = json.load(f)
    NMF_COH = mm.get("NMF", {}).get("coherence_c_v")
    LDA_COH = mm.get("LDA", {}).get("coherence_c_v")
    NMF_K   = mm.get("NMF", {}).get("K")
    LDA_K   = mm.get("LDA", {}).get("K")
except Exception:
    pass

# ---------- Helpers ----------
TOPN_WORDS = 12

def nmf_keywords(k, n=TOPN_WORDS):
    idx = np.argsort(H[k])[::-1][:n]
    return ", ".join(terms[idx])

# Robust LDA keyword getter
def lda_keywords(k, n=TOPN_WORDS):
    """Return top words for LDA topic k even if id2word is missing."""
    try:
        k = int(k)
        if getattr(lda, "id2word", None) is not None:
            pairs = lda.show_topic(k, topn=n)
            return ", ".join(str(w) for (w, _) in pairs)
        termids = [tid for (tid, _) in lda.get_topic_terms(k, topn=n)]
        words = []
        for tid in termids:
            try:
                words.append(dictionary[tid] if dictionary is not None else str(tid))
            except KeyError:
                words.append(str(tid))
        return ", ".join(words)
    except Exception as e:
        WARN.append(f"LDA keywords failed for topic {k}: {type(e).__name__}: {e}")
        return "(keywords unavailable)"

def similar_examples(x_vec, topn=3):
    if X_corpus is None or df_docs is None or x_vec is None:
        return []
    sims = cosine_similarity(x_vec, X_corpus).ravel()
    nn = sims.argsort()[::-1][:topn]
    out = []
    for i in nn:
        row = df_docs.iloc[i]
        snippet = (row.get("text","") or "")[:220].replace("\n"," ") + "..."
        out.append({
            "score": float(sims[i]),
            "topic": int(row.get("topic", -1)) if "topic" in row else None,
            "category": row.get("true_category", ""),
            "snippet": snippet
        })
    return out

def safe_vect_transform(text_clean):
    if not (VECT_OK and DIM_OK):
        return None
    try:
        return vect.transform([text_clean])
    except NotFittedError:
        return None

def predict_with_nmf(text: str):
    if not (VECT_OK and NMF_OK and DIM_OK):
        return {"model":"NMF","topic_id":None,"confidence":0.0,"top2":[],"keywords":"","vector":None}
    clean = preprocess_for_vect(text)
    x = safe_vect_transform(clean)
    w = nmf.transform(x)[0]
    probs = w / (w.sum() + 1e-12)
    order = probs.argsort()[::-1]
    k = int(order[0])
    return {
        "model": "NMF",
        "topic_id": k,
        "confidence": float(probs[k]),
        "top2": [(int(i), float(probs[i])) for i in order[:2]],
        "keywords": nmf_keywords(k),
        "vector": x
    }

def predict_with_lda(text: str):
    if dictionary is None or lda is None:
        return {"model":"LDA","topic_id":None,"confidence":0.0,"top2":[],"keywords":"","probs":None}
    toks = preprocess_tokens_for_lda(text)
    if not toks:
        toks = [w for w in clean_text(text).split() if w.isalpha()]
    bow = dictionary.doc2bow(toks)
    dist = lda.get_document_topics(bow, minimum_probability=0.0)
    if not dist:
        return {"model":"LDA","topic_id":None,"confidence":0.0,"top2":[],"keywords":"","probs":None}
    K = lda.num_topics
    probs = np.zeros(K, dtype=float)
    for t, p in dist:
        probs[int(t)] = float(p)
    order = probs.argsort()[::-1]
    k = int(order[0])
    top2 = [(int(i), float(probs[i])) for i in order[:2]]
    kws = lda_keywords(k, n=TOPN_WORDS)
    return {
        "model": "LDA",
        "topic_id": k,
        "confidence": float(probs[k]),
        "top2": top2,
        "keywords": kws,
        "probs": probs
    }

# ---------- UI ----------
st.title("🧠 Topic Modeling on News Articles")

# ===== Model picker =====
st.markdown("### Choose model")
st.caption(
    "⭐ **Recommended:** NMF — higher coherence on this "
    "[dataset](https://www.kaggle.com/datasets/pariza/bbc-news-summary) "
    "and richer analytics (dominant category, similar-article lookup)."
)

choice = st.radio(
    "",
    ["NMF (recommended)", "LDA (baseline)"],
    horizontal=True,
    index=0
)
choice = "NMF" if choice.startswith("NMF") else "LDA"

# Mini metrics row
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("NMF coherence", f"{NMF_COH:.3f}" if NMF_COH else "—")
mc2.metric("LDA coherence", f"{LDA_COH:.3f}" if LDA_COH else "—")
mc3.metric("NMF K", str(NMF_K) if NMF_K is not None else "—")
mc4.metric("LDA K", str(LDA_K) if LDA_K is not None else "—")

# Feature checklist
nmf_has_dom = bool(TOPIC2CAT)
has_corpus  = (X_corpus is not None)
fc1, fc2 = st.columns(2)
with fc1:
    st.markdown("**NMF features**")
    st.write("✅ Confidence")
    st.write("✅ similar-article lookup")
    st.write("✅ Top-5 topic probabilities")
    st.write(("✅ " if nmf_has_dom else "➖ ") + "Dominant category label")
    st.write("✅ Key words from dominant category")

with fc2:
    st.markdown("**LDA features**")
    st.write("✅ Confidence")
    st.write("✅ similar-article lookup")
    st.write("✅ Top-5 topic probabilities")
    st.write(("❌ " if nmf_has_dom else "➖ ") + "Dominant category label")
    st.write("❌ Key words from dominant category")
    

# ---------- Input (upload OR sample) ----------
st.markdown("### Paste text or upload a file")

uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
uploaded_text = None
if uploaded is not None:
    try:
        raw = uploaded.read()
        try:
            uploaded_text = raw.decode("utf-8")
        except UnicodeDecodeError:
            uploaded_text = raw.decode("latin-1")
        uploaded_text = uploaded_text.strip()
        if len(uploaded_text) > 50000:  # keep extreme files in check
            uploaded_text = uploaded_text[:50000]
        st.caption(f"Loaded {len(uploaded_text.split()):,} words / {len(uploaded_text):,} characters from **{uploaded.name}**.")
    except Exception as e:
        st.error(f"Could not read file: {type(e).__name__}: {e}")

SAMPLE_TEXT = (
    "Shares of a leading chipmaker jumped after the company reported record quarterly revenue, "
    "citing surging demand for AI accelerators in data centers. The gains helped lift major stock "
    "indexes even as the central bank signaled it would keep interest rates higher for longer to "
    "cool inflation. Analysts pointed to strong cloud spending and new product launches as drivers "
    "of guidance, while regulators in Europe opened an antitrust inquiry into exclusive supply deals. "
    "In parallel, a large social media platform announced new privacy controls and encrypted messaging "
    "by default, aiming to address criticism from lawmakers and consumer groups."
)

# Text area uses uploaded text if present; otherwise the sample
text = st.text_area(
    "Text",
    height=200,
    value=(uploaded_text if uploaded_text else SAMPLE_TEXT),
    placeholder="Paste or type your article paragraph here…"
)

# -------- Predict --------
if st.button("Predict") and text.strip():
    if choice == "NMF":
        if not (VECT_OK and NMF_OK and DIM_OK):
            st.error("NMF unavailable. Ensure a FITTED tfidf_vectorizer.joblib and a matching nmf_model.joblib from the SAME run.")
        else:
            res = predict_with_nmf(text)
            st.subheader(f"Topic {res['topic_id']}")
            st.metric("Confidence", f"{res['confidence']:.2f}")
            st.write("**Keywords from dominant category:**", res["keywords"])

            dom = TOPIC2CAT.get(res['topic_id'])
            pur = TOPIC2PURITY.get(res['topic_id'])
            if dom is not None and pur is not None:
                st.info(f"Dominant category in training: **{dom}** ({pur:.0%})")

            top2 = res["top2"]
            if len(top2) == 2 and abs(top2[0][1] - top2[1][1]) < 0.10:
                st.warning(f"Close call: Topic {top2[0][0]} vs Topic {top2[1][0]}")

            x = res["vector"]
            w = nmf.transform(x)[0]
            probs = w / (w.sum() + 1e-12)
            order = probs.argsort()[::-1][:5]
            st.bar_chart(pd.Series(probs[order], index=[f"T{int(i)}" for i in order]))

            exs = similar_examples(x, topn=3)
            if exs:
                st.write("**Similar articles:**")
                for e in exs:
                    st.caption(f"(sim {e['score']:.2f}) Topic {e['topic']} | {e['category']}")
                    st.write(e["snippet"])

    else:  # LDA
        if dictionary is None or lda is None:
            st.error("LDA artifacts not available.")
        else:
            res = predict_with_lda(text)
            st.subheader(f"Topic {res['topic_id']}")
            st.metric("Confidence", f"{res['confidence']:.2f}")
            st.write("**Top words:**", res["keywords"])

            if res.get("probs") is not None:
                probs = res["probs"]
                order = probs.argsort()[::-1][:5]
                st.bar_chart(pd.Series(probs[order], index=[f"T{int(i)}" for i in order]))
                if len(order) >= 2 and abs(probs[order[0]] - probs[order[1]]) < 0.10:
                    st.warning(f"Close call: Topic {int(order[0])} vs Topic {int(order[1])} (Δ={abs(probs[order[0]]-probs[order[1]]):.2f})")

            if (BASE / "documents_with_topics.csv").exists() and VECT_OK and DIM_OK:
                x_tf = vect.transform([preprocess_for_vect(text)])
                exs = similar_examples(x_tf, topn=3)
                if exs:
                    st.write("**Similar articles:**")
                    for e in exs:
                        st.caption(f"(sim {e['score']:.2f}) Topic {e['topic']} | {e['category']}")
                        st.write(e["snippet"])

# Footer
st.markdown("<div style='text-align: center;'>Made with ❤️ for Elevvo Internship Task 5</div>", unsafe_allow_html=True)

