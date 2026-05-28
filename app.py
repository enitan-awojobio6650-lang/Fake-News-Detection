import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FakeShield — Fake News Detector",
    page_icon="🛡️",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

* { font-family: 'Inter', sans-serif; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f;
    color: #f0f0f0;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at top, #1a1040 0%, #0a0a0f 60%);
    min-height: 100vh;
}

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.1;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 2.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

.badge {
    display: inline-block;
    background: rgba(167, 139, 250, 0.1);
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #a78bfa;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    backdrop-filter: blur(10px);
}

.result-fake {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-real {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(16,185,129,0.05));
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-icon {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
}

.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.result-fake .result-label { color: #f87171; }
.result-real .result-label { color: #34d399; }

.confidence-bar-container {
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    height: 8px;
    margin: 1rem auto;
    max-width: 300px;
    overflow: hidden;
}

.confidence-bar-fake {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ef4444, #f87171);
    transition: width 0.8s ease;
}

.confidence-bar-real {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 0.8s ease;
}

.confidence-text {
    color: #9ca3af;
    font-size: 0.9rem;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.stat-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #a78bfa;
}

.stat-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.how-it-works {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 2rem;
}

.step {
    display: flex;
    align-items: flex-start;
    margin-bottom: 1rem;
    gap: 1rem;
}

.step-num {
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    color: white;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    flex-shrink: 0;
    font-family: 'Syne', sans-serif;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #f0f0f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    min-height: 160px !important;
}

.stTextArea textarea:focus {
    border-color: rgba(167,139,250,0.5) !important;
    box-shadow: 0 0 0 2px rgba(167,139,250,0.1) !important;
}

.stButton button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    letter-spacing: 0.03em !important;
}

.stButton button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

footer { display: none; }
#MainMenu { display: none; }
header { display: none; }

.footer-note {
    text-align: center;
    color: #374151;
    font-size: 0.75rem;
    margin-top: 3rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load model and vectoriser ────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()


# ── Text cleaning (same as notebook) ────────────────────────
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


# ── Header ───────────────────────────────────────────────────
st.markdown('<div style="text-align:center"><span class="badge">🛡️ AI Powered</span></div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">FakeShield</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fake News Detection — TechCrush AI/ML Bootcamp Capstone</p>', unsafe_allow_html=True)

# ── Stats row ────────────────────────────────────────────────
st.markdown("""
<div class="stat-grid">
    <div class="stat-box">
        <div class="stat-number">72,134</div>
        <div class="stat-label">Articles Trained On</div>
    </div>
    <div class="stat-box">
        <div class="stat-number">3</div>
        <div class="stat-label">Models Built</div>
    </div>
    <div class="stat-box">
        <div class="stat-number">WELFake</div>
        <div class="stat-label">Dataset</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input area ───────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📰 Paste your article below")
article = st.text_area(
    label="Article text",
    placeholder="Paste a news article headline or full body text here...",
    height=180,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyse = st.button("🔍  Analyse Article")

st.markdown('</div>', unsafe_allow_html=True)


# ── Prediction ───────────────────────────────────────────────
if analyse:
    if not article.strip():
        st.warning("Please paste an article before clicking Analyse.")
    elif len(article.split()) < 5:
        st.warning("Please enter a longer article — at least a few sentences for a reliable result.")
    else:
        with st.spinner("Analysing..."):
            cleaned = clean_text(article)
            vectorised = tfidf.transform([cleaned])
            prediction = model.predict(vectorised)[0]
            probability = model.predict_proba(vectorised)[0]

            fake_prob = round(probability[0] * 100, 1)
            real_prob = round(probability[1] * 100, 1)
            confidence = max(fake_prob, real_prob)

        if prediction == 0:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-icon">🚨</div>
                <div class="result-label">FAKE NEWS DETECTED</div>
                <p style="color:#fca5a5; margin:0.3rem 0">The model flagged this article as likely fake.</p>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-fake" style="width:{fake_prob}%"></div>
                </div>
                <p class="confidence-text">{confidence}% confidence</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-real">
                <div class="result-icon">✅</div>
                <div class="result-label">REAL NEWS</div>
                <p style="color:#6ee7b7; margin:0.3rem 0">The model classified this article as likely real.</p>
                <div class="confidence-bar-container">
                    <div class="confidence-bar-real" style="width:{real_prob}%"></div>
                </div>
                <p class="confidence-text">{confidence}% confidence</p>
            </div>
            """, unsafe_allow_html=True)

        # Breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("🔴 Fake Probability", f"{fake_prob}%")
        with c2:
            st.metric("🟢 Real Probability", f"{real_prob}%")


# ── How it works ─────────────────────────────────────────────
with st.expander("⚙️ How does this work?"):
    st.markdown("""
    <div class="how-it-works">
        <div class="step">
            <div class="step-num">1</div>
            <div><strong>Text Cleaning</strong> — Your article is lowercased, URLs and punctuation removed, stopwords filtered out, and words reduced to their root forms using stemming.</div>
        </div>
        <div class="step">
            <div class="step-num">2</div>
            <div><strong>TF-IDF Vectorisation</strong> — The cleaned text is converted into a numerical vector of 100,000 features, weighting words that are distinctive to fake or real news.</div>
        </div>
        <div class="step">
            <div class="step-num">3</div>
            <div><strong>Logistic Regression</strong> — The trained model analyses the word weights and outputs a probability score for fake vs real.</div>
        </div>
        <div class="step">
            <div class="step-num">4</div>
            <div><strong>Result</strong> — The class with the higher probability is returned as the prediction along with a confidence percentage.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("📊 About the Model"):
    st.markdown("""
    | Detail | Value |
    |--------|-------|
    | Algorithm | Logistic Regression |
    | Training Data | WELFake Dataset (Verma et al., 2021) |
    | Dataset Size | 72,134 news articles |
    | Features | TF-IDF (100k features, bigrams) |
    | Classes | Fake (0) / Real (1) |
    | Source | Kaggle — saurabhshahane/fake-news-classification |
    """)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<p class="footer-note">
    TechCrush AI/ML Bootcamp — Cohort 5 Capstone Project<br>
    Built with Streamlit · Model trained on WELFake Dataset · Verma et al. (2021)
</p>
""", unsafe_allow_html=True)
