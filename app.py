import nltk
import ssl
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Fix for SSL certificate issues during download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data for TextBlob/NRCLex
nltk.download('punkt_tab') # Required for tokenization
nltk.download('punkt')     # Legacy support
nltk.download('wordnet')   # For word analysis
nltk.download('omw-1.4')
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import pandas as pd
from nrclex import NRCLex
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(
    page_title="AI Emotion & Sentiment",
    page_icon="🎭",
    layout="wide"
)

# 2. Custom CSS
st.markdown("""
    <style>
    .stTextArea textarea { border: 2px solid #636efa !important; }
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3em;
        background-color: #636efa; color: white; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar
with st.sidebar:
    st.title("Settings")
    show_raw = st.checkbox("Show Raw Data", value=False)
    st.info("Engine 1: VADER (Sentiment)\nEngine 2: NRCLex (Emotions)")
    if st.button("🔄 Reset App"):
        st.rerun()

# 4. Main Header
st.title("🎭 AI Emotion & Sentiment Analysis")
st.caption("Advanced NLP Dashboard for deep text analysis")
st.markdown("---")

# 5. Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Input Text")
    text = st.text_area(
        "Enter text to analyze deep emotions:", 
        placeholder="Example: I am so surprised and happy about the news, but also a bit anxious!",
        height=200
    )
    analyze_btn = st.button("Analyze Everything ✨")

with col2:
    if analyze_btn and text.strip() != "":
        # --- ENGINE 1: VADER ---
        analyzer = SentimentIntensityAnalyzer()
        vader_scores = analyzer.polarity_scores(text)
        compound = vader_scores["compound"]

        if compound >= 0.05:
            label, emoji, bg, txt = "POSITIVE", "😊", "rgba(40, 167, 69, 0.2)", "#28a745"
        elif compound <= -0.05:
            label, emoji, bg, txt = "NEGATIVE", "😡", "rgba(220, 53, 69, 0.2)", "#dc3545"
        else:
            label, emoji, bg, txt = "NEUTRAL", "😐", "rgba(108, 117, 125, 0.2)", "#6c757d"

        # Display Result Card
        st.markdown(f"""
            <div style="background-color:{bg}; padding:25px; border-radius:15px; border: 2px solid {txt}; text-align: center;">
                <h1 style='color: {txt}; margin:0;'>{emoji} {label}</h1>
                <p style='color: {txt}; margin:0;'>Overall Sentiment Score: {compound}</p>
            </div>
            """, unsafe_allow_html=True)

        # --- ENGINE 2: NRCLex (Emotions) ---
        emotion_object = NRCLex(text)
        # Filter for actual emotions, removing general sentiment labels
        raw_emotions = emotion_object.affect_frequencies
        filtered_emotions = {k: v for k, v in raw_emotions.items() if k not in ['positive', 'negative', 'anticip', 'anticipation']}
        
        df_emo = pd.DataFrame(list(filtered_emotions.items()), columns=['Emotion', 'Score'])
        
        st.subheader("📊 Emotion Spectrum")
        fig_pie = px.pie(
            df_emo, values='Score', names='Emotion', 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    elif analyze_btn:
        st.warning("Please enter text first.")
    else:
        st.info("Results will appear here.")

# 6. Bottom Section: Visualizations
if analyze_btn and text.strip() != "":
    st.markdown("---")
    low_col1, low_col2 = st.columns(2)
    
    with low_col1:
        st.subheader("📈 Sentiment Intensity")
        df_vader = pd.DataFrame({
            "Type": ["Neg", "Neu", "Pos"],
            "Score": [vader_scores["neg"], vader_scores["neu"], vader_scores["pos"]]
        })
        fig_bar = px.bar(df_vader, x="Type", y="Score", color="Type", 
                         color_discrete_map={"Neg": "#dc3545", "Neu": "#6c757d", "Pos": "#28a745"})
        st.plotly_chart(fig_bar, use_container_width=True)

    with low_col2:
        st.subheader("☁️ Word Cloud")
        wc = WordCloud(background_color="white", width=400, height=200).generate(text)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig_wc)

    if show_raw:
        st.write("VADER Data:", vader_scores)
        st.write("Emotion Data:", filtered_emotions)