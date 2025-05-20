import streamlit as st
import pandas as pd
import os
import joblib
from collections import Counter
import plotly.express as px

# MUST be the very first Streamlit command
st.set_page_config(page_title="Chatbot Analytics Dashboard", layout="wide")

# Constants
CSV_FILE = 'dataset.csv'
MODEL_FILE = 'topic_model.pkl'

# Load or create CSV file
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["query", "topic", "satisfaction"]).to_csv(CSV_FILE, index=False)

# Load model if exists
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = None

# Helper functions
def load_data():
    try:
        return pd.read_csv(CSV_FILE)
    except Exception:
        return pd.DataFrame(columns=["query", "topic", "satisfaction"])

def save_interaction(query, topic, satisfaction):
    df = load_data()
    new_entry = pd.DataFrame([{
        "query": query.strip(),
        "topic": topic.strip(),
        "satisfaction": int(satisfaction)
    }])
    updated_df = pd.concat([df, new_entry], ignore_index=True)
    updated_df.to_csv(CSV_FILE, index=False)

def infer_topic_from_query(query):
    if model:
        try:
            pred = model.predict([query])
            return pred[0]
        except Exception:
            pass
    # fallback simple keyword heuristic
    query = query.lower()
    if any(word in query for word in ["bill", "payment"]):
        return "Billing"
    elif any(word in query for word in ["return", "refund"]):
        return "Returns"
    elif any(word in query for word in ["support", "issue", "error"]):
        return "Technical Support"
    elif any(word in query for word in ["product", "feature", "info"]):
        return "Product Info"
    return "Other"

def analyze_data(df):
    total_queries = len(df)
    most_common_topics = []
    avg_satisfaction = 0
    if not df.empty:
        if 'topic' in df.columns:
            most_common_topics = Counter(df['topic']).most_common(5)
        if 'satisfaction' in df.columns:
            df['satisfaction'] = pd.to_numeric(df['satisfaction'], errors='coerce')
            avg_satisfaction = df['satisfaction'].mean()
    return df, total_queries, most_common_topics, avg_satisfaction

# UI
st.title("📊 Chatbot User Analytics")

# Sidebar: Log Interaction
st.sidebar.header("📝 Log Interaction")
query = st.sidebar.text_area("User Query")
custom_topic = st.sidebar.text_input("Custom Topic", placeholder="Enter topic (e.g., Billing Issue)")
default_topics = ["Technical Support", "Product Info", "Billing", "Returns", "Other"]
topic_dropdown = st.sidebar.selectbox("Or select a topic", default_topics)
satisfaction = st.sidebar.slider("Satisfaction (1-5)", 1, 5)

if st.sidebar.button("Submit Interaction"):
    if query.strip():
        if custom_topic.strip():
            topic = custom_topic.strip()
        else:
            topic = infer_topic_from_query(query)
            st.sidebar.info(f"🔍 Topic auto-detected as: **{topic}**")
        save_interaction(query, topic, satisfaction)
        st.sidebar.success("✅ Interaction saved!")
        # Reload data after saving to update charts immediately
        df = load_data()
        df, total_queries, most_common_topics, avg_satisfaction = analyze_data(df)
    else:
        st.sidebar.error("❌ Please enter a valid query.")
else:
    # If no new submission, just load data once
    df = load_data()
    df, total_queries, most_common_topics, avg_satisfaction = analyze_data(df)

st.header("📈 Summary Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Queries", total_queries)
col2.metric("Avg. Satisfaction", f"{avg_satisfaction:.2f} / 5" if total_queries else "N/A")

if not df.empty:
    st.subheader("🧠 Topic Distribution")
    topic_counts = df['topic'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig_pie = px.pie(topic_counts, names='Topic', values='Count', title='Topic Distribution')
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("📊 Satisfaction Ratings Distribution")
    fig_hist = px.histogram(df, x='satisfaction', nbins=5, title='Satisfaction Ratings Histogram',
                            labels={'satisfaction': 'Satisfaction Rating'})
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("📌 Top 5 Topics")
    top_topics_df = pd.DataFrame(most_common_topics, columns=["Topic", "Count"])
    st.bar_chart(top_topics_df.set_index("Topic"))

    st.download_button(
        label="⬇️ Download Interactions as CSV",
        data=df.to_csv(index=False),
        file_name="interactions_export.csv",
        mime="text/csv"
    )

st.subheader("📂 All Logged Interactions")
if not df.empty:
    st.dataframe(df, use_container_width=True)
else:
    st.info("No interactions logged yet.")
