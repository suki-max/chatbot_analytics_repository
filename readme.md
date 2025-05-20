# Chatbot Analytics Dashboard

This project is a **Streamlit-based analytics dashboard** for chatbot interactions. It allows logging user queries, auto-classifying topics using a machine learning model, recording satisfaction scores, and visualizing usage patterns through charts and metrics.

---

## Features

-  Log chatbot queries along with satisfaction scores (1–5)
-  Auto-detect query topics using a trained ML model (`scikit-learn`)
-  Save and view logged interactions in a CSV file
-  Visualize:
  - Topic distribution (Pie Chart)
  - Satisfaction histogram
  - Top 5 most common topics (Bar Chart)
-  Download interaction data as CSV
-  Real-time analytics dashboard built with `Streamlit` and `Plotly`

---

## Machine Learning Overview

- **Model Used:** Logistic Regression (`scikit-learn`)
- **Text Vectorizer:** `TfidfVectorizer`
- **Purpose:** Classify user queries into pre-defined topics
- **Fallback:** Keyword-based topic detection if model fails or isn’t available
- **Model File:** `topic_model.pkl`

---

## Project Structure

