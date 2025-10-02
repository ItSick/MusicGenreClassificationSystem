import streamlit as st
import time as time
import pandas as pd
from plotly import express as px
import joblib
import streamlit as st
from PIL import Image


st.balloons()
st.write("### Data Sience + ML 2024-2025")

# --- CONFIGURATION ---
# Set the page configuration for a wide, attractive layout
st.set_page_config(
    page_title="עבודת הגשה ML",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Replace with your actual data
NAME = "איציק אדרי"
TITLE = "מהנדס תוכנה ומוסיקאי"
DESCRIPTION = """
:על הפרוייקט
<br>
אימון מודל לזהות לאיזה ג'אנר שייך השיר 
"""
EMAIL = "salesmen1@gmail.com"


# --- STYLING ---
# Using Markdown for simple, elegant styling
st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
}
.header-container {
    padding: 30px 0 10px 0;
}
.stMarkdown h1 {
    font-size: 3.5rem;
    font-weight: 700;
    color: #333333;
}
.stMarkdown h2 {
    font-size: 1.8rem;
    color: #555555;
    margin-top: -10px;
}
.stMarkdown h3 {
    color: #007BFF;
    border-bottom: 2px solid #007BFF;
    padding-bottom: 5px;
    margin-top: 30px;
}
.stProgress > div > div > div > div {
    background-color: #007BFF;
}
.profile-description {
    font-size: 1.1rem;
    line-height: 1.6;
    color: #4a4a4a;
    float: right;
}
.stExpander {
    border: 1px solid #dcdcdc;
    border-radius: 8px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}
.stExpander:hover {
    box-shadow: 4px 4px 12px rgba(0,0,0,0.1);
}

/* Ensure wide mode content is nicely centered and spaced */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5rem;
    padding-right: 5rem;
}
</style>
""", unsafe_allow_html=True)


# --- HEADER & INTRODUCTION ---
st.markdown('<div class="header-container"></div>', unsafe_allow_html=True)

# Use columns for a clear separation of profile picture (or icon) and text
col1, col2 = st.columns([1, 3])

with col1:
    # A simple, large emoji as a placeholder for a profile picture
    # If you have a picture, use: Image.open("your_picture.png")
    st.markdown("<p style='font-size: 100px; text-align: center;'><img src=''/></p>", unsafe_allow_html=True)

with col2:
    st.markdown(f'<p class="profile-description">{NAME}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="profile-description">{TITLE}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="profile-description">{DESCRIPTION}</p>', unsafe_allow_html=True)

st.write("---") # Separator line

