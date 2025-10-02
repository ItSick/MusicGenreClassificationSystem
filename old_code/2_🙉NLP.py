import streamlit as st
import pandas as pd

data = pd.read_csv("./brain_stroke.csv")

st.title("NLP")
st.write(data)


