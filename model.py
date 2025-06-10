import streamlit as st

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.eval()  # set ke mode eval kalau untuk inference
    return model
