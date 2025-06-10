import streamlit as st
import torch

@st.cache_resource
def load_model():
    model = torch.load('best.pt', map_location=torch.device('cpu'))
    model.eval()  # set ke mode eval kalau untuk inference
    return model
