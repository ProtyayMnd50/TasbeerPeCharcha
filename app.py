import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

# Set page layout to wide
st.set_page_config(layout="wide",page_title="TasbeerPeCharcha")

st.title("Visual Question Answering")
st.write("Upload an image and enter a question to get an answer.")

col1,col2 =st.columns(2)

##VILT code
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Image upload
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    st.image(uploaded_file, use_column_width=True)

# Question input
with col2:
    question = st.text_input("Question")

    # Process the image and question when both are provided
    if uploaded_file and question is not None:
        if st.button("Ask Question"):
           pass