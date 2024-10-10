import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

st.title("Visual Question Answering")
st.write("Upload an image and enter a question to get an answer.")