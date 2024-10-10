import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

# Set page layout to wide
st.set_page_config(layout="wide",page_title="TasbeerPeCharcha")

##VILT code
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def get_answer(image, text):
    try:
        # Load and process the image
        img = Image.open(BytesIO(image)).convert("RGB")

        # Prepare inputs
        encoding = processor(img, text, return_tensors="pt")

        # Forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]

        return answer

    except Exception as e:
        return str(e)


st.title("Visual Question Answering")
st.write("Upload an image and enter a question to get an answer.")





col1,col2 =st.columns(2)

# Image upload
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    # st.image(uploaded_file, use_column_width=True)

# modified section if question not asked
    if uploaded_file is not None:  # Check if a file is uploaded
        st.image(uploaded_file, use_column_width=True)
    else:
        st.write("Please upload an image to continue.")

# Question input
with col2:
    question = st.text_input("Question")

    # Process the image and question when both are provided
    if uploaded_file and question is not None:
        if st.button("Ask Question"):
            image = Image.open(uploaded_file)
            image_byte_array = BytesIO()
            image.save(image_byte_array, format='JPEG')
            image_bytes = image_byte_array.getvalue() 

             # Get the answer from the model
            answer = get_answer(image_bytes, question)

            st.info("Your question:"+question)
            st.success("Answer: "+answer)
             