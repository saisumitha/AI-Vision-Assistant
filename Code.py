import streamlit as st
import datetime
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
import os , random 
from PIL import Image
from matplotlib import pyplot as plt 
HUGGINGFACEHUB_API_TOKEN  =   "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"
from transformers import ViTImageProcessor, ViTForImageClassification
import requests
import torch
from transformers import DetrImageProcessor
from transformers import DetrForObjectDetection
from PIL import ImageDraw


# Load the pdf files from the path
loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})

# Vector store
vector_store = FAISS.from_documents(text_chunks, embeddings)

# Create llm
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
                     repo_id=repo_id,
                     model_kwargs={"temperature": 0.5, "max_new_tokens": 500})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Your Query", key='input')
            submit_button = st.form_submit_button(label='Send')

        if user_input:  # Check if there is user input before processing and displaying
            output = conversation_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="adventurer")


# Tab 3 content (Image Classifier)
def Image_content():

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Process the uploaded image
        image = Image.open(uploaded_file)
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        # Display the uploaded image
        st.image(image, caption=f"Classified as: {predicted_class}", use_column_width=True)


# Tab 4 content (Object Detector)
def Object_content():

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="unique_key")

    if uploaded_file is not None:
        # Process the uploaded image
        image = Image.open(uploaded_file)

        # Object detection model
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Display the uploaded image with bounding boxes
        st.image(image, caption="Detected Objects", use_column_width=True)

        # Display bounding boxes and labels
        for i in range(len(results['boxes'])):
            x_min, y_min, x_max, y_max = results['boxes'][i]
            label = results['labels'][i]
            text11 = model.config.id2label[label.item()]

            # Create a drawing object
            draw = ImageDraw.Draw(image)

            # Draw the bounding box on the image
            draw.text((x_min, y_min), text11, fill="red")
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)

        # Display the modified image with bounding boxes in Streamlit
        st.image(image, caption="Modified Image with Bounding Boxes", use_column_width=True)

        # Display text output
        st.write("Detected Objects:")
        for i in range(len(results['boxes'])):
            st.write(f"Detected {text11} with confidence {round(results['scores'][i].item(), 3)} at location {x_min, y_min, x_max, y_max}")

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]

initialize_session_state()

# Add a title and a short description
st.title("TEXTQUEST")
st.write("TEXTQUEST is an interactive chatbot application.")

# Create tabs
tabs = st.tabs(["Home", "ChatBot", "Image Classifier", "Object Detecter"])

# Tab 1 content (Home)
with tabs[0]:
    st.title("Welcome to TEXTQUEST!")
    st.write("TEXTQUEST is an interactive chatbot application.")
    # Add any additional information or features about your application here

# Tab 2 content (ChatBot)
with tabs[1]:
    st.title("ChatBot ")
    # Call your chatbot functions here
    display_chat_history()

# Tab 3 content (Image Classifier)
with tabs[2]:
    st.title("Image Classifier ")
    # Call your image classifier functions here
    Image_content()

# Tab 4 content (Object Detector)
with tabs[3]:
    st.title("Object Detector ")
    # Call your object detector functions here
    Object_content()



