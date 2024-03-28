# AI-Vision-Assistant
Introduction
TEXTQUEST is an interactive application that combines a chatbot functionality with image classification and object detection features. This README provides instructions on how to install, use, and contribute to the application.

Installation
Clone the repository:

    git clone <repository_url>

Install the required dependencies:
  
    pip install -r requirements.txt
    
Ensure that the required models and data files are available. Specifically, make sure you have the PDF documents in the specified directory for the chatbot module and the pretrained models for image classification and object detection.

Usage
Run the application:

    streamlit run Code.py
    
Once the application starts, you will see a navigation bar with different tabs: Home, ChatBot, Image Classifier, and Object Detector.

Home Tab: Provides a brief introduction to the application.

ChatBot Tab: Interact with the chatbot by typing your questions in the input box and clicking "Send". The chat history will be displayed along with the bot's responses.

Image Classifier Tab: Upload an image file (JPEG or PNG) and the app will classify the main object in the image.

Object Detector Tab: Upload an image file (JPEG or PNG) and the app will detect and highlight objects present in the image.

Functionality
ChatBot: The chatbot utilizes a conversational retrieval chain for answering user queries. It maintains a chat history and provides responses based on the context of the conversation.

Image Classifier: The app uses a Vision Transformer (ViT) model for image classification. After uploading an image, it predicts the main object present in the image along with a confidence score.

Object Detector: The app employs a Detection Transformer (DETR) model for object detection. It identifies objects in the uploaded image and draws bounding boxes around them, along with labels and confidence scores.

