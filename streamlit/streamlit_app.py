### Code to develop a streamlit app
### Using the model I made in CNN for 3D prints
import streamlit as st
import preprocessImg as pImg
import torch
import torchvision.models as models
import pyTorchModel as py
import os

st.title("3D Printing Issue Solver")

def predictImage(processedImage):

    class_names = {0: 'good', 1: 'spaghetti', 2: 'stringing', 3: 'underextrusion'}


    # Get the parent directory of the current directory (streamlit)
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model weights in the CNNBuilding folder
    model_weights_path = os.path.join(parent_dir, "..", "CNNBuilding", "models_in_folder", "CNNModelV0_2.pth")


    model = py.pyTorchModel()
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    model.eval()
    output = model(processedImage)
    
    predicted_index = output.argmax(1).item()
    predicted_class_name = class_names[predicted_index]
    
    return predicted_class_name



st.write("Hi there! Welcome to the 3D Print App, where we try to solve your 3D Printing issues")

uploaded_image = st.file_uploader("Upload Image Here")

if uploaded_image is not None:
    st.write("Image you have uploaded is below:")
    st.image(uploaded_image)
    processedImage = pImg.preProcess(uploaded_image, width= 100)

    prediction = predictImage(processedImage)
    st.write(f"The predicted problem is {prediction} ")
else:
    st.write("No image yet")