### Code to develop a streamlit app
### Using the model I made in CNN for 3D prints
import streamlit as st
import preprocessImg as pImg

st.write("Hi there! Welcome to the 3D Print App, where we try to solve your 3D Printing issues")

uploaded_image = st.file_uploader("Upload Image Here")

st.write("Image you have uploaded is below:")
st.image(uploaded_image)

processedImage = pImg.preProcess(uploaded_image)


