import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

st.write("""
# Simple image processing   App

This app predicts converts the colour image to gray image of specified scale


""")

#st.sidebar.header('upload image to convert')



# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input image",type=["png","jpg","jpeg"])
#st.write(uploaded_file)
#if uploaded_file is not None:
input_image = Image.open(uploaded_file)
#input_image=cv2.imread(uploaded_file)
st.sidebar.image(input_image,use_column_width=True)
    
    

#else:
    
        
lower_limit = st.sidebar.slider('Minimum Gray scale value', 0,10,5)
upper_limit = st.sidebar.slider('Maximum Gray Scale value ', 20,255,100)
#gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
#st.write(Gray_image)
#image = load_img(uploaded_file, target_size=(224, 224))
#gray_image = img_to_array(input_image)

new_img = np.array(input_image.convert('RGB'))
img = cv2.cvtColor(new_img, 1)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
st.write("gray_image")
st.image(gray_image)
st.write("Gray_image_filtered")
colour_filtered = cv2.inRange(gray_image, lower_limit, upper_limit)

           
st.image(colour_filtered)
