import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cv2
from PIL import Image,ImageEnhance

st.set_option("deprecation.showfileUploaderEncoding", False)
st.write("""
# Simple image processing   App
""")

#st.sidebar.header('upload image to convert')



# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input image",type=["png","jpg","jpeg"])
#st.write(uploaded_file)
#if uploaded_file is not None:
#our_image = Image.open(uploaded_file)
#input_image=cv2.imread(uploaded_file)

    
if uploaded_file is not None:
            our_image = Image.open(uploaded_file)
            st.text('Original Image')
            # st.write(type(our_image))
            st.sidebar.image(our_image,use_column_width=True)

#else:
    
        
#lower_limit = st.sidebar.slider('Minimum Gray scale value', 0,10,5)
#upper_limit = st.sidebar.slider('Maximum Gray Scale value ', 20,255,100)
#gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
#st.write(Gray_image)
#image = load_img(uploaded_file, target_size=(224, 224))
#gray_image = img_to_array(input_image)
enhance_type = st.sidebar.radio('Enhance Type', ['Original', 'Gray-Scale', 'Contrast', 'Brightness', 'Blurring'])
if enhance_type == 'Gray-Scale':
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # st.write(new_img)
            st.image(gray)

if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider('Contrast', 0.5, 3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

if enhance_type == 'Brightness':
            c_rate = st.sidebar.slider('Brightness', 0.5, 3.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

if enhance_type == 'Blurring':
            new_img = np.array(our_image.convert('RGB'))
            blur_rate = st.sidebar.slider('Blurring', 0.5, 3.5)
            img = cv2.cvtColor(new_img, 1)
            blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
            st.image(blur_img)
      
st.set_option("deprecation.showfileUploaderEncoding", False)   

