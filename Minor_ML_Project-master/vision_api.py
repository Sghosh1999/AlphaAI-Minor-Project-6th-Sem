#Updated on 14/01/ 3:43a.m

import streamlit as st
import cv2
from PIL import Image
import numpy as np
import sys
import os
import pytesseract

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'

def face_detection(img_array):
    gray_1 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray_2 = cv2.cvtColor(gray_1, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_2, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(gray_1, (x, y), (x+w, y+h), (255, 0, 0), 2)
    gray_1 = cv2.cvtColor(gray_1, cv2.COLOR_BGR2RGB)
    return gray_1

def Canny_Edge(img_array):
    blur_img = cv2.Canny(img_array,100,200)
    return blur_img




def main():
    summr = st.selectbox("Choose the summarizer",("Face Detection","Canny edge Detection","OCR"))
    uploaded_file = st.file_uploader("Upload Image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Input', use_column_width=True)
        img_array = np.array(image)
    
        if summr == 'Face Detection':
            gray_1 = face_detection(img_array)
            if st.button('Process'):
                st.image(gray_1, caption='Uploaded Image.', use_column_width=True)
        if summr == 'Canny edge Detection':
            blurred_img = Canny_Edge(img_array)
            
            if st.button('Process'):
                st.image(blurred_img, caption='Uploaded Image.', use_column_width=True)
        
        if summr == 'OCR':
            config = ('-l eng --oem 1 --psm 3')
            if st.button('Recognize text'):
                text = pytesseract.image_to_string(img_array, config=config)
                st.success('Recognized text')
                st.info(text)

                f = open('file.txt','w')
                f.write(str(text))
                f.close()
                
    

    

    
    
    

        
    
    



if __name__ == '__main__':
    main()
