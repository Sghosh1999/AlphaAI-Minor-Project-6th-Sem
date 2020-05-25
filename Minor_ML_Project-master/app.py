import streamlit as st
import dataset_analysis
import geo_climate
import text_summ
import vision_api
import classification
import regression
import newsclass
from PIL import Image


def main():
    # Title
    st.title("AlphaAI")

    # Sidebar
    activities = ["Home", "Dataset Explorer", "ML Classifiers", "ML Regression", "News Classification", "Text Summarizer", "Real World Data Distribution", "Vision API"]
    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice == "Home":
        st.header('Empowering companies to jumpstart AI and generate real-world value')
        st.subheader('Use exponential technologies to your advantage and lead your industry with confidence through innovation.')

        
        image = Image.open('images/img0.jpg')
        st.image(image, use_column_width=True, caption='Data Mining')

    if choice == "Dataset Explorer":
        st.subheader("Dataset Explorer")
        dataset_analysis.main()
    if choice == "Real World Data Distribution":
        geo_climate.main()
    if choice == "ML Regression":
        regression.main()
    if choice == "ML Classifiers":
        classification.main()
    if choice == "Vision API":
        vision_api.main()
    if choice == "Text Summarizer":
        text_summ.main()
    if choice == "News Classification":
        newsclass.main()



if __name__ == '__main__':
    main()
