import streamlit as st
from PIL import Image
import cv2 
import numpy as np
from io import BytesIO
import io
# from io import StringIO

def main():
    selected_box = st.sidebar.selectbox('Choose one of the following',('Welcome','CLAHE','Object Detection IMAGE')
    )

    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'CLAHE':
        photo()
    if selected_box == 'Object Detection IMAGE':
        objectdetection()


def welcome():
    
    st.title('Duck Egg Crack Detection')

def photo():

    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        image = original_image.rotate(0)
        
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
            st.image(image,width=300)

        with col2:
            st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)
            filter = st.sidebar.radio('Covert your photo to:', ['Original', 
            # 'CLAHE colored' , 
            'CLAHE']) #Add the filter in the sidebar
            
            # if filter == 'CLAHE colored':
            #     converted_img = np.array(image.convert('RGB'))
            #     lab_img= cv2.cvtColor(converted_img,cv2.COLOR_BGR2LAB)
            #     l, a, b = cv2.split(lab_img)
            #     clahe= cv2.createCLAHE(clipLimit=5) 
            #     clahe_img= clahe.apply(l) 
            #     updated_lab_img2 = cv2.merge((clahe_img,a,b))
            #     CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
                
            #     st.image(CLAHE_img,width=400)

            if filter == 'CLAHE':
                converted_img = np.array(image.convert('RGB'))
                image_bw = cv2.cvtColor(converted_img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=5)
                final_img = clahe.apply(image_bw) 
                st.image(final_img,width=300)
                result_img = Image.fromarray(final_img)
                buffer = io.BytesIO()
                result_img.save(buffer, 'JPEG')
                buffer.seek(0)
                st.download_button(label="Download Clahe Image", data=buffer, file_name="CLAHE_image.jpg", mime='image/jpeg')
                # processed_image = Image.fromarray(final_img)

                # processed_image.save("processed_claheimage.jpg")
            else: 
                # st.image(image, width=300)
                print("wala")

def objectdetection():
    st.title('Object Detection with CLAHE')


    uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

    if uploaded_file is not None:
        processed_image = Image.open(uploaded_file)

        st.image(processed_image,width=400)
        st.button('Detect')

        filter = st.sidebar.radio('Detect Image Using:', ['Yolov7','Yolov4','Faster R-CNN','SSD', 'Retina-net'])


if __name__ == "__main__":
    main()