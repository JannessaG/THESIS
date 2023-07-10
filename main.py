import streamlit as st
from PIL import Image
import cv2 
import numpy as np
from io import BytesIO
import io
import subprocess
from subprocess import Popen
import os
import sys


def main():
    selected_box = st.sidebar.selectbox('Choose one of the following',('Welcome','CLAHE','Crack Detection IMAGE (YOLOV7)','Crack Detection IMAGE (YOLOV4)')
    )

    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'CLAHE':
        photo()
    if selected_box == 'Crack Detection IMAGE (YOLOV7)':
        detection_yolov7()
    if selected_box == 'Crack Detection IMAGE (YOLOV4)':
        detection_yolov4()

def welcome():
    
    st.title('Duck Egg Crack Detection')

def photo():
    st.title("Contrast Limited Adaptive Histogram Equalization")
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
   
def detection_yolov4():

    # Function to detect objects
    def detect_objects(image, confidence_threshold):
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, confidence_threshold)

        objects_detected = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                confidence = confidences[i]
                objects_detected.append((label, confidence, (x, y, w, h)))

        return objects_detected

    # Streamlit app
    st.title("YOLOv4 Object Detection")

    # Load YOLOv4-tiny model
    net = cv2.dnn.readNet("C:/Users/janne/OneDrive/Documents/SCRATCH/THESIS/yolov4weights/yolov4-tiny-custom_best.weights", "C:/Users/janne/OneDrive/Documents/SCRATCH/THESIS/yolov4weights/yolov4-tiny-custom.cfg")
    classes = []
    with open("C:/Users/janne/OneDrive/Documents/SCRATCH/THESIS/yolov4weights/obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    confidence_threshold_default = 0.5

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the image file
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
         
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, confidence_threshold_default, 0.01)
        # Detect objects in the image
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.image(image, channels="BGR", width=350)
        with col2:
            detected_objects = detect_objects(image, confidence_threshold)

            # Draw bounding boxes on the image
            for label, confidence, (x, y, w, h) in detected_objects:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)

            # Display the image with bounding boxes
            st.image(image, channels="BGR", width=350)

def detection_yolov7():

    def detect_objects_with_subprocess(image_path):
        # Replace the command and paths with your YOLOv7 setup
        process = Popen(["python","yolov7/detect.py", "--source", f"{image_path}", "--weights", "yolov7/YoloV7.pt", "--conf", "0.5"], stdout=subprocess.PIPE, shell=True)
        # Execute the command and capture the output
        output,err = process.communicate()
        if err:
            print('Error communicating with process:', err)
        else:
            print('Process output:', output, file=sys.stderr)
        process.wait()
        return output.decode()


    def get_detected_image_path():
        # Get the path to the detected image
        detect_folder = os.path.join("runs", "detect")
        exp_folders = [f for f in os.listdir(detect_folder) if os.path.isdir(os.path.join(detect_folder, f))]
        if exp_folders:
            exp_folder = max(exp_folders)  # Get the experiment folder with the highest number
            detected_image_path = os.path.join(detect_folder, exp_folder, "temp_image.jpg")
            return detected_image_path
        return None


    
    st.title("YOLOv7 Object Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        col1, col2 = st.columns( [0.5, 0.5])
        with col1:
            st.image(image, channels="BGR", caption="Original Image",width=350)

            # Save the uploaded image to a temporary file
            temp_image_path = "temp_image.jpg"
            cv2.imwrite(temp_image_path, image)

            # Perform object detection using subprocess
            detected_objects = detect_objects_with_subprocess(temp_image_path)

            # Get the path to the detected image
            detected_image_path = get_detected_image_path()
        with col2:
            # Display the detected image
            if detected_image_path and os.path.exists(detected_image_path):
                detected_image = cv2.imread(detected_image_path)
                st.image(detected_image, channels="BGR", caption="Detected Objects",width=350)
            else:
                st.warning("Object detection failed.")

            # Clean up the detected image and temporary image file
            if detected_image_path:
                os.remove(detected_image_path)
            os.remove(temp_image_path)
            
if __name__ == "__main__":
    main()