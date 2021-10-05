  # Importing required libraries, obviously
import logging
import logging.handlers
import threading
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
from typing import Union
import time


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
    

# Loading pre-trained parameters for the cascade classifier
try:
    faceCascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml') # Face Detection
    model =load_model('FacialExpressionModel.h5')  #Load model
    Classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] # Emotion that will be predicted
except Exception:
    st.write("Error loading cascade classifiers")
    
    
class VideoTransformer(VideoTransformerBase):

    def transform(self, frame):
        label=[]
        img = frame.to_ndarray(format="bgr24")
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        Classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3,1)
        for (x,y,w,h) in faces:
                    a=cv2.rectangle(in_image,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_gray = gray[y:y+h,x:x+w]
                    roi_gray = cv2.resize(roi_gray,(img_size,img_size))  ##Face Cropping for prediction
                    facess=faceCascade.detectMultiScale(roi_gray)
                    if np.sum([roi_gray])!=0:
                        roi_color=frame[y:y+h,x:x+w]
                        final_image=cv2.resize(roi_color,(img_size,img_size))
                        final_image=np.expand_dims(final_image,axis=0)
                        final_image=final_image/255
                        predict=model.predict(final_image)
                        label=Classes[np.argmax(predict)]
                        label_position = (x,y)
                        b=cv2.putText(a,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)   # Text Adding
                        st.image(b,channels="BGR")
                    else:
                        b=cv2.putText(a,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        st.image(b,channels="BGR")

        return b
    

def face_detect():
    class VideoTransformer(VideoTransformerBase):
        frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
        in_image: Union[np.ndarray, None]
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None
            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            in_image = frame.to_ndarray(format="bgr24")
            out_image = in_image[:, ::-1, :]  # Simple flipping for example.

            with self.frame_lock:
                self.in_image = in_image
                self.out_image = out_image

            return in_image

    ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)
    x = 0
    p = st.empty()
    while ctx.video_transformer and x<1000000:
        
        
            with ctx.video_transformer.frame_lock:
                in_image = ctx.video_transformer.in_image
                out_image = ctx.video_transformer.out_image
                frame = in_image
                img_size = 224

            if in_image is not None :
                gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray)
                a = np.zeros([480,640,4], dtype=np.uint8)
                for (x,y,w,h) in faces:
                    a=cv2.rectangle(in_image,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_gray = gray[y:y+h,x:x+w]
                    roi_gray = cv2.resize(roi_gray,(img_size,img_size))  ##Face Cropping for prediction
                    facess=faceCascade.detectMultiScale(roi_gray)
                    if np.sum([roi_gray])!=0:
                        roi_color=frame[y:y+h,x:x+w]
                        final_image=cv2.resize(roi_color,(img_size,img_size))
                        final_image=np.expand_dims(final_image,axis=0)
                        final_image=final_image/255
                        predict=model.predict(final_image)
                        label=Classes[np.argmax(predict)]
                        label_position = (x,y)
                        b=cv2.putText(a,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)   # Text Adding
                        p.image(b,channels="BGR",width=None)
                        # p.header(label)
                        time.sleep(0.01)
                        x += 1
                    # else:
                    #     b=cv2.putText(a,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    #     st.image(b,channels="BGR")
 

def main():
    
    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:green ;padding:10px">
    <h2 style="color:white;text-align:center;">Real Time Face Emotion Recognisation app</h2>
    <style>#"This app is created by Sibani choudhury" {text-align: center}</style>
    </div>
    </body>
    """
  

    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("This app is created by Ashutosh Kumar")
    st.write("Model built by transfer learning from EfficientNetB2")
    st.write("**Instructions while using the APP**")
    st.write('''
                
                1. Click on the Start button to start.
                
                2. Allow the webcam access and WebCam window will open afterwards. 
        
                3. It will load the realtime face emotion detection block with the prediction.
                
                4. Click on  Stop  to end.
                
                5. Still webcam window didnot open, then follow the following step in chrome:
                   1) Navigate via address-bar to chrome://flags/#unsafely-treat-insecure-origin-as-secure in Chrome.
                   2) Find and enable the Insecure origins treated as secure section.
                   3) Add the streamlit web link addresses so to ignore the secure origin policy for. (Include the port number if required.)
                   4) Save and restart Chrome and reload the link and enjoy your emotion.''')
    
    face_detect()
    
    
if __name__ == "__main__":
    main()
