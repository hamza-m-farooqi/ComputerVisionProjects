import os
import cv2
import numpy as np
import tensorflow as tf


# Reading All Files at once
model=tf.keras.models.load_model("HandwrittenDigitRecognizer.model")
Image_Number=0
while os.path.isfile(f"C:\Hamza\Python\Personal\Code\ComputerVision\HandwrittenDigitsRecognition\DigitsSet_DrawnWithPaint\{Image_Number}.png"):
    try:
        Image_File=cv2.imread(f"C:\Hamza\Python\Personal\Code\ComputerVision\HandwrittenDigitsRecognition\DigitsSet_DrawnWithPaint\{Image_Number}.png")[:,:,0]
        Image_Width, Image_Height = Image_File.shape[:2]
        Image_File= cv2.resize(Image_File, (Image_Width//2, Image_Height//2))
        Image_File=np.invert(np.array([Image_File]))
        Image_Prediction=model.predict(Image_File)
        print(f"Image Number {Image_Number} is predicted as {np.argmax(Image_Prediction)}")
    except:
        print("Some Error Occurred")
    finally:
        Image_Number +=1