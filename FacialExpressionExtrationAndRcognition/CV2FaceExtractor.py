import cv2
import os
from config import (
    BASE_DIRECTORY_FOR_INPUT_VIDEOS,
    BASE_DIRECTORY_FOR_OUTPUT,
    BASE_DIRECTORY_FOR_Extracted_Frames,
    SAMPLE_VIDEOS,
    BASE_DIRECTORY_FOR_CV2_FRONTAL_FACE_HAAR_CLASSIFIER,
    FRONTAL_FACE_HAAR_CLASSIFIERS,
)


def detect_faces(imageName, imagePath, outputPath, classifierName):
    os.makedirs(outputPath, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + classifierName)
    image = cv2.imread(imagePath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Save the face regions as separate images
    for i, (x, y, w, h) in enumerate(faces):
        face_region = image[y : y + h, x : x + w]
        face_image_name =f"_face_{i}_in_"+ imageName + "_" + f".jpg"
        outputPath = os.path.join(outputPath, face_image_name)
        print("Face Saved at ",face_image_name)
        cv2.imwrite(outputPath, face_region)


# Example usage

# InputImage = BASE_DIRECTORY_FOR_Extracted_Frames + "man-explaining-1/frame_0.jpg"
# OutputPath = BASE_DIRECTORY_FOR_OUTPUT + "ExtractedFaces/man-explaining-1/frame_0/"
# detect_faces(InputImage, OutputPath, FRONTAL_FACE_HAAR_CLASSIFIERS[0])
