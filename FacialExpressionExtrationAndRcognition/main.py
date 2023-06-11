from config import (
    BASE_DIRECTORY_FOR_INPUT_VIDEOS,
    BASE_DIRECTORY_FOR_Extracted_Frames,
    BASE_DIRECTORY_FOR_Extracted_Faces,
    SAMPLE_VIDEOS,
    FRONTAL_FACE_HAAR_CLASSIFIERS,
    BASE_DIRECTORY_FOR_LANDMARKED_Faces,
)
from CV2FaceExtractor import detect_faces
from VideoFramesExtractor import extract_frames
from MediapipeFacialLandmarksExtractor import detect_face_landmarks


for video in SAMPLE_VIDEOS:
    print("Going to Fetch Frames for " + video)
    videoName = str(video).split(".")[0]
    extracted_frames_info = extract_frames(
        videoName,
        BASE_DIRECTORY_FOR_INPUT_VIDEOS + video,
        BASE_DIRECTORY_FOR_Extracted_Frames,
    )
    print("--------------------------------")
    for FRONTAL_FACE_HAAR_CLASSIFIER in FRONTAL_FACE_HAAR_CLASSIFIERS:
        print("Going to Extract Face using ",FRONTAL_FACE_HAAR_CLASSIFIER)
        extracted_faces_info=[]
        for frame in extracted_frames_info:
            extracted_face_output_path = BASE_DIRECTORY_FOR_Extracted_Faces + videoName + "/"+str(FRONTAL_FACE_HAAR_CLASSIFIER).split(".")[0]
            face_info= detect_faces(
                imageName=str(frame["frameName"]).split(".")[0],
                imagePath=frame["framePath"],
                outputPath=extracted_face_output_path,
                classifierName=FRONTAL_FACE_HAAR_CLASSIFIER,
            )
            for face in face_info:
                extracted_faces_info.append(face)
        for extracted_face in extracted_faces_info:
            landmarked_face_image_path = BASE_DIRECTORY_FOR_LANDMARKED_Faces+videoName+"/"+str(FRONTAL_FACE_HAAR_CLASSIFIER).split(".")[0]+"/"
            print("Landmarks for ",extracted_face["faceImageName"])
            detect_face_landmarks(imageName=extracted_face["faceImageName"],imagePath=extracted_face["faceImagePath"],outputPath=landmarked_face_image_path)
