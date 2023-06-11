from config import (
    BASE_DIRECTORY_FOR_INPUT_VIDEOS,
    BASE_DIRECTORY_FOR_Extracted_Frames,
    BASE_DIRECTORY_FOR_Extracted_Faces,
    SAMPLE_VIDEOS,
    BASE_DIRECTORY_FOR_CV2_FRONTAL_FACE_HAAR_CLASSIFIER,
    FRONTAL_FACE_HAAR_CLASSIFIERS,
)
from CV2FaceExtractor import detect_faces
from VideoFramesExtractor import extract_frames


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
        for frame in extracted_frames_info:
            extracted_face_output_path = BASE_DIRECTORY_FOR_Extracted_Faces + videoName + "/"+FRONTAL_FACE_HAAR_CLASSIFIER
            detect_faces(
                imageName=str(frame["frameName"]).split(".")[0],
                imagePath=frame["framePath"],
                outputPath=extracted_face_output_path,
                classifierName=FRONTAL_FACE_HAAR_CLASSIFIER,
            )
