BASE_DIRECTORY_FOR_INPUT_VIDEOS = (
    "FacialExpressionExtrationAndRcognition/Resources/Videos/"
)
BASE_DIRECTORY_FOR_OUTPUT = (
    "FacialExpressionExtrationAndRcognition/Output/"
)
BASE_DIRECTORY_FOR_Extracted_Frames=BASE_DIRECTORY_FOR_OUTPUT+"ExtractedFrames/"
BASE_DIRECTORY_FOR_Extracted_Faces=BASE_DIRECTORY_FOR_OUTPUT+"ExtractedFaces/"
# SAMPLE_VIDEOS = ["man-explaining-1.mp4", "man-explaining-2.mp4", "man-explaining-3.mp4"]
SAMPLE_VIDEOS = ["man-explaining-2-5_1.mp4","man-explaining-2-5_2.mp4"]

BASE_DIRECTORY_FOR_CV2_FRONTAL_FACE_HAAR_CLASSIFIER = (
    "FacialExpressionExtrationAndRcognition/Resources/CV2Classifiers/"
)
FRONTAL_FACE_HAAR_CLASSIFIERS = [
    # "haarcascade_frontalface_default.xml",
    # "haarcascade_frontalface_alt.xml",
    # "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
]
