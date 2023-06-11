import cv2
import os
from config import BASE_DIRECTORY_FOR_INPUT_VIDEOS,BASE_DIRECTORY_FOR_Extracted_Frames,SAMPLE_VIDEOS

def extract_frames(videoName,videoPath, outputPath):
    outputPath=outputPath+videoName+"/"
    os.makedirs(outputPath, exist_ok=True)
    video = cv2.VideoCapture(videoPath)
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second:", fps)
    frame_count = 0
    frames_info=[]
    success = True
    while success:
        success, frame = video.read()
        if success:
            fram_name=f"frame_{frame_count}.jpg"
            frame_path = os.path.join(outputPath,fram_name )
            cv2.imwrite(frame_path, frame)
            frames_info.append({"frameName":fram_name,"framePath":frame_path})
            frame_count += 1
            print("frame: ",frame_count)
    video.release()
    print("Frames extracted:", frame_count)
    print("Extraction complete!")
    return frames_info

