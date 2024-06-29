import os
import cv2 
import glob 
import numpy as np

def video2imgs(input_video_path=None,output_folder=None,frame_interval=30):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(input_video_path)

    frame_count = 0
    extracted_count = 0

    if not cap.isOpened():
        print(f"无法打开视频文件: {input_video_path}")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔frame_interval帧保存一张图片
            if frame_count % frame_interval == 0:
                output_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
                cv2.imwrite(output_filename, frame)
                print(f"保存图片: {output_filename}")
                extracted_count += 1

            frame_count += 1

        cap.release()
        print(f"总共提取了{extracted_count}张图片")
    
    
if __name__ == '__main__':
    
    sources = ['IR', 'VIS']
    
    for source in sources:
        input_video_path = f'calibration_data/videos/{source}_syn.MP4'
        output_folder    = f'calibration_data/images/{source}'
        video2imgs(input_video_path=input_video_path, output_folder=output_folder)