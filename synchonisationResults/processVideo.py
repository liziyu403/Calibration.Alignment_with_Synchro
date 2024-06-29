import moviepy.editor as mp
import sys


def delayFrame(input_path, output_path, delay_seconds):
    video = mp.VideoFileClip(input_path)
    
    # 前置位补帧
    black_clip = mp.ColorClip(size=video.size, color=(0, 0, 0), duration=delay_seconds)
    black_clip = black_clip.set_fps(video.fps)
    # 前置位与后置位拼接
    delayed_video = mp.concatenate_videoclips([black_clip, video])
    
    delayed_video.write_videofile(output_path, codec='libx264')


def deleteFrames(input_path, output_path, delete_seconds):
    video = mp.VideoFileClip(input_path)
    
    # 删除前置位帧
    trimmed_video = video.subclip(delete_seconds, video.duration)
    
    trimmed_video.write_videofile(output_path, codec='libx264')


def retainFrames(input_path, output_path, retain_seconds):
    video = mp.VideoFileClip(input_path)
    
    # 保留前n秒的视频
    retained_video = video.subclip(0, retain_seconds)
    
    retained_video.write_videofile(output_path, codec='libx264')


if __name__ == "__main__":
    try:
        if len(sys.argv) < 5:
            print("Usage: python video_processing.py <operation> <input_path> <output_path> <seconds>")
            assert len(sys.argv) >= 5
        else:
            operation = sys.argv[1]
            input_path = sys.argv[2]
            output_path = sys.argv[3]
            seconds = float(sys.argv[4])
            
            if operation == "delay":
                delayFrame(input_path, output_path, seconds)
            elif operation == "delete":
                deleteFrames(input_path, output_path, seconds)
            elif operation == "retain":
                retainFrames(input_path, output_path, seconds)
            else:
                print("Unknown operation. Use 'delay', 'delete', or 'retain'.")
    except:
        input_path = '../calibration_data/videos/IR.mp4'
        output_path = './output.mp4'
        delay_seconds = 6
        delete_seconds = 6
        retain_seconds = 6
        
        # Default operation examples for testing
        # delayFrame(input_path, output_path, delay_seconds)
        deleteFrames(input_path, output_path, delete_seconds)
        # retainFrames(input_path, output_path, retain_seconds)
