import os
import numpy as np
import librosa
import moviepy.editor as mp

import matplotlib.pyplot as plt

from scipy.signal import correlate

from synchonisationResults.processVideo import delayFrame, deleteFrames, retainFrames # 自建库


def audioWriter(audio1, audio2, temp_audio_path1, temp_audio_path2):
    audio1.write_audiofile(temp_audio_path1)
    audio2.write_audiofile(temp_audio_path2)
    
    
def audioReader(temp_audio_path1, temp_audio_path2):
    y1, sr1 = librosa.load(temp_audio_path1, sr=None)
    y2, sr2 = librosa.load(temp_audio_path2, sr=None)
    return  y1, sr1, y2, sr2


def computeCorrelation(y1_normalized, y2_normalized, sr1):

    # 计算 correlation
    corr = correlate(y1_normalized, y2_normalized, mode='full')
    # 计算时间轴
    lags = np.arange(-len(y1_normalized) + 1, len(y2_normalized))
    # 找到最大相关性的时间偏移
    best_offset = lags[np.argmax(corr)] / sr1
    print(f"音频2相对于音频1的时间偏移量为 {best_offset:.2f} 秒")
    
    return best_offset, lags, corr
    
    
def plotGraphs(y1_normalized, y2_normalized, sr1, sr2, best_offset, lags, corr):
    # 重新调整音频2的开始时间
    y2_shifted = np.roll(y2_normalized, int(best_offset * sr2))

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 绘制音频波形图
    librosa.display.waveshow(y1, sr=sr1, ax=axes[0])
    axes[0].set(title='Near Infrared Video Background Sound Waveform', ylabel='Amplitude')
    axes[0].grid(True)

    librosa.display.waveshow(y2, sr=sr2, ax=axes[1])
    axes[1].set(title='Visible Video Background Sound Waveform', ylabel='Amplitude')
    axes[1].grid(True)

    librosa.display.waveshow(y1_normalized, sr=sr1, ax=axes[2])
    axes[2].set(title='Normalized Near Infrared Sound Waveform (10s)', ylabel='Amplitude')
    axes[2].grid(True)

    librosa.display.waveshow(y2_normalized, sr=sr2, ax=axes[3])
    axes[3].set(title='Normalized Visible Sound Waveform (10s)', ylabel='Amplitude')
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig('./synchonisationResults/original_audio_waveforms.png')

    # 绘制 correlation
    plt.figure(figsize=(10, 5))
    plt.plot(lags / sr1, corr)
    plt.title('Cross-correlation between Normalized Audio Signals (10s)')
    plt.xlabel('Time Lag (s)')
    plt.ylabel('Correlation')
    plt.axvline(x=best_offset, color='r', linestyle='--', label=f'Best Offset: {best_offset:.2f} s')
    plt.legend()
    plt.grid(True)
    plt.savefig('./synchonisationResults/correlation_plot.png')


    # 将偏移后的音频叠加在一起
    alpha = 0.5
    y1_overlay = y1_normalized[:len(y2_shifted)]
    y2_overlay = y2_shifted[:len(y1_normalized)]
    overlay = y1_overlay + alpha * y2_overlay

    # 显示叠加后的音频
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(y1_normalized)) / sr1, y1_normalized, label='y1 (Near Infrared)', alpha=0.5)
    plt.plot(np.arange(len(y2_shifted)) / sr2, y2_shifted, label='y2 (Visible) shifted', alpha=0.5)
    plt.title('Overlay of Normalized Audio Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig('./synchonisationResults/audio_overlay.png')

def deleteAudios(temp_audio_path1, temp_audio_path2):
    # 删除临时音频文件
    if os.path.exists(temp_audio_path1):
        os.remove(temp_audio_path1)
    if os.path.exists(temp_audio_path2):
        os.remove(temp_audio_path2)


if __name__ == "__main__":

    video_path1 = './calibration_data/videos/IR.mp4'
    video_path2 = './calibration_data/videos/VIS.mp4'
    audio_threshold = 0.0 # 暂不考虑

    # 从路径中提取视频和音频
    clip1 = mp.VideoFileClip(video_path1)
    clip2 = mp.VideoFileClip(video_path2)
    audio1 = clip1.audio
    audio2 = clip2.audio
    # 保存音频到临时文件
    temp_audio_path1 = "./synchonisationResults/temp_audio1.wav"
    temp_audio_path2 = "./synchonisationResults/temp_audio2.wav"
    
    audioWriter(audio1, audio2, temp_audio_path1, temp_audio_path2)
    y1, sr1, y2, sr2 = audioReader(temp_audio_path1, temp_audio_path2)
    # 取音频的前10秒将音频信号的振幅归一化到相同尺度
    y1_normalized = librosa.util.normalize(y1[:10*sr1])
    y2_normalized = librosa.util.normalize(y2[:10*sr2])
    
    best_offset, lags, corr = computeCorrelation(y1_normalized, y2_normalized, sr1)
    
    plotGraphs(y1_normalized, y2_normalized, sr1, sr2, best_offset, lags, corr)
    
    deleteAudios(temp_audio_path1, temp_audio_path2)
    
    if best_offset>0:
        output_path = '.'+video_path1.split('.')[1]+'_syn_tmp.mp4'
        deleteFrames(video_path1, output_path, best_offset)
        retainFrames(output_path, '.'+video_path1.split('.')[1]+'_syn.mp4', 40)
        retainFrames(video_path2, '.'+video_path2.split('.')[1]+'_syn.mp4', 40)
    else:
        output_path = '.'+video_path2.split('.')[1]+'_syn_tmp.mp4'
        deleteFrames(video_path2, output_path, best_offset)   
        retainFrames(output_path, '.'+video_path2.split('.')[1]+'_syn.mp4', 40)
        retainFrames(video_path1, '.'+video_path1.split('.')[1]+'_syn.mp4', 40)