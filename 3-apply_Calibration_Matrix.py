import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm 
import glob
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def calibrateImage(image_path, save_path, mtx, dist, searchAlpha, alpha):
    try:
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        if img is None:
            logging.error(f"[ !无法加载需要畸变校正的图像，位置：{image_path} ]")
            return

        if searchAlpha:
            fig = plt.figure(figsize=(15, 2.9))
            gs = GridSpec(2, 6, width_ratios=[3, 1, 1, 1, 1, 1], height_ratios=[1, 1])

            # 左子图显示原畸变图像
            ax_orig = fig.add_subplot(gs[:, 0])
            ax_orig.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax_orig.set_title('Original Image')
            ax_orig.axis('off')
            
            alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 遍历预设的 alpha 值

            for i, alpha in enumerate(alphas):
                h, w = img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))

                # 使用新相机矩阵校正图像
                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

                # 按需裁剪图像
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]

                # 存储校正后的图像
                alpha_save_path = save_path.replace('.jpg', f'_alpha_{alpha}.jpg')
                # cv2.imwrite(alpha_save_path, dst) # 暂不写入
                logging.info(f"Calibrated image with alpha {alpha} saved to {alpha_save_path}")

                # 右子图显示不同 alpha 下的校正结果
                ax = fig.add_subplot(gs[i // 5, (i % 5) + 1])
                ax.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
                ax.set_title(f'alpha={alpha}')
                # ax.axis('off')

            plt.tight_layout()
            output_path = os.path.join(alpha_save_path.split('/')[0], alpha_save_path.split('/')[1], 'Alpha_comparison.jpg')
            plt.savefig(output_path)
            plt.show()
            
        else:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))

            # 使用新相机矩阵校正图像
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # 按需裁剪图像
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]

            # 存储校正后的图像
            cv2.imwrite(save_path, dst)            

    except Exception as e:
        logging.error(f"[ !校正错误: {str(e)}")


def applyCalibration(source=None, mode='single', calibrationPath=None, searchAlpha=False, alpha=1.0):
    
    try:
        # 加载相机校正参数
        with open(f'./calibration_matrix/{source}_source_calibration_parameters.pkl', 'rb') as f:
            calibration_data = pickle.load(f)
            mtx = calibration_data['camera_matrix']
            dist = calibration_data['dist_coeff']

        if mode == 'single':
            # 单张图像校正
            image_path = 'frame_rgb.jpg'  
            if searchAlpha:
                save_path = f'calibrationResults/searchAplpha_{source}.jpg'
            else:
                save_path = f'calibrationResults/corrected_frame_{source}.jpg'
            calibrateImage(image_path, save_path, mtx, dist, searchAlpha, alpha)
        
        elif mode == 'group' or searchAlpha:
            # 整个文件夹下所有文件校正
            if not os.path.exists(calibrationPath):
                logging.error(f"Calibration path {calibrationPath} does not exist.")
                return
            
            image_paths = glob.glob(f'{calibrationPath}/*.jpg')
            if not image_paths:
                logging.error(f"[ !目录 {calibrationPath} 下未找到图像 ]")
                return
            
            if searchAlpha:
                save_dir = f'calibrationResults/searchAplpha_{source}'
            else:                
                save_dir = f'calibrationResults/corrected_{os.path.basename(calibrationPath)}_imgs'
                
            os.makedirs(save_dir, exist_ok=True)
            
            for image_path in tqdm(image_paths, desc="正在校正文件夹中的所有图像"):
                file_name = os.path.basename(image_path)
                save_path = os.path.join(save_dir, f'corrected_{file_name}')
                calibrateImage(image_path, save_path, mtx, dist, searchAlpha, alpha)
                
                if searchAlpha:
                    print("> 请选择你认为合适的Alpha值")
                    break
                
    except Exception as e:
        logging.error(f"[ !校正错误: {str(e)} ]")


if __name__ == '__main__':
    
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='图像畸变校正参数设置')
    parser.add_argument('--source', type=str, default='both', choices=['vis', 'ir', 'both'], help='数据来源类型: vis (可见光相机) or ir (近红外相机)')
    parser.add_argument('--mode', type=str, default='group', choices=['single', 'group'], help='模式: single (处理单张图片) 或 group (处理整组图片)')
    parser.add_argument('--calibration_path', type=str, default='./calibration_data/images', help='待矫正图片所在文件夹')
    parser.add_argument('--searchAlpha', type=bool, default=False, help='遍历搜索合适的校正超参 alpha')
    parser.add_argument('--Alpha', type=float, default=.0, help='校正 alpha')

    args = parser.parse_args()

    global TRANSLATOR
    TRANSLATOR = {'vis': '可见光相机', 'ir': '近红外相机'}
    
    if args.source == 'both':
        for source in ['vis','ir']:
            applyCalibration(source=source, mode=args.mode, calibrationPath=os.path.join(args.calibration_path, source), searchAlpha=args.searchAlpha, alpha=args.Alpha)
    else:
        applyCalibration(source=args.source, mode=args.mode, calibrationPath=os.path.join(args.calibration_path, args.source), searchAlpha=args.searchAlpha, alpha=args.Alpha)

    print()
