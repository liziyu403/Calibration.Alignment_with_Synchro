import cv2
import numpy as np
import glob
import argparse
import pickle
import os
from tqdm import tqdm 


def getCalibrationMatrix(chessboard_size=None, source=None):
    
    # 预设棋盘格的世界坐标
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # 存储世界坐标和图像坐标
    objpoints = []
    imgpoints = []

    # 加载所有的棋盘格图像
    images = glob.glob(f'calibration_data/images/{source}/*.jpg')
    if len(images) < 25:
        logging.error("[ !图片数量少于25张 ]")
        exit()

    for fname in tqdm(images, desc=f"正在生成{TRANSLATOR[source]}校正矩阵："):
        img = cv2.imread(fname)
        if img is None:
            logging.error(f"[ !加载图像 {fname} 失败 ]")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray Image', gray)
        cv2.waitKey(100)

        # 查找棋盘格角点
        # ret：如果找到所有指定数量的角点，则返回 True，否则返回 False。
        # corners：形状为 (N, 1, 2)，其中 N 是检测到的角点的数量，每个角点由 (x, y) 坐标表示。
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)
            # 提高角点的精确度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # 绘制并显示角点
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
        else:
            logging.warning(f"[ !未在图像<{fname}>检测到棋盘特征 ]")

    cv2.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0:
        logging.error("[ !未检测到棋盘格特征 ]")
        exit()

    # 初始化相机矩阵和畸变系数，不定义的话可以直接是 None
    h, w = gray.shape[:2]
    camera_matrix = np.array([[w, 0, w / 2],
                              [0, w, h / 2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)
    
    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, dist_coeffs)

    # 可视化标定误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"Total error: {mean_error / len(objpoints)}")

    # 存储相机校正参数至 pkl
    calibration_data = {'camera_matrix': mtx, 'dist_coeff': dist}
    matrix_path = f'./calibration_matrix/{source}_source_calibration_parameters.pkl'

    # 或创建目录
    if not os.path.exists(matrix_path.split('/')[1]):
        os.makedirs(matrix_path.split('/')[1])
        
    with open(matrix_path, 'wb') as f:
        pickle.dump(calibration_data, f)

    print(f"\n:: 相机矫正参数已存入 > > > {matrix_path}\n")


if __name__ == '__main__':
    
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='图像畸变校正参数设置')
    parser.add_argument('--source', type=str, default='both', choices=['vis', 'ir', 'both'], help='数据来源类型: vis (可见光相机) or ir (近红外相机)')

    args = parser.parse_args()
    
    CHESSBOARD_SIZE = (9, 8)

    global TRANSLATOR
    TRANSLATOR = {'vis': '可见光相机', 'ir': '近红外相机'}
    
    if args.source == 'both':
        for source in ['vis','ir']:
            getCalibrationMatrix(chessboard_size=CHESSBOARD_SIZE, source=source)
    else:
        getCalibrationMatrix(chessboard_size=CHESSBOARD_SIZE, source=args.source)
