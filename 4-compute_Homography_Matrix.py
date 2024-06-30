import numpy as np
import cv2
import os

def find_corners(images, chessboard_size):
    objpoints = []
    imgpoints_vis = []
    imgpoints_ir = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    for img_vis_path, img_ir_path in images:
        img_vis = cv2.imread(img_vis_path)
        img_ir = cv2.imread(img_ir_path)

        gray_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2GRAY)
        gray_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)

        ret_rgb, corners_rgb = cv2.findChessboardCorners(gray_rgb, chessboard_size, None)
        ret_ir, corners_ir = cv2.findChessboardCorners(gray_ir, chessboard_size, None)

        if ret_rgb and ret_ir:
            objpoints.append(objp)
            imgpoints_vis.append(corners_rgb)
            imgpoints_ir.append(corners_ir)

            cv2.drawChessboardCorners(img_vis, chessboard_size, corners_rgb, ret_rgb)
            cv2.drawChessboardCorners(img_ir, chessboard_size, corners_ir, ret_ir)
            cv2.imshow('RGB', img_vis)
            cv2.imshow('IR', img_ir)
            cv2.waitKey(500)  

    cv2.destroyAllWindows()
    return objpoints, imgpoints_vis, imgpoints_ir

def calculate_homography(images, chessboard_size):
    objpoints, imgpoints_vis, imgpoints_ir = find_corners(images, chessboard_size)
    
    # 确保角点匹配的数量相同
    assert len(imgpoints_vis) == len(imgpoints_ir)

    # 使用所有角点计算单应矩阵
    H, status = cv2.findHomography(np.array(imgpoints_vis).reshape(-1, 2), 
                                   np.array(imgpoints_ir).reshape(-1, 2), 
                                   cv2.RANSAC, 5.0)

    # 存储单应矩阵
    save_dir = './homographyResults'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'homography.npy'), H)
    print(f"\n:: 相机矫正参数已存入 > > > {os.path.join(save_dir, 'homography.npy')}\n")

if __name__ == '__main__':
    images = [
        ('./calibrationResults/corrected_vis_imgs/corrected_frame_0010.jpg', 
         './calibrationResults/corrected_ir_imgs/corrected_frame_0010.jpg'),
        # ('./calibrationResults/corrected_vis_imgs/corrected_frame_0015.jpg', 
        #  './calibrationResults/corrected_ir_imgs/corrected_frame_0015.jpg'),
        # ('./calibrationResults/corrected_vis_imgs/corrected_frame_0020.jpg', 
        #  './calibrationResults/corrected_ir_imgs/corrected_frame_0020.jpg'),
        # ('./calibrationResults/corrected_vis_imgs/corrected_frame_0025.jpg', 
        #  './calibrationResults/corrected_ir_imgs/corrected_frame_0025.jpg'),
        # ('./calibrationResults/corrected_vis_imgs/corrected_frame_0030.jpg', 
        #  './calibrationResults/corrected_ir_imgs/corrected_frame_0030.jpg'),
        # 添加更多的图像对
    ]
    CHESSBOARD_SIZE = (9, 8)
    
    calculate_homography(images, chessboard_size=CHESSBOARD_SIZE)
                                        