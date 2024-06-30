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
            cv2.waitKey(0)
            cv2.imshow('IR', img_ir)
            cv2.waitKey(0)
        else:
            print(f"Chessboard corners not found for pair: {img_vis_path} and {img_ir_path}")

    cv2.destroyAllWindows()
    return objpoints, imgpoints_vis, imgpoints_ir

def stereo_calibrate_and_rectify(images, chessboard_size):
    objpoints, imgpoints_vis, imgpoints_ir = find_corners(images, chessboard_size)
    
    # 确保角点匹配的数量相同
    if len(imgpoints_vis) == 0 or len(imgpoints_ir) == 0:
        print("No corners found in images. Please check the images and try again.")
        return None, None, None, None
    
    assert len(imgpoints_vis) == len(imgpoints_ir)

    # 获取图像大小
    img_vis = cv2.imread(images[0][0])
    img_ir = cv2.imread(images[0][1])
    img_size = img_vis.shape[:2][::-1]

    # 双目标定，假设内参已矫正
    ret, mtx_vis, dist_vis, mtx_ir, dist_ir, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_vis, imgpoints_ir, None, None, None, None, img_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    # 立体校正
    print(R,T)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_vis, dist_vis, mtx_ir, dist_ir, img_size, R, T)

    map1x, map1y = cv2.initUndistortRectifyMap(mtx_vis, dist_vis, R1, P1, img_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx_ir, dist_ir, R2, P2, img_size, cv2.CV_16SC2)

    return map1x, map1y, map2x, map2y

def rectify_images(images, map1x, map1y, map2x, map2y):
    if map1x is None or map1y is None or map2x is None or map2y is None:
        print("Error in stereo calibration. Cannot rectify images.")
        return

    for img_vis_path, img_ir_path in images:
        img_vis = cv2.imread(img_vis_path)
        img_ir = cv2.imread(img_ir_path)

        rectified_vis = cv2.remap(img_vis, map1x, map1y, cv2.INTER_LINEAR)
        rectified_ir = cv2.remap(img_ir, map2x, map2y, cv2.INTER_LINEAR)

        cv2.imshow('Rectified RGB', rectified_vis)
        cv2.imshow('Rectified IR', rectified_ir)
        cv2.waitKey(500)

        # 保存矫正后的图像
        save_dir = './rectified_images'
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'rectified_vis_' + os.path.basename(img_vis_path)), rectified_vis)
        cv2.imwrite(os.path.join(save_dir, 'rectified_ir_' + os.path.basename(img_ir_path)), rectified_ir)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    calibration_images = [
        ('./calibrationResults/corrected_vis_imgs/corrected_frame_0010.jpg', 
         './calibrationResults/corrected_ir_imgs/corrected_frame_0010.jpg'),
        # 添加更多的棋盘格图像对
    ]
    CHESSBOARD_SIZE = (9, 8)

    # 计算立体校正的映射矩阵
    map1x, map1y, map2x, map2y = stereo_calibrate_and_rectify(calibration_images, CHESSBOARD_SIZE)

    # 使用映射矩阵对齐其他图像
    test_images = [
        ('./calibrationResults/corrected_vis_imgs/corrected_frame_0010.jpg', 
         './calibrationResults/corrected_ir_imgs/corrected_frame_0010.jpg'),
        # 添加更多的测试图像对
    ]
    rectify_images(test_images, map1x, map1y, map2x, map2y)
