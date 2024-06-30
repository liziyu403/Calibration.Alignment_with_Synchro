import cv2
import numpy as np

def apply_homography(frame):

    H = np.load('./homographyResults/homography.npy')

    # 读取新的RGB图像和对应的IR图像
    img_rgb_new = cv2.imread(f'./calibrationResults/corrected_vis_imgs/{frame}.jpg')
    img_ir = cv2.imread(f'./calibrationResults/corrected_ir_imgs/{frame}.jpg')

    height, width = img_ir.shape[:2]
    # 透视变换
    aligned_rgb = cv2.warpPerspective(img_rgb_new, H, (width, height))

    # 调整两幅图像的大小一致
    img_ir_resized = cv2.resize(img_ir, (width, height))

    # RGB2为灰度图像
    gray_rgb = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY)
    aligned_rgb_gray_3ch = cv2.cvtColor(gray_rgb, cv2.COLOR_GRAY2BGR)

    # 图像混叠
    alpha = 0.5 
    blended_image = cv2.addWeighted(aligned_rgb_gray_3ch, alpha, img_ir_resized, 1 - alpha, 0)

    print("点击任意键结束")
    cv2.imshow('Blended Image', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    
    frame = 'corrected_frame_0010'
    apply_homography(frame)