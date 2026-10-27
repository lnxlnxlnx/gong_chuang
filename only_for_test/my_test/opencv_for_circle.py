import cv2
import numpy as np


def color_circle_position (img):     
    """
    色环识别辅助实现定位
    :param img: 输入的图像 BGR三通道
    :return: 三个色环的中心坐标 x1,y1,x2,y2,x3,y3
    """
    erode_hsv = cv2.erode(img, None, iterations=2)  # 腐蚀 粗的变细
    kernel = np.ones((7, 7), np.uint8)#5,5
    diRange_hsv = cv2.dilate(erode_hsv, kernel, 1) # 膨胀 填补空洞
    gray_img = cv2.cvtColor(diRange_hsv, cv2.COLOR_BGR2GRAY) # 转化为单通道灰度图
    
    # 限制对比度自适应直方图均衡
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))#5.0 
    clahed = clahe.apply(gray_img) # 【注意！原代码这里算完clahed，但是后面没用上，是bug！】
    
    # 计算形态学梯度（增强物体边缘）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel)
    
    result = cv2.GaussianBlur(gradient, (7, 7), 3, 3) # 高斯模糊 平滑边缘
    
    eqal_img = cv2.convertScaleAbs(result, alpha=4, beta=0) # 再整体增强对比度
    cv2.imshow("video2", eqal_img)
    eqal_img = cv2.GaussianBlur(eqal_img, (7, 7), 3, 3) # 再一次高斯模糊 平滑边缘
 
    retval, threshold_img = cv2.threshold(eqal_img, 70, 255, cv2.THRESH_BINARY) # 二值化
 
    threshold_img = cv2.GaussianBlur(threshold_img, (9, 9), 3, 3) # 最后再来一次高斯模糊 平滑边缘
 
    # 霍夫圆检测
    circles = cv2.HoughCircles(threshold_img, cv2.HOUGH_GRADIENT_ALT, 1.5, 50, param1=100, param2=0.95, minRadius=15,
                               maxRadius=50)
    cv2.imshow("video", gray_img)
 
    cv2.imshow("video3", threshold_img)
 
    try:
        if(len(circles[0])==3):
            circles = np.uint16(np.around(circles))
            # 遍历
            for circle in circles[0, :]:
                cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
                cv2.circle(img, (circle[0], circle[1]), 2, (255, 0, 0), 2)          
            circle_all=[circles[0][0],circles[0][1],circles[0][2]]
            circle_list =  sorted(circle_all, key=lambda x: x[0]) # 按x坐标从左到右排序
            return (circle_list[0][0],circle_list[0][1],circle_list[1][0],circle_list[1][1],circle_list[2][0],circle_list[2][1]) # 依次返回色环的中心坐标
    except Exception as e:
        print("未检测到3个圆",e)
        pass
    return None # 找不到返回None


# ============ 调用示例：图片测试 ============
if __name__ == "__main__":
    # 读取图片，替换成你的图片路径
    img = cv2.imread("output/debug/rest.png")
    if img is None:
        print("图片读取失败")
        exit()
    # 调用函数
    res = color_circle_position(img)
    if res is not None:
        x1,y1,x2,y2,x3,y3 = res
        print(f"圆1:({x1},{y1}), 圆2:({x2},{y2}), 圆3:({x3},{y3})")
    else:
        print("没有识别到3个色环！")

    cv2.imshow("result",img)
    cv2.waitKey(0) # 等待按键，必须要有！
    cv2.destroyAllWindows()
