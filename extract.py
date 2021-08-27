import cv2
import os
import numpy as np
from tqdm import tqdm

curDir = os.curdir  # 获取当前执行python文件的文件夹
print(curDir)
# 具体文件名修改一下
sourceDir = os.path.join(curDir, 'input')
resultDir = os.path.join(curDir, 'output')


def extractPicture(sourceDir, resultDir):

    img_list = os.listdir(sourceDir)
    index = 0;

    print(img_list)
    print(len(img_list))

    for i in tqdm(range(0, len(img_list))):

        # 1.导入图片
        img_src = cv2.imread("img/origin/" + img_list[i])

        # 2.灰度处理与二值化
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        # 3.连通域分析
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,
                                                            cv2.CHAIN_APPROX_SIMPLE)

        # 4.制作掩膜图片
        img_mask = np.zeros(img_src.shape, np.uint8)
        cv2.drawContours(img_mask, contours, -1, (255, 255, 255), -1)

        # 5.执行与运算
        img_result = cv2.bitwise_and(img_src, img_mask)

        # 6.显示结果
        # cv2.imshow("img_src", img_src)
        # cv2.imshow("img_mask", img_mask)
        # cv2.imshow("img_result", img_result)

        cv2.waitKey()
        cv2.destroyAllWindows()

        # 保存图片
        cv2.imwrite("img/origin/" + str(index) + ".png", img_result)


if __name__ == '__main__':
    extractPicture(sourceDir, resultDir)
