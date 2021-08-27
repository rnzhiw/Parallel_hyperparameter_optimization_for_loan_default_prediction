import cv2
import os

from tqdm import tqdm

# img = cv2.imread("input/GF2_2.tif")
# print(img.shape)

# cropped = img[0:120, 0:160]  # 裁剪坐标为[y0:y1, x0:x1]
# cv2.imwrite("output/test.jpg", cropped)

curDir = os.curdir  # 获取当前执行python文件的文件夹
print(curDir)
sourceDir = os.path.join(curDir, 'input')
resultDir = os.path.join(curDir, 'output')

def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img

def resolution_lower_handler(sourceDir, resultDir):
    img_list = os.listdir(sourceDir)
    index = 0;

    print(img_list)
    print(len(img_list))


    for i in tqdm(range(0, len(img_list))):
        img = cv2.imread("GID/ann_dir/train" + img_list[i])
        w, h, g = img.shape
        # print(img.shape)
        print(w, h)
        for j in range(0, w, 128):
            for k in range(0, h, 128):
                if j > w - 256 and k <= h - 256:
                    cropped = img[j:w, k:k+256]
                    new_img = cv2_letterbox_image(cropped, (256, 256))
                    cv2.imwrite("GID/train/labels/" + str(index) + ".png", new_img)
                    index = index + 1
                elif j <= w - 256 and k > h - 256:
                    cropped = img[j:j+256, k:h]
                    new_img = cv2_letterbox_image(cropped, (256, 256))
                    cv2.imwrite("GID/train/labels/" + str(index) + ".png", new_img)
                    index = index + 1
                elif j > w - 256 and k > h - 256:
                    cropped = img[j:w, k:h]
                    new_img = cv2_letterbox_image(cropped, (256, 256))
                    cv2.imwrite("GID/train/labels/" + str(index) + ".png", new_img)
                    index = index + 1
                else:
                    cropped = img[j:j + 256, k:k + 256]
                    cv2.imwrite("GID/train/labels/result_" + str(index) + ".png", cropped)
                    index = index + 1

if __name__ == '__main__':
    resolution_lower_handler(sourceDir, resultDir)