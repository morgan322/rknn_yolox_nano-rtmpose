import cv2
import numpy as np

def convert_yuv_to_jpg(yuv_list, width, height):

    y_size = width * height
    uv_size = y_size // 2


    y_list = yuv_list[:y_size]
    uv_list = yuv_list[y_size : y_size + uv_size]

    yuv = np.array(y_list+uv_list, dtype=np.uint8)
    yuv = yuv.reshape((height * 3 // 2, width))
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV21)
    



    # 将 RGB 图像保存为 JPG 文件
    cv2.imwrite('output.jpg', rgb)

file_path = "../yuv.txt"  

yuv_str = ''
# 打开文件
with open(file_path, "r") as file:
    for line in file:
        yuv_str += line
print(len(yuv_str))


yuv_list = []
for i in yuv_str.replace("[","").replace("]","").split(","):
    yuv_list.append(int(i.strip()))
print(len(yuv_list))


convert_yuv_to_jpg(yuv_list,640,480)
