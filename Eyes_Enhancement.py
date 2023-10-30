import numpy as np
import cv2
import time
import math
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties # 导入字体库
from skimage.metrics import structural_similarity as ssim

# *------------0.画图配置工作------------*
chinese_font_name = "SimHei"  # 一种常见的中文字体
# 创建字体属性对象
font = FontProperties(family=chinese_font_name)
def show_pic(img,title):
    # img:输入图像 title：'标题(字符串格式)'
    plt.figure()
    plt.imshow(img, 'gray')
    plt.title(title, fontproperties=font)
    plt.axis('off')
    plt.show()

# *------------1.获取原图像，并对图像进行颜色空间转换------------*
p1_start_time = time.time()

raw_img = cv2.imread("test.jpg")
raw_imgHSV=cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)

p1_end_time = time.time()
p1_time=p1_end_time-p1_start_time
print("步骤1的算法耗时：%fs" % p1_time )

raw_imgRGB = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

# 分步作图展示
show_pic(raw_imgRGB,'raw_imgRGB(原始图像)')

# *------------2.获取亮度V通道信号，并进行傅里叶变换------------*
p2_start_time = time.time()

img_V=raw_imgHSV[:,:,2]
# 快速傅里叶变换算法得到频率分布
f = np.fft.fft2(img_V)
# 默认结果中心点位置是在左上角,调用fftshift()函数转移到中间位置
fshift = np.fft.fftshift(f)

p2_end_time = time.time()
p2_time=p2_end_time-p2_start_time
print("步骤2的算法耗时：%fs" % p2_time )

# fft结果是复数, 其绝对值结果是振幅
fimg = np.log(np.abs(fshift))

# # 观察频域信号的分布，以便确定截至频率
# fimg_info=[]
# for i in range(fimg.shape[0]):
#     fimg_info.append([np.min(fimg[i]),np.mean(fimg[i]),np.max(fimg[i])])
# print(fimg_info)

# 分步作图展示
show_pic(img_V,'img_V(图像V通道时域图)')
show_pic(fimg,'fimg(图像V通道频域图)')

# *------------3.对频域亮度值进行巴特沃斯滤波，进一步得到高/低通频谱------------*
# 创建巴特沃斯低通滤波器
def butterworth_filter (img, order, cutoff_frequency):
    """ butterworth filter genreator
    H(u, v) = 1 / (1 + (D(u, v) / cutoff_frequency)^(2 * order))
    Args:
        img:               输入灰度图
        order:             滤波器阶数
        cutoff_frequency:  截至频率
    """
    # 中心位置
    h, w = img.shape[0],img.shape[1]
    cx, cy = w // 2, h // 2
# 优化代码后
    # 创建频率域网格
    u, v = np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)
    D = np.sqrt(u ** 2 + v ** 2)
    # 计算滤波器
    tmp = 1 / (1 + (D / cutoff_frequency) ** (2 * order))
# 优化代码前
    # tmp = np.zeros((h, w))
    # for i in range(h):
    #     for j in range(w):
    #         dis= np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
    #         tmp[i, j] = 1 / (1 + (dis / cutoff_frequency) ** (2 * order))

    return tmp

p3_start_time = time.time()

cutoff_frequency =  5.7    # 截止频率
butterworth_order = 4  # 巴特沃斯滤波器的阶数
# 应用滤波器
l_filtered=butterworth_filter(fshift, butterworth_order, cutoff_frequency)
l_filtered_fft =fshift * l_filtered  #  低频亮度灰度图像信号频谱
h_filtered_fft =fshift-l_filtered_fft # 高频亮度灰度图像信号频谱

p3_end_time = time.time()
p3_time=p3_end_time-p3_start_time
print("步骤3的算法耗时：%fs" % p3_time )


l_filtered_img = np.abs(np.fft.ifft2(np.fft.ifftshift(l_filtered_fft))) #时域低频亮度灰度图像信号
h_filtered_img = np.abs(np.fft.ifft2(np.fft.ifftshift(h_filtered_fft))) #时域高频亮度灰度图像信号

# 分步作图展示
show_pic(l_filtered,'巴特沃斯低通滤波器')
show_pic(l_filtered_img,'l_filtered_img(滤波后图像V通道低频时域图)')
show_pic(h_filtered_img,'h_filtered_img(滤波后图像V通道高频时域图)')


# *------------4.将频域高频信号叠加在原亮度信号，并做傅里叶反变换和直方图均衡化------------*
# 创建函数将灰度值映射在[0,255]之间
def convert_gray_depth8(image):
    image_r = (2**8)*(image - image.min())/(image.max() - image.min())
    return(image_r)

p4_start_time = time.time()

# 高频信号叠加在原亮度信号
img_fshift=fshift+h_filtered_fft
# 增强的亮度信号进行傅里叶反变换
img_V_R=convert_gray_depth8(np.abs(np.fft.ifft2(np.fft.ifftshift(img_fshift))))
# 将亮度、对比度增强的V通道信号进行直方图均衡化，实现局部对比度的提高并拓展全局亮度
img_V_eqh = cv2.equalizeHist(img_V_R.astype(np.uint8))

p4_end_time = time.time()
p4_time=p4_end_time-p4_start_time
print("步骤4的算法耗时：%fs" % p4_time )

# 分步作图展示
show_pic(img_V_R,'img_V_R(叠加增强的V通道时域图)')
show_pic(img_V_eqh,'img_V_eqh(直方图均衡化后增强的V通道时域图)')


# *------------5.将最终图像增强的V通道信号替换至原图中，并将最终图像进行保存------------*
p5_start_time = time.time()

result_hsv=raw_imgHSV
result_hsv[:,:,2]=img_V_eqh
result_BGR=cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite("result.png",result_BGR)

p5_end_time = time.time()
p5_time=p5_end_time-p5_start_time
print("步骤5的算法耗时：%fs" % p5_time )
# 总体算法耗时
total_time =p1_time+p2_time+p3_time+p4_time+p5_time
print("总共算法耗时：%fs" % total_time )

result_RGB=cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)
# 分步作图展示
show_pic(result_RGB,'result_RGB(最终图像增强效果图)')


# *----------------------6.图像质量评价----------------------*
## PSNR评价指标
# 创建峰值信噪比（PSNR）评价函数【PSNR值越高，表示图像质量越好】
def PSNR (original_image,processed_image):
    # original_image： 原始图像灰度图
    # processed_image：处理后图像灰度图
    # 将图像转换为NumPy数组
    original_array = np.array(original_image, dtype=np.float64)
    processed_array = np.array(processed_image, dtype=np.float64)
    # 计算均方误差（MSE）
    mse = np.mean((original_array - processed_array) ** 2)
    # 计算PSNR
    max_pixel_value = 255  # 对于8位图像
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)
    return psnr

## SSIM评价指标
# 结构相似性指数（SSIM）是一种用于图像质量评估的指标，它衡量了两个图像之间的结构相似性
# [-1,1] 1表示两个图像非常相似，-1表示两个图像差异非常大。通常，较高的SSIM值表示更好的图像质量。

# 原始图像灰度图
raw_imggray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

# 一、通过本文设计方法处理后的灰度图
result_gray=cv2.cvtColor(result_BGR, cv2.COLOR_BGR2GRAY)

# 二、未均衡化得到的结果的灰度图
result_neqh=raw_imgHSV
result_neqh[:,:,2]=img_V_R
result_neqh_BGR=cv2.cvtColor(result_neqh, cv2.COLOR_HSV2BGR)
result_neqh_gray=cv2.cvtColor(result_neqh, cv2.COLOR_BGR2GRAY)

# 三、另一种增强方法：1.灰度化和直方图均衡化的预处理，2.自适应阈值化和对比度调整 3.纹理增强处理，并将其与原始图像合并为彩色RGB图像
equalized = cv2.equalizeHist(raw_imggray)
# 防曝光和失真处理，自适应阈值化
_, thresholded = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 对比度调整
alpha = 1.5
beta = 50
exposure_corrected = cv2.convertScaleAbs(equalized, alpha=alpha, beta=beta)
# 纹理增强处理，高斯滤波平滑图像
smoothed = cv2.GaussianBlur(exposure_corrected, (5, 5), 0)
# 纹理增强滤波器
filtered = cv2.fastNlMeansDenoising(smoothed, None, 10, 7, 21)
# 合并原图和处理后的图像
another_result = cv2.addWeighted(raw_img, 0.7, cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR), 0.3, 0)
another_result_RGB = cv2.cvtColor(another_result, cv2.COLOR_BGR2RGB)
another_result_gray= cv2.cvtColor(another_result, cv2.COLOR_BGR2GRAY)

print("PSNR评价：result_neqh_gray:%f  another_result_gray:%f  result_gray:%f  "%(PSNR(raw_imggray,result_neqh_gray),PSNR(raw_imggray,another_result_gray),PSNR(raw_imggray,result_gray)))
print("SSIM评价：result_neqh_gray:%f  another_result_gray:%f  result_gray:%f  "%(ssim(raw_imggray,result_neqh_gray),ssim(raw_imggray,another_result_gray),ssim(raw_imggray,result_gray)))
print("对比度评价：result_neqh_gray:%f  another_result_gray:%f  result_gray:%f  "%(np.std(result_neqh_gray) ,np.std(another_result_gray),np.std(result_gray)))

# *----------------------7.处理结果整体对比展示----------------------*

plt.subplot(331), plt.imshow(img_V, 'gray'), plt.title('img_V(图像V通道时域图)', fontproperties=font)
plt.axis('off')
plt.subplot(332), plt.imshow(fimg, 'gray'), plt.title('fimg(图像V通道频域图)', fontproperties=font)
plt.axis('off')
plt.subplot(333), plt.imshow(l_filtered, 'gray'), plt.title('巴特沃斯低通滤波器', fontproperties=font)
plt.axis('off')
plt.subplot(334), plt.imshow(l_filtered_img, 'gray'), plt.title('l_filtered_img(滤波后图像V通道低频时域图)', fontproperties=font)
plt.axis('off')
plt.subplot(335), plt.imshow(h_filtered_img, 'gray'), plt.title('h_filtered_img(滤波后图像V通道高频时域图)', fontproperties=font)
plt.axis('off')
plt.subplot(336), plt.imshow(img_V_R, 'gray'), plt.title('img_V_R(叠加增强的V通道时域图)', fontproperties=font)
plt.axis('off')
plt.subplot(337), plt.imshow(img_V_eqh, 'gray'),plt.title('img_V_eqh(直方图均衡化后增强的V通道时域图)', fontproperties=font)
plt.axis('off')
plt.subplot(338), plt.imshow(raw_imgRGB, 'gray'), plt.title('raw_imgRGB(原始图像)', fontproperties=font)
plt.axis('off')
plt.subplot(339), plt.imshow(result_RGB), plt.title('result_RGB(最终图像增强效果图)', fontproperties=font)
plt.axis('off')
plt.show()

