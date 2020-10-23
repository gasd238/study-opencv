# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
 
# img = cv2.imread('test.jpg') #이미지 불러오기
# cv2.imshow("image",img) #이미지 보여주기 (윈도우창 이름, 불러온 이미지)
# k = cv2.waitKey(0)  #키보드 눌림 대기
# if k == 27:# ESC키
#     cv2.destroyAllWindows()
# elif k == ord('s'): #저장하기 버튼
#     cv2.imwrite("test2.png",img)
#     cv2.destroyAllWindows()

# #image croping
# img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
# image_50x50 = cv2.resize(img, (50, 50))


# fig, ax = plt.subplots(1,2, figsize=(10,5))
# ax[0].imshow(img, cmap='gray')
# ax[0].set_title('Original Image')
# ax[1].imshow(image_50x50, cmap='gray')
# ax[1].set_title('Resized Image')
# plt.show()


# #image crop
# img = cv2.imread('test.jpg') #이미지 불러오기
# img_cropped = img[100:130, 0:80]

# cv2.imshow("test", img_cropped)
# k = cv2.waitKey(0)  #키보드 눌림 대기
# if k == 27:# ESC키
#     cv2.destroyAllWindows()
# elif k == ord('s'): #저장하기 버튼
#     cv2.imwrite("test2.png",img_cropped)
#     cv2.destroyAllWindows()

# #image blur
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# image_blurry = cv2.blur(img, (10,10))

# kernel = np.ones((10,10)) / 25.0 # 모두 더하면 1이 되도록 정규화
# image_kernel = cv2.filter2D(img, -1, kernel)

# plt.imshow(image_kernel)
# plt.show()

# image_very_blurry = cv2.GaussianBlur(img, (11,11), 0)

# plt.imshow(image_very_blurry, cmap='gray')
# plt.show()

# # load the image and show it
# image = cv2.imread("test.jpg")
# cv2.imshow("Original", image)
 
# # grab the dimensions of the image and calculate the center of the image
# (h, w) = image.shape[:2]
# (cX, cY) = (w / 2, h / 2)
 
# # rotate our image by 45 degrees
# M = cv2.getRotationMatrix2D((cX, cY), -45, 1)
# rotated = cv2.warpAffine(image, M, (w, h))
# cv2.imshow("Rotated by 45 Degrees", rotated)

# cv2.waitKey(0)

# #load the image and show it
# image = cv2.imread("test.jpg")
# cv2.imshow("Original", image)
 
# # 회전의 중심축을 정의하지 않으면 그림의 중심이 됨
# rotated = imutils.rotate(image, 45)  
# cv2.imshow("Rotated by 45 Degrees", rotated)
 
# # 회전의 중심 축을 정의하면 해당 중심축으로 회전을 함.
# rotated = imutils.rotate(image, 45, center=(0, 0)) # 회전 중심축 TOP LEFT  
# cv2.imshow("Rotated by 45 Degrees 2", rotated)

# cv2.waitKey(0)

# image = cv2.imread("test.jpg")
# cv2.imshow("Original", image)
 
# # X축 뒤집기
# flipped = cv2.flip(image, 0)
# cv2.imshow("X axis", flipped)
 
# # Y축 뒤집기
# flipped = cv2.flip(image, 1)
# cv2.imshow("Y axis", flipped)
 
# # X, Y축 동시
# flipped = cv2.flip(image, -1)
# cv2.imshow("Both Flipped", flipped)
 
 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image = cv2.imread("test.jpg")
# # File Size : 579Kb
# print("Image Size : ",image.size) # (475 x 600 x 3) ==> 855Kb
# print("Image Shape : ",image.shape) # (height, width, channel)
# print("Image Data Type : ", image.dtype) # unsigned integer 8 bit
# cv2.imwrite("beach.jpg", image) # file size : 173 Kb
 
# cv2.waitKey(0)
 
# image = cv2.imread("test.jpg")
# (b,g,r) = image[100, 200] # x = 200, y = 100
# print(b, g, r)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
 
# # Change One Pixel
# image[20, 20] = (0,0,255)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
 
# # Change Part, X, Y 좌표값 고려
# image[20:50, 20:100] = (0,255,0)
# cv2.imshow("Image", image)
# cv2.waitKey(0)
 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 이미지 로드 후 RGB로 변환
# image_bgr = cv2.imread('test.jpg')
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# # 사각형 좌표: 시작점의 x,y  ,넢이, 너비
# rectangle = (0, 56, 256, 150)

# # 초기 마스크 생성
# mask = np.zeros(image_rgb.shape[:2], np.uint8)

# # grabCut에 사용할 임시 배열 생성
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)

# # grabCut 실행
# cv2.grabCut(image_rgb, # 원본 이미지
#            mask,       # 마스크
#            rectangle,  # 사각형
#            bgdModel,   # 배경을 위한 임시 배열
#            fgdModel,   # 전경을 위한 임시 배열 
#            5,          # 반복 횟수
#            cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화
# # 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
# mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# # 이미지에 새로운 마스크를 곱행 배경을 제외
# image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]

# # plot
# plt.imshow(image_rgb_nobg)
# plt.show()

# image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

# # 커널 생성(대상이 있는 픽셀을 강조)
# kernel = np.array([[0, -1, 0],
#                    [-1, 5, -1],
#                    [0, -1, 0]])

# # 커널 적용 
# image_sharp = cv2.filter2D(image, -1, kernel)

# fig, ax = plt.subplots(1,2, figsize=(10,5))
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title('Original Image')
# ax[1].imshow(image_sharp, cmap='gray')
# ax[1].set_title('Sharp Image')
# plt.show()