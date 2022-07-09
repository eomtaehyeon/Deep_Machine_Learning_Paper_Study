
import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지 단순 합성

# ---① 연산에 사용할 이미지 읽기
img1 = cv2.imread('images/cat_06.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('images/cat_ragdoll_02.png', cv2.IMREAD_COLOR)

# ---② 이미지 덧셈
img3 = img1 + img2  # 더하기 연산
img4 = cv2.add(img1, img2) # OpenCV 함수

imgs = {'img1':img1, 'img2':img2, 'img1+img2': img3, 'cv.add(img1, img2)': img4}

# ---③ 이미지 출력
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,2, i + 1)
    plt.imshow(v[:,:,::-1])
    plt.title(k)
    plt.xticks([]); plt.yticks([])

plt.show()