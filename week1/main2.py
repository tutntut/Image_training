import cv2

img = cv2.imread('01.jpg')

print(img)
print(img.shape)

cv2.rectangle(img, pt1=(259,89), pt2=(380,348),color=(255,0,0),thickness=4)

cv2.circle(img, center=(320, 220), radius=100, color=(0, 255, 0), thickness=2)

cropped_img = img[89:348, 259:380]

img_resized = cv2.resize(img,(512,256))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('result', img)
cv2.imshow('cropped_result', cropped_img)
cv2.imshow('resized_result', img_resized)
cv2.imshow('rgb_result', img_rgb)
cv2.waitKey(0)