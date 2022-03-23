import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('models/instance_norm/the_scream.t7')

img = cv2.imread('imgs/04.jpg')

cropped_img = img[140:370, 480:810]

h,w,c = cropped_img.shape

cropped_img = cv2.resize(cropped_img, dsize = (500, int(h/w*500)))

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(cropped_img, mean=MEAN_VALUE)

net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1, 2, 0))

output += MEAN_VALUE
output = np.clip(output, 0, 255)
output = output.astype('uint8')

output = cv2.resize(output, dsize =(w,h))

img[140:370, 480:810] = output

cv2.imshow('result', output)
cv2.imshow('result', img)
cv2.waitKey(0)

