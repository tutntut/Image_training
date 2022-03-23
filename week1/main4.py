import cv2

cap = cv2.VideoCapture('04.mp4')

while True:
    ret, img = cap.read()
    
    if ret == False:
        break
    
    img = cv2.resize(img, dsize=(640,360))
    
    #img = img[100:200, 150:250]
    
    cv2.imshow('result', img)
    
    if cv2.waitKey(10) == ord('q'):
        break
    
    