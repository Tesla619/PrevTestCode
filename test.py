from email.mime import image
import cv2, time, os, tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(20)

def showCamera():
    cap = cv2.VideoCapture(0)
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    
    x1 = 0
    y1 = 0
    x2 = int(width)
    y2 = int(height)
        
    if (cap.isOpened() == False):
        print("Error opening file...")
        return

    (success, image) = cap.read()

    startTime = 0
    
    while success:
        currentTime = time.time()

        fps = 1/(currentTime - startTime)
        startTime = currentTime     
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
        
        cv2.putText(image, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),2)
        cv2.imshow("Results", image)

        key = cv2.waitKey(1) & 0XFF
        if key == ord("q"):
            break

        (success, image) = cap.read()
    cv2.destroyAllWindows()
    
showCamera()