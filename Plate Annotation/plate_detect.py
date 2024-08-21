from car_detect import car_detection

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


def plate_detection(img_src):
    plate_detect = YOLO("./license_plate_detector.pt")
    img = np.array(img_src)
    
    plates = plate_detect.predict(img_src, conf=0.7)
    
    output = []
    
    for plate in plates:
        data = plate.boxes.data.tolist()
        for box in data:
            # draw border around the car
            # img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
            
            plate = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            # Resize
            plate = cv2.resize(plate, (1000,1000))
            
            # Crop only the plate
            plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            _, plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            output.append(Image.fromarray(plate))
    
    return output
            
            
if __name__ == '__main__':
    a = car_detection('./pic/452533647_1716748245756349_1609499174989890082_n.png')
    print(len(a))
    b = plate_detection(a[0])
    
    b[0].show()
 
    