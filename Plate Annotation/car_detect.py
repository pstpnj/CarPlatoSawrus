import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


def car_detection(img_url):
    output = []

    # load the model
    car_detect = YOLO('./yolov8n.pt') # Model for detecting the car

    img_src = Image.open(img_url)
    img = np.array(img_src)

    # define id for each vehicle
    vehicles_id = [2, 3, 5, 7]
    
    # detect the car using pre-trained model
    vehicles = car_detect.predict(img_src, conf=0.5, classes=vehicles_id)

    for veh in vehicles:
        # get data of the vehicle
        datas = veh.boxes.data.tolist()
        # print(data)
        for data in datas:
            box = data[:4]
            # draw border around the car
            # img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
            
            # crop the car
            car = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            
			# Resize 
            # car = cv2.resize(car, (1000,1000))
            
            # Picture of each car
            output.append(Image.fromarray(car))
                
    return output

if __name__ == '__main__':
    output = car_detection('./pic/bangkok-traffic-1.jpg')
    for i in output:
        i.show()