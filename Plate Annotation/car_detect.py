import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


def car_detection(img_url):
    output = []

    # load the model
    car_detect = YOLO('./yolov8n.pt') # Model for detecting the car
    license_plate_detector = YOLO('./license_plate_detector.pt') # Model for detecting the license plate

    img = Image.open('./pic/traffic2.jpeg')
    img = np.array(img)

    # detect the car using pre-trained model
    vehicles = car_detect.predict(img, conf=0.7)

    # define id for each vehicle
    vehicles_id = [2, 3, 5, 7]

    for veh in vehicles:
        # get data of the vehicle
        datas = veh.boxes.data.tolist()
        # print(data)
        for data in datas:
            # get the id of the vehicle
            id = data[5]
            if id in vehicles_id:
                box = data[:4]
                # draw border around the car
                img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
                # crop the car
                car = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                output.append(car)
                
    return output

if __name__ == '__main__':
    output = car_detection('./pic/traffic2.jpeg')
    for i, car in enumerate(output):
        cv2.imshow(f'car {i}', car)
    cv2.waitKey(0)
    cv2.destroyAllWindows()