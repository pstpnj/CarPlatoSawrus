from ultralytics import YOLO
from PIL import Image
from PIL import ImageFilter

model = YOLO('OCR/best4.pt')
model.conf = 0.7

image = Image.open(r'OCR/image7.jpg')
resize = image.resize((640,640))
blur = resize.filter(ImageFilter.GaussianBlur(1))

results = model(blur)
for i in results:
    print(i.boxes.data.tolist())
    i.show()