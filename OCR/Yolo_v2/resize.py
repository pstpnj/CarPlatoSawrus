from PIL import Image
import os

def resize_image(image_path):
    with Image.open(image_path) as img:
        new_width = 320
        new_height = 320
        resized_img = img.resize((new_width, new_height))
        
        resized_img.save(image_path)


# path = 'C:/Users/kosen/Desktop/PBLLLLLLL/resized/valid/images/'
for k in ['valid', 'test', 'train']:
    path = f'/{k}/images/'
    for i in os.listdir(path):
        img_path = path + i
        resize_image(img_path)
