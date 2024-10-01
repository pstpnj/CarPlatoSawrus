import time
from PIL import Image
import cv2
from cv2 import dnn_superres
from PIL import Image

from ultralytics import YOLO
model = YOLO("OCR/best.pt")

text = ['0', '1', 'ก', 'ข', 'ค', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', '2', 'ฌ', 'ญ', 'ฎ', 'ฐ', 'ฒ', 'ณ', 'ด', '3', 'ต', 
        'ถ', 'ท', 'ธ', 'น', 'บ', 'ผ', 'พ', '4', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', '5', 'ห', 
        'ฬ', 'อ', 'ฮ', '6', '7', '8', '9', 'AngThong', 'Ayutthaya', 'Bangkok', 'BuengKan', 'Buriram','Chachoengsao',
        'ChaiNat', 'Chaiyaphum', 'Chanthaburi', 'ChiangMai', 'ChiangRai', 'Chonburi','Chumphon', 'Kalasin',
        'KamphaengPhet', 'Kanchanaburi', 'KhonKaen', 'Krabi', 'Lampang', 'Lamphun','Loei', 'LopBuri', 'MahaSarakham',
        'Mukdahan', 'NakhonNayok', 'NakhonPathom', 'NakhonPhanom','NakhonRatchasima', 'NakhonSawan', 'NakhonSiThammarat',
        'Nan', 'NongBuaLamphu', 'NongKhai','Nonthaburi', 'PathumThani', 'Phatthalung', 'Phayao', 'Phetchabun',
        'Phetchaburi', 'Phichit','Phitsanulok', 'Phrae', 'Phuket', 'PrachinBuri', 'PrachuapKhiriKhan', 'Ratchaburi',
        'Rayong','RoiEt', 'SaKaeo', 'SakonNakhon', 'SamutPrakan', 'SamutSakhon', 'SamutSongkhram', 'SaraBuri', 
        'SiSaKet', 'SingBuri', 'Songkhla', 'Sukhothai', 'SuphanBuri', 'SuratThani', 'Surin', 'Tak','Trang',
        'UbonRatchathani', 'UdonThani', 'UthaiThani', 'Yala', 'Yasothon']


def findnumber(arr):
    arr = sorted(arr,key=lambda x: x[0])
    n = ''
    p = ''
    for i in arr:
        if i[-1] >= 47:
            p += text[int(i[-1])]
            continue
        n += text[int(i[-1])]
    return [n, p]


t1 = time.time()
t2 = time.time()
print('text',t2-t1)


img = cv2.imread('Plate/result/color/plate8.png')

sr = dnn_superres.DnnSuperResImpl_create()
path = 'OCR/LapSRN_x8.pb'
sr.readModel(path)
sr.setModel('lapsrn', 8)
upscaled = sr.upsample(img)

threshold_img = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
threshold_img = cv2.threshold(threshold_img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)[1]



image = Image.fromarray(threshold_img)
final = image.resize((640,640))
results = model(final)


t2 = time.time()
print( "Time for prediction", t2-t1 )
for i in results: 
    i.show()
boxes = results[0].boxes.data.tolist()
print(findnumber(boxes))
t3 = time.time()
print("Time for extract",t3-t2)



# Image enchance
#threshold_img = cv2.adaptiveThreshold(threshold_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
#threshold_img = cv2.medianBlur(threshold_img, 1)