from train import *
from train_ import *
import cv2
from bs4 import BeautifulSoup
import time
import urllib.request
import requests
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


session = requests.Session()
out_dir = 'demo_img'

if not os.path.exists(out_dir):
        os.makedirs(out_dir)
response = session.get('http://www.taifex.com.tw/cht/captcha', cookies={'from-my': 'browser'})
with open(os.path.join(out_dir, 'demo.jpg'), 'wb') as file:
    file.write(response.content)
    file.flush() 

model = torch.load('torch_model_.pkl')

transform = transforms.Compose([
    transforms.ToTensor(), ]
    )

img = cv2.imread(os.path.join(out_dir, 'demo.jpg'))
img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_ = transform(img_)


class_names = range(10)

model.eval()
with torch.no_grad():
    out = str('')
    for w in range(0, 138, 23):
        crop_img = img_[:, :, w:w+23]
        crop_img = crop_img[np.newaxis, :, :, :]
        outputs = model(crop_img.to(device))
        _, preds = torch.max(outputs, 1)
        out += str(class_names[preds])

print(out)
cv2.imshow('demo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

