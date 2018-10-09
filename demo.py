from train import *
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

model = torch.load('torch_model.pkl')

img = cv2.imread(os.path.join(out_dir, 'demo.jpg'))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

model.eval()
with torch.no_grad():
    out = str('')
    for w in range(0, 138, 23):
        crop_img = img_gray[:, w:w+23]
        tmp = torch.from_numpy(crop_img).float()
        tmp = tmp.unsqueeze(0)
        tmp = tmp.unsqueeze(0)
        outputs = model(tmp.to(device))
        _, preds = torch.max(outputs, 1)
        out += class_names[preds]

print(out)
cv2.imshow('demo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

