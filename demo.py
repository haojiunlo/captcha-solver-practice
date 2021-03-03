import cv2
import os
import time
import torch
import requests
import argparse
import numpy as np
from models import ConvBnNet, Resnet
from torchvision import transforms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Net", dest="net",
                        help="model arch (defult ConvBnNet)", type=str)
    parser.set_defaults(net='ConvBnNet')
    parser.add_argument("--model_path", dest="model_path", help="model predix (defult prefix)", type=str)
    parser.set_defaults(model_path="model/torch_model.pkl")
    args = parser.parse_args()

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # output_dir
    out_dir = 'demo_img'

    # load trained model
    if args.net == 'ConvBnNet':
        model = ConvBnNet([10, 20, 40], 10)
    elif args.net == 'Resnet':
        model = Resnet([10, 20, 40], 10)
    else:
        raise ValueError("Oops! not valid model arch. ")

    model.load_state_dict(torch.load(args.model_path))

    # img transform
    transform = transforms.Compose([transforms.ToTensor()])

    # request session
    session = requests.Session()
    response = session.get('http://www.taifex.com.tw/cht/captcha', cookies={'from-my': 'browser'})

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'demo.jpg'), 'wb') as file:
        file.write(response.content)
        file.flush()

    img_cv2 = cv2.imread(os.path.join(out_dir, 'demo.jpg'))
    img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img = transform(img)


    class_names = range(10)

    model.eval()
    with torch.no_grad():
        out = ""
        img_w = img.shape[2]

        # split 6 digits
        digit_w = img.shape[2] // 6
        for w in range(0, img_w - 2, digit_w):
            crop_img = img[:, :, w : w + digit_w]
            crop_img = crop_img[np.newaxis, :, :, :]
            outputs = model(crop_img.to(device))
            _, preds = torch.max(outputs, 1)
            out += str(class_names[preds])

    print(out)
    cv2.imshow('demo', img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
