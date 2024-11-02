import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import pytesseract
from datetime import datetime
import os
import re
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
from time import perf_counter

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

image_folder = 'images9'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]

images.sort()

with open("coco1.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

area = [(0, 0), (0, 499), (1019, 499), (1019, 0)] 

count = 0
list1 = []
processed_numbers = []

key = b'Enkripsiplat3652'  # 16 bytes key for AES-128

cipher = AES.new(key, AES.MODE_ECB)

with open("car_plate_data.txt", "a") as file:
    file.write("\n{:<15}\t{:<10}\t{:<8}\t{:<12}\t{:<32}\t{:<15}\n".format("NoPlat", "Tanggal", "Waktu", "Klasifikasi", "Enkripsi AES", "Waktu Enkripsi (ms)")) 

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    count += 1

    if frame is None:
        break
   
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
   
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        if result >= 0:
                crop = frame[y1:y2, x1:x2]
                
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Grayscale', gray_crop)
                
                denoised = cv2.fastNlMeansDenoising(gray_crop, None, 30, 7, 21)
                _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imshow('Binary', binary)

                morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(morph, config=custom_config).strip()

                text = re.sub(r'[^A-Za-z0-9\s]', '', text)
                text = re.sub(r'\s+', ' ', text)  
                text = text.strip()  
                print(text)
                
                processed_numbers.append(text)
                list1.append(text)
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                numeric_part = re.findall(r'\d+', text)
                if numeric_part:
                    last_digit = numeric_part[-1][-1]
                    odd_even = "genap" if int(last_digit) % 2 == 0 else "ganjil"
                else:
                    odd_even = "tidak terbaca"
                
                # Mengukur waktu enkripsi AES dalam milidetik menggunakan perf_counter
                start_time = perf_counter()

                padded_text = pad(text.encode(), AES.block_size)
                encrypted_text = cipher.encrypt(padded_text)

                end_time = perf_counter()
                encryption_time = (end_time - start_time) * 1000  # Konversi ke milidetik
                encrypted_text_base64 = base64.b64encode(encrypted_text).decode('utf-8')
             
                with open("car_plate_data.txt", "a") as file:
                    file.write("{:<15}\t{:<10}\t{:<8}\t{:<12}\t{:<32}\t{:<15.3f}\n".format(text, current_datetime.split()[0], current_datetime.split()[1], odd_even, encrypted_text_base64, encryption_time))
                
                cv2.imshow('crop', gray_crop)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
