from flask import Flask, render_template, request, redirect, url_for, flash, session
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pytesseract
import os
import re
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
model = YOLO('model/best.pt')
key = b'Enkripsiplat3652'  # 16 bytes key for AES-128
cipher = AES.new(key, AES.MODE_ECB)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['filename'] = filename  # Simpan nama file di session
            return redirect(url_for('index'))

    # Ambil data dari session untuk ditampilkan
    filename = session.get('filename')
    prediction_image = session.get('prediction_image')
    prediction = session.get('prediction')
    extracted_text = session.get('extracted_text')
    encrypted_text = session.get('encrypted_text')
    decrypted_text = session.get('decrypted_text')

    return render_template('index.html', filename=filename, prediction_image=prediction_image, prediction=prediction, extracted_text=extracted_text, encrypted_text=encrypted_text, decrypted_text=decrypted_text)

@app.route('/predict/<filename>')
def predict_plate(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    frame = cv2.imread(img_path)
    
    if frame is None:
        return "Image not found", 404
    
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    prediction = ""
    class_list = open("coco1.txt").read().splitlines()
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(float, row[:6])  
        confidence = row[4] if len(row) > 5 else None  # Mengambil confidence score jika tersedia
        c = class_list[int(d)]
        prediction += f"Class: {c}, Confidence: {confidence:.2f}\n" if confidence else f"Class: {c}\n"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Simpan gambar yang diprediksi dengan bounding boxes
    prediction_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_' + filename)
    cv2.imwrite(prediction_image_path, frame)

    # Simpan hasil ke session
    session['prediction_image'] = 'prediction_' + filename
    session['prediction'] = prediction
    
    return redirect(url_for('index'))

@app.route('/extract/<filename>')
def extract_text(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], session.get('prediction_image'))  # Menggunakan gambar yang telah diprediksi
    frame = cv2.imread(img_path)

    if frame is None:
        return "Image not found", 404

    # Ambil bounding box dari prediksi
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    extracted_text = ""
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(float, row[:6])  # Unpack 6 nilai
        crop = frame[int(y1):int(y2), int(x1):int(x2)]  # Crop gambar sesuai bounding box
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray_crop, None, 30, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, np.ones((1, 1), np.uint8))

        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(morph, config=custom_config).strip()
        text = re.sub(r'[^A-Za-z0-9\s]', '', text).strip()
        extracted_text += f"{text} "

    # Simpan hasil ke session
    session['extracted_text'] = extracted_text.strip()

    return redirect(url_for('index'))

@app.route('/encrypt/<filename>')
def encrypt_text(filename):
    extracted_text = session.get('extracted_text')
    
    if not extracted_text:
        flash("No text available to encrypt. Please extract text first.")
        return redirect(url_for('index'))

    # AES Encryption
    padded_text = pad(extracted_text.encode(), AES.block_size)
    encrypted_text = cipher.encrypt(padded_text)
    encrypted_text_base64 = base64.b64encode(encrypted_text).decode('utf-8')

    # Simpan hasil ke session
    session['encrypted_text'] = encrypted_text_base64

    return redirect(url_for('index'))

@app.route('/decrypt', methods=['POST'])
def decrypt_text():
    cipher_text = request.form.get('cipher_text')

    try:
        encrypted_data = base64.b64decode(cipher_text)
        cipher = AES.new(key, AES.MODE_ECB)
        decrypted_text = unpad(cipher.decrypt(encrypted_data), AES.block_size).decode('utf-8')
        session['decrypted_text'] = decrypted_text
    except Exception as e:
        flash("Failed to decrypt the text. Please ensure the cipher text is correct.")

    return redirect(url_for('index') + '#decryption-box')


if __name__ == '__main__':
    app.run(debug=True)
