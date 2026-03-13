import cv2
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras

# ===== تحميل النموذج =====
model = keras.models.load_model('cnn_model.keras')
DIGITS = '0123456789'

IMG_DIR = r"C:\Users\badr1\Desktop\plate_project\data\ds\img"
ANN_DIR = r"C:\Users\badr1\Desktop\plate_project\data\ds\ann"

# ===== 1. Detection =====
def detect_plate(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 13, 15, 15)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rect_kern)
    sobel = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    sobel = np.absolute(sobel)
    sobel = (255 * sobel / sobel.max()).astype("uint8")
    morph = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, rect_kern)
    _, thresh = cv2.threshold(morph, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / h
        area = w * h
        img_area = img.shape[0] * img.shape[1]
        if 2.0 < ratio < 7.0 and area > img_area * 0.005:
            return img[y:y+h, x:x+w], (x, y, w, h), img
    return None, None, img

# ===== 2. Segmentation =====
def segment_characters(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2,
                      interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    h_plate = thresh.shape[0]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > h_plate * 0.3 and w > 5 and w < h_plate * 1.5:
            char_img = thresh[y:y+h, x:x+w]
            char_img = cv2.resize(char_img, (28, 28))
            chars.append((x, char_img))
    chars = sorted(chars, key=lambda c: c[0])
    return [c[1] for c in chars]

# ===== 3. Recognition =====
def recognize_chars(chars):
    result = ""
    for char in chars:
        char_input = char.reshape(1, 28, 28, 1).astype("float32") / 255.0
        pred = model.predict(char_input, verbose=0)
        digit = DIGITS[np.argmax(pred)]
        confidence = np.max(pred)
        if confidence > 0.5:
            result += digit
    return result

# ===== 4. Pipeline complet =====
def read_plate(img_path):
    plate, bbox, original = detect_plate(img_path)
    if plate is None or plate.size == 0:
        return original, "Non détecté"
    chars = segment_characters(plate)
    if not chars:
        return original, "Pas de caractères"
    text = recognize_chars(chars)
    # رسم النتيجة
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(original, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return original, text

# ===== 5. تجربة على 9 صور =====
images = os.listdir(IMG_DIR)[:9]
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for i, img_name in enumerate(images):
    img_path = os.path.join(IMG_DIR, img_name)
    result, text = read_plate(img_path)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    ax = axes[i//3][i%3]
    ax.imshow(result_rgb)
    ax.set_title(f"{img_name}\n→ {text}", fontsize=9)
    ax.axis('off')
    print(f"{img_name}: {text}")

plt.tight_layout()
plt.savefig('final_result.png')
print("\nOK - final_result.png cree!")