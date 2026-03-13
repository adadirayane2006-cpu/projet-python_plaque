import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

IMG_DIR = r"C:\Users\badr1\Desktop\plate_project\data\ds\img"

def detect_plate(img_path):
    img = cv2.imread(img_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # تصفية + كشف الحواف
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    
    # إيجاد الcontours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    
    plate = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / h
            if 2 < ratio < 6:  # نسبة شكل الplaque
                plate = original[y:y+h, x:x+w]
                cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 3)
                break
    
    return original, plate

# تجربة على 6 صور
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
images = os.listdir(IMG_DIR)[:6]

for i, img_name in enumerate(images):
    result, plate = detect_plate(os.path.join(IMG_DIR, img_name))
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    ax = axes[i//3][i%3]
    ax.imshow(result_rgb)
    ax.set_title(img_name)
    ax.axis('off')

plt.tight_layout()
plt.savefig('detection_result.png')
print("OK - detection_result.png cree!")