import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

IMG_DIR = r"C:\Users\badr1\Desktop\plate_project\data\ds\img"
ANN_DIR = r"C:\Users\badr1\Desktop\plate_project\data\ds\ann"
OUTPUT_DIR = r"C:\Users\badr1\Desktop\plate_project\chars_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import json

def get_plate_from_annotation(img_path, img_name):
    """استخراج الplaque باستخدام annotation الحقيقية"""
    json_path = os.path.join(ANN_DIR, img_name + ".json")
    img = cv2.imread(img_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
    for obj in data.get('objects', []):
        if obj['geometryType'] == 'rectangle':
            ext = obj['points']['exterior']
            x1, y1 = int(ext[0][0]), int(ext[0][1])
            x2, y2 = int(ext[1][0]), int(ext[1][1])
            plate = img[y1:y2, x1:x2]
            return plate
    return None

def segment_characters(plate_img):
    """تقطيع الحروف من الplaque"""
    # تحويل لرمادي
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # تكبير الصورة باش نشوفو أحسن
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # إزالة الضوضاء
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # إيجاد الcontours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    chars = []
    h_plate = thresh.shape[0]
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # فلترة الحروف بناءً على الحجم
        if h > h_plate * 0.3 and w > 5 and w < h_plate * 1.5:
            char_img = thresh[y:y+h, x:x+w]
            char_img = cv2.resize(char_img, (28, 28))
            chars.append((x, char_img))  # x باش نرتبهم
    
    # ترتيب الحروف من اليسار لليمين
    chars = sorted(chars, key=lambda c: c[0])
    return [c[1] for c in chars], thresh

# تجربة على 4 صور
images = os.listdir(IMG_DIR)[:4]
fig, axes = plt.subplots(len(images), 2, figsize=(14, 12))

for i, img_name in enumerate(images):
    img_path = os.path.join(IMG_DIR, img_name)
    plate = get_plate_from_annotation(img_path, img_name)
    
    if plate is not None and plate.size > 0:
        chars, thresh = segment_characters(plate)
        
        # عرض الplaque الأصلية
        axes[i][0].imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f"{img_name} - Plaque originale")
        axes[i][0].axis('off')
        
        # عرض الthreshold مع الحروف
        axes[i][1].imshow(thresh, cmap='gray')
        axes[i][1].set_title(f"{len(chars)} caractères détectés")
        axes[i][1].axis('off')
        
        print(f"{img_name}: {len(chars)} caractères trouvés")
        
        # حفظ الحروف
        for j, char in enumerate(chars):
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}_char{j}.png"), char)

plt.tight_layout()
plt.savefig('segmentation_result.png')
print("OK - segmentation_result.png cree!")