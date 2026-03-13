import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

IMG_DIR = r"C:\Users\badr1\Desktop\plate_project\data\ds\img"

def detect_plate_improved(img_path):
    img = cv2.imread(img_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Bilateral filter - يحافظ على الحواف
    blur = cv2.bilateralFilter(gray, 13, 15, 15)

    # 2. Morphological operations - يبرز الplaque
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rect_kern)

    # 3. Sobel - كشف الحواف الأفقية
    sobel = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    sobel = np.absolute(sobel)
    sobel = (255 * sobel / sobel.max()).astype("uint8")

    # 4. Closing - يوحد المنطقة
    morph_close = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, rect_kern)

    # 5. Threshold
    _, thresh = cv2.threshold(morph_close, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Erosion + Dilation
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 7. إيجاد الcontours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_found = False
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / h
        area = w * h
        img_area = img.shape[0] * img.shape[1]

        if 2.0 < ratio < 7.0 and area > img_area * 0.005:
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 3)
            plate_found = True
            break

    if not plate_found:
        # fallback - الطريقة الأولى
        edges = cv2.Canny(blur, 30, 200)
        contours2, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:20]
        for c in contours2:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if 2 < w/h < 6:
                    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    break

    return original

# تجربة على 9 صور
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
images = os.listdir(IMG_DIR)[:9]

for i, img_name in enumerate(images):
    result = detect_plate_improved(os.path.join(IMG_DIR, img_name))
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    ax = axes[i//3][i%3]
    ax.imshow(result_rgb)
    ax.set_title(img_name)
    ax.axis('off')

plt.tight_layout()
plt.savefig('detection_improved.png')
print("OK - detection_improved.png cree!")