import cv2
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG_DIR = r"C:\Users\badr1\Desktop\plate_project\data\ds\img"
ANN_DIR = r"C:\Users\badr1\Desktop\plate_project\data\ds\ann"

def load_boxes(img_name):
    json_path = os.path.join(ANN_DIR, img_name + ".json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    boxes = []
    for obj in data.get('objects', []):
        if obj['geometryType'] == 'rectangle':
            ext = obj['points']['exterior']
            x1, y1 = int(ext[0][0]), int(ext[0][1])
            x2, y2 = int(ext[1][0]), int(ext[1][1])
            boxes.append((x1, y1, x2, y2))
    return boxes

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
images = os.listdir(IMG_DIR)[:6]

for i, img_name in enumerate(images):
    img = cv2.imread(os.path.join(IMG_DIR, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for (x1, y1, x2, y2) in load_boxes(img_name):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    ax = axes[i//3][i%3]
    ax.imshow(img)
    ax.set_title(img_name)
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_plates.png')
print("OK - fichier sample_plates.png cree!")