import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ===== 1. تحميل النموذج والداتا =====
print("Chargement du modele et des donnees...")
model = keras.models.load_model('cnn_model.keras')
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# ===== 2. Test Global =====
print("\n===== TEST GLOBAL =====")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy : {test_acc*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ===== 3. Predictions =====
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# ===== 4. Classification Report =====
print("\n===== RAPPORT DE CLASSIFICATION =====")
print(classification_report(y_test, y_pred_classes,
      target_names=[str(i) for i in range(10)]))

# ===== 5. Confusion Matrix =====
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Matrice de Confusion — CNN', fontsize=14, fontweight='bold')
plt.xlabel('Prediction')
plt.ylabel('Valeur Reelle')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nconfusion_matrix.png cree!")

# ===== 6. Test sur images individuelles =====
print("\n===== TEST SUR 10 IMAGES ALEATOIRES =====")
indices = np.random.choice(len(x_test), 10, replace=False)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

correct = 0
for i, idx in enumerate(indices):
    img = x_test[idx]
    true_label = y_test[idx]
    pred = np.argmax(model.predict(img.reshape(1,28,28,1), verbose=0))
    confidence = np.max(model.predict(img.reshape(1,28,28,1), verbose=0))

    if pred == true_label:
        correct += 1
        color = 'green'
        status = 'OK'
    else:
        color = 'red'
        status = 'ERREUR'

    ax = axes[i//5][i%5]
    ax.imshow(img.reshape(28,28), cmap='gray')
    ax.set_title(f"Vrai: {true_label} | Pred: {pred}\n{status} ({confidence*100:.1f}%)",
                 color=color, fontsize=9)
    ax.axis('off')
    print(f"Image {idx}: Vrai={true_label} | Pred={pred} | Conf={confidence*100:.1f}% | {status}")

plt.suptitle(f'Test individuel — {correct}/10 corrects', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('test_individuel.png')
print(f"\nResultat: {correct}/10 corrects")
print("test_individuel.png cree!")

# ===== 7. Accuracy par classe =====
print("\n===== ACCURACY PAR CHIFFRE =====")
for digit in range(10):
    mask = y_test == digit
    acc_digit = np.mean(y_pred_classes[mask] == y_test[mask])
    bar = '█' * int(acc_digit * 20)
    print(f"Chiffre {digit}: {acc_digit*100:.2f}% {bar}")