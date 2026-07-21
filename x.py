import matplotlib.pyplot as plt
import numpy as np

# Simulated ROC Data
fpr = np.linspace(0, 1, 100)
tpr_yolov6 = np.sqrt(fpr) * 0.9
tpr_yolov8 = fpr**0.4

# Accuracy per epoch (simulated)
epochs = np.arange(1, 11)
acc_yolov6 = [0.65, 0.67, 0.70, 0.71, 0.73, 0.735, 0.745, 0.75, 0.751, 0.752]
acc_yolov8 = [0.70, 0.74, 0.77, 0.79, 0.80, 0.81, 0.82, 0.83, 0.834, 0.835]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
axes[0].plot(fpr, tpr_yolov6, label='YOLOv6 (AUC ≈ 0.78)', linestyle='--')
axes[0].plot(fpr, tpr_yolov8, label='YOLOv8 (AUC ≈ 0.90)', linestyle='-')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[0].set_title("ROC Curve Comparison")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()
axes[0].grid(True)

# Accuracy Curve
axes[1].plot(epochs, acc_yolov6, marker='o', label='YOLOv6')
axes[1].plot(epochs, acc_yolov8, marker='s', label='YOLOv8')
axes[1].set_title("Accuracy Over Epochs")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

# Save and show
plt.tight_layout()
plt.savefig("yolo_v6_vs_v8_comparison.png")
plt.show()
