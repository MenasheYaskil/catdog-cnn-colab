# 📓 Experiment Report — Cats vs Dogs (CNN)

> **Author:** **Menashe Yaskil**

This report summarizes the configuration, training runs, metrics, and takeaways for the baseline CNN model.

---

## 1) Setup
- **Platform:** Google Colab (CPU/GPU)
- **Checkpoint path:** `/content/drive/MyDrive/DataSet/checkpoints/...`
- **Image size:** 150×150 (CPU fallback: 128×128)
- **Batch size:** 50 (CPU fallback: 32)
- **Epochs:** up to 25 with EarlyStopping
- **Augmentation:** rotation/shift/zoom/flip (moderate)
- **Optimizer/Loss:** Adam / Binary Crossentropy

> Fill in your run details below (times, device, exact best epoch).

- **Device:** T4 / P100 / CPU (choose one)
- **Total training time:** `XX:YY` (mm:ss)
- **Best epoch:** `E`

---

## 2) Dataset Summary
Update counts from your notebook:
- **Train cats:** `N`
- **Train dogs:** `N`
- **Test cats:** `N`
- **Test dogs:** `N`
- **Real pictures (inference set):** `N`

Ensure classes are balanced; otherwise consider `class_weight` or collecting more samples.

---

## 3) Metrics
| Metric | Value |
|---|---|
| **Best Validation Accuracy** | `0.XX` |
| **Best Validation Loss** | `0.XX` |
| **AUC** (if enabled) | `0.XX` |

Extract programmatically (example):
```python
best_val_acc = max(history.history['val_accuracy'])
best_val_loss = min(history.history['val_loss'])
print(best_val_acc, best_val_loss)
```

Confusion matrix (optional):
```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

test_generator.reset()
p = model.predict(test_generator, verbose=0).ravel()
y_pred = (p > 0.5).astype(int)
y_true = test_generator.classes[:len(y_pred)]

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))
```

---

## 4) Curves
Include the generated figures (save from the notebook to `assets/`):

![Accuracy](assets/accuracy.png)
![Loss](assets/loss.png)

---

## 5) Observations & Next Steps
- Baseline is stable; validation closely tracks training → limited overfitting
- If accuracy plateaus early, consider:
  - Slightly larger input (e.g., 180×180) and BatchNorm
  - LR scheduling (`ReduceLROnPlateau`)
  - More/cleaner data and balanced classes
- For a strong boost: try **transfer learning** (EfficientNet/MobileNet) in a separate branch

---

## 6) Reproducibility Checklist
- Fixed random seed (`42`)
- Same dataset split every run
- Same augmentation parameters
- Saved best checkpoint and exact `requirements.txt`
