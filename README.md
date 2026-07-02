# MTech---Mini-Project-
```markdown
# Brain Tumor Classification using DenseNet201 and Focal Loss

This project implements a deep learning model to classify brain MRI scans into four categories: Glioma, Meningioma, Pituitary tumor, and No Tumor. 

## 🚀 Project Overview
- **Model Architecture:** DenseNet201 (Transfer Learning)
- **Input Size:** 224x224x3
- **Optimization:** Adam Optimizer with focal loss to handle potential class imbalances.
- **Augmentation:** Utilized `Albumentations` for robust data preprocessing including CLAHE, rotations, and flips.
- **Performance:** Achieved ~98% accuracy on the test dataset.

## 📊 Dataset
The model is trained on the Brain Tumor MRI Dataset, which consists of Training and Testing folders divided into four classes.

## 🛠️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/brain-tumor-classification.git
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow albumentations opencv-python-headless matplotlib seaborn pillow
   ```
3. Run the notebook or script to train/evaluate the model.

## 📈 Results
### Classification Report
| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Glioma | 0.99 | 0.94 | 0.97 |
| Meningioma | 0.94 | 0.97 | 0.95 |
| No Tumor | 0.99 | 1.00 | 0.99 |
| Pituitary | 0.98 | 0.99 | 0.99 |

### Confusion Matrix
Included in the repository to visualize performance across all classes.

## 🧠 Custom Focal Loss
Since MRI datasets can be imbalanced, we implemented a custom Focal Loss function to focus training on hard-to-classify examples:
```python
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    # ... implementation details ...
```

## 📜 License
Distributed under the MIT License.
```
