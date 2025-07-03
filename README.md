# 🧠 Support Vector Machines (SVM) for Breast Cancer Classification

## 🔍 Objective
This project uses Support Vector Machines (SVM) with both **Linear** and **RBF kernels** to classify tumors as **benign** or **malignant** using the Breast Cancer Wisconsin dataset.

---

## 📁 Dataset
- Source: [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
- Filename: `breast-cancer.csv`

---

## 📌 Key Concepts Covered
- Margin maximization
- Kernel trick
- Hyperparameter tuning (C, gamma)
- Evaluation using accuracy, precision, recall, F1-score
- GridSearchCV for optimization

---

## 🔧 Tools Used
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 🚀 Model Results

| Model       | Accuracy |
|-------------|----------|
| Linear SVM  | 95.6%    |
| RBF SVM     | 98.2%    |

✅ Best Model: RBF SVM  
🔍 Best Parameters: `{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}`

---

## 📂 Files in Repository
- `svm_breast_cancer_classification.py` – main code
- `breast-cancer.csv` – dataset (if permitted)
- `README.md` – this documentation
