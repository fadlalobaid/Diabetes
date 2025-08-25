import pandas as pd  # التعامل مع الجداول والبيانات
import numpy as np  # التعامل مع الأرقام بسرعة كبيرة

# للتعامل مع الصور البيانية
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import (
    train_test_split,
)  # تقسيم البيانات لجزأين (تدريب/اختبار)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# قياس مدى دقة التنبؤ في النتائج الحقيقية
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.metrics import classification_report

# يشتغلو على البيانات الغير متوازنة ويخلوها متوازنة
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# يجيب البيانات القديمة والجديدة ونشوف الفرق
from collections import Counter

# إعدادات الشكل
plt.style.use("fivethirtyeight")

# تجاهل التحذيرات
import warnings

warnings.filterwarnings("ignore")

# استدعاء البيانات
data = pd.read_csv("diabetes.csv")

# ------- إنشاء النموذج -------
x = data.drop("Outcome", axis=1)
y = data["Outcome"]

# موازنة البيانات
rm = RandomOverSampler(random_state=42)
x_res, y_res = rm.fit_resample(x, y)

# تقسيم البيانات
x_train, x_test, y_train, y_test = train_test_split(
    x_res, y_res, test_size=0.2, random_state=42
)

print("Training set size:", x_train.shape)
print("Test set size:", x_test.shape)

print(f"old data set shape {Counter(y)}")
print(f"new data set shape {Counter(y_res)}")

# النماذج
model1 = LogisticRegression()
model2 = SVC()
model3 = RandomForestClassifier(n_estimators=100, class_weight="balanced")
model4 = GradientBoostingClassifier(n_estimators=1000)

columns = [
    "LogisticRegression",
    "SVC",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
]
result1, result2, result3 = [], [], []


# دالة حساب المقاييس
def cal(model):
    model.fit(x_train, y_train)  # التدريب على البيانات
    pre = model.predict(x_test)  # التنبؤ بالنتائج

    # ⚠️ الترتيب الصحيح: (y_true, y_pred)
    accuracy = accuracy_score(y_test, pre)
    recall = recall_score(y_test, pre)
    f1 = f1_score(y_test, pre)

    result1.append(accuracy)
    result2.append(recall)
    result3.append(f1)

    sns.heatmap(confusion_matrix(y_test, pre), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model.__class__.__name__}")
    plt.show()

    print(f"Algorithm Name : {model.__class__.__name__}")
    print(f"accuracy : {accuracy:.2f} | recall : {recall:.2f} | f1 : {f1:.2f}")
    print("=" * 60)


# استدعاء الدالة لكل نموذج
cal(model1)
cal(model2)
cal(model3)
cal(model4)

# النتائج النهائية
FinalResult = pd.DataFrame(
    {"Algorithm": columns, "Accuracy": result1, "Recall": result2, "FScore": result3}
)

# رسم بياني
plt.figure(figsize=(20, 5))
plt.plot(FinalResult.Algorithm, result1, marker="o", label="Accuracy")
plt.plot(FinalResult.Algorithm, result2, marker="s", label="Recall")
plt.plot(FinalResult.Algorithm, result3, marker="^", label="F1")
plt.legend()
plt.title("Comparison of Algorithms")
plt.xlabel("Algorithms")
plt.ylabel("Score")
plt.grid(True)
plt.show()

print(FinalResult)
