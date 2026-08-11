import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Documents\DSA\DailyDialog.csv")

print(df.isnull().sum())

X = df["text"]
y = df["sentiment"]

le = LabelEncoder()
y = pd.Series(le.fit_transform(y),name="sentiment")

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

tv = TfidfVectorizer()
X_train = tv.fit_transform(X_train)
X_test = tv.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train,y_train)

pred = lr.predict(X_test)
pred_labels = le.inverse_transform(pred)
print(pred_labels)

acc = accuracy_score(y_test,pred)
print("Accuracy: ",acc)

precision = precision_score(y_test,pred,average="macro")
print("Precision: ",precision)

recall = recall_score(y_test,pred,average="macro")
print("Recall: ",recall)

f1 = f1_score(y_test,pred,average="macro")
print("F1 Score: ",f1)

cm = confusion_matrix(y_test,pred)
plt.title("Confusion Matrix")
sns.heatmap(cm,annot=True,xticklabels=le.classes_,yticklabels=le.classes_)
plt.show()

report = classification_report(y_test,pred,target_names=le.classes_)
print("Classification report: ")
print(report)

#Prediction on completely new text
new_text = ["I am extremely happy today"]

test_data = tv.transform(new_text)

pred_test = lr.predict(test_data)
print(le.inverse_transform(pred_test))