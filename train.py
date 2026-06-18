import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle, os

# ── Load Dataset ──────────────────────────────────────────
df = pd.read_csv('dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(' Dataset loaded! Shape:', df.shape)

# ── Data Cleaning ─────────────────────────────────────────
df.drop(columns=['Over18', 'StandardHours',
                 'EmployeeCount', 'EmployeeNumber'],
        errors='ignore', inplace=True)

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print(' Data cleaned! Shape:', df.shape)

# ── EDA Plot 1 — Target Distribution ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df['Attrition'].value_counts()
axes[0].bar(['No Attrition', 'Attrition'], counts.values,
            color=['#2196F3', '#F44336'], edgecolor='black')
axes[0].set_title('Attrition Distribution', fontweight='bold')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')
axes[1].pie(counts.values, labels=['No Attrition', 'Attrition'],
            autopct='%1.1f%%', colors=['#2196F3', '#F44336'])
axes[1].set_title('Attrition Percentage', fontweight='bold')
plt.suptitle('Target Variable Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model/eda_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ── EDA Plot 2 — Correlation Heatmap ─────────────────────
plt.figure(figsize=(16, 12))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlBu_r', center=0,
            linewidths=0.5, annot_kws={'size': 7})
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model/eda_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Feature Engineering ───────────────────────────────────
top_features = [
    'OverTime', 'MaritalStatus', 'StockOptionLevel', 'Age',
    'JobLevel', 'YearsWithCurrManager', 'JobSatisfaction',
    'YearsAtCompany', 'MonthlyIncome', 'TotalWorkingYears'
]

X = df[top_features]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_sc, y_train)
print('Features ready!')

# ── Train Model ───────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_res, y_res)

y_pred = rf.predict(X_test_sc)
y_prob = rf.predict_proba(X_test_sc)[:,1]

print(f' Accuracy : {round(accuracy_score(y_test, y_pred)*100, 2)}%')
print(f' ROC-AUC  : {round(roc_auc_score(y_test, y_prob)*100, 2)}%')
print()
print(classification_report(y_test, y_pred,
      target_names=['No Attrition', 'Attrition']))

# ── Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.title('Confusion Matrix', fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Save Model ────────────────────────────────────────────
os.makedirs('model', exist_ok=True)

with open('model/attrition_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(' Model saved → model/attrition_model.pkl')
print(' Scaler saved → model/scaler.pkl')