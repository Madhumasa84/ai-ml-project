import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
data = pd.read_csv('data.csv')
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

data[num_cols] = imputer_num.fit_transform(data[num_cols])
data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
X = data.drop(columns=['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'])
y = data[['xyz_vaccine', 'seasonal_vaccine']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rf_xyz = RandomForestClassifier(random_state=42)
rf_seasonal = RandomForestClassifier(random_state=42)

rf_xyz.fit(X_train, y_train['xyz_vaccine'])
rf_seasonal.fit(X_train, y_train['seasonal_vaccine'])

xyz_pred_probs = rf_xyz.predict_proba(X_test)[:, 1]
seasonal_pred_probs = rf_seasonal.predict_proba(X_test)[:, 1]
xyz_auc = roc_auc_score(y_test['xyz_vaccine'], xyz_pred_probs)
seasonal_auc = roc_auc_score(y_test['seasonal_vaccine'], seasonal_pred_probs)

print(f'ROC AUC for xyz_vaccine: {xyz_auc}')
print(f'ROC AUC for seasonal_vaccine: {seasonal_auc}')
print(f'Mean ROC AUC: {(xyz_auc + seasonal_auc) / 2}'
xyz_pred_probs_full = rf_xyz.predict_proba(X)[:, 1]
seasonal_pred_probs_full = rf_seasonal.predict_proba(X)[:, 1]
submission = pd.DataFrame({
    'respondent_id': data['respondent_id'],
    'xyz_vaccine': xyz_pred_probs_full,
    'seasonal_vaccine': seasonal_pred_probs_full
})
submission.to_csv('submission.csv', index=False)





