import os
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = os.path.join('saved_models', 'svm_sentimen_3kelas.pkl')
DATA_PATH = os.path.join('dataset', 'GMaps_Review_Preprocessed.csv')
OUTPUT_PATH = os.path.join('dataset', 'GMaps_Review_Labeled.csv')
THRESHOLD = 0.70

data = joblib.load(MODEL_PATH)
model = data['model']
vectorizer = data['vectorizer']

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {len(df)} rows")

df.columns = df.columns.str.strip()

TEXT_COL = 'text_clean'
if TEXT_COL not in df.columns:
    TEXT_COL = 'Review' if 'Review' in df.columns else df.columns[0]
    print(f"Warning: 'text_clean' not found, using '{TEXT_COL}'")
    df[TEXT_COL] = df[TEXT_COL].fillna('').astype(str).apply(clean_text)
else:
    df[TEXT_COL] = df[TEXT_COL].fillna('').astype(str)

X_vec = vectorizer.transform(df[TEXT_COL])

sentimen_list = []
confidence_list = []

for i in range(X_vec.shape[0]):
    proba = model.predict_proba(X_vec[i:i+1])[0]
    confidence = proba.max()
    if confidence < THRESHOLD:
        sentimen_list.append('NETRAL')
    else:
        sentimen_list.append(model.classes_[proba.argmax()])
    confidence_list.append(confidence)

df['sentimen'] = sentimen_list
df['keyakinan'] = (np.array(confidence_list) * 100).round(2)

print("\nDistribusi sentimen:")
print(df['sentimen'].value_counts())
print()

df.to_csv(OUTPUT_PATH, index=False)
print(f"Hasil disimpan ke: {OUTPUT_PATH}")
print(f"Total: {len(df)} review")
for s in ['POSITIF', 'NEGATIF', 'NETRAL']:
    print(f"  {s}: {(df['sentimen']==s).sum()}")
