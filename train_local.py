import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from core.preprocessing import clean_text

EMOSI_MAP = {
    'SENANG': 'POSITIF',
    'CINTA': 'POSITIF',
    'SEDIH': 'NEGATIF',
    'TAKUT': 'NEGATIF',
    'MARAH': 'NEGATIF',
    'JIJIK': 'NEGATIF',
}

DATA_PATH = os.path.join('dataset', 'GMaps_Review_Classified.csv')
MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_sentimen_3kelas.pkl')

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {len(df)} rows")

df.columns = df.columns.str.strip()

TEXT_COL = 'Review' if 'Review' in df.columns else df.columns[0]
LABEL_COL = 'labels' if 'labels' in df.columns else \
            'label_emosi' if 'label_emosi' in df.columns else \
            'sentimen' if 'sentimen' in df.columns else \
            'label' if 'label' in df.columns else df.columns[1]

print(f"Using text column : {TEXT_COL}")
print(f"Using label column: {LABEL_COL}")

df[TEXT_COL] = df[TEXT_COL].fillna('').astype(str)
df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

df['sentimen_3kelas'] = df[LABEL_COL].map(EMOSI_MAP).fillna('NETRAL')
print("Label distribution (3 kelas):")
print(df['sentimen_3kelas'].value_counts())

X = df[TEXT_COL]
y = df['sentimen_3kelas']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\nTraining SVM (kernel=linear, probability=True)...")
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(f"Akurasi: {accuracy:.4f}")
print(classification_report(y_test, y_pred, digits=4))

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump({'model': model, 'vectorizer': vectorizer}, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
print("Training completed successfully!")
