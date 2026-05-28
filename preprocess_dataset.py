import os
import sys
import pandas as pd
from core.preprocessing import preprocess_indonesian

INPUT_PATH = os.path.join('dataset', 'GMaps_Review_Cleaned.csv')
OUTPUT_PATH = os.path.join('dataset', 'GMaps_Review_Preprocessed.csv')

df = pd.read_csv(INPUT_PATH)
print(f"Dataset loaded: {len(df)} rows", flush=True)

df.columns = df.columns.str.strip()

TEXT_COL = 'Review' if 'Review' in df.columns else df.columns[0]
df[TEXT_COL] = df[TEXT_COL].fillna('').astype(str)

print("Preprocessing with Sastrawi stemmer + stopword removal...", flush=True)
results = []
total = len(df)
for i, text in enumerate(df[TEXT_COL]):
    results.append(preprocess_indonesian(text))
    if (i + 1) % 500 == 0 or i == 0:
        print(f"  Progress: {i+1}/{total} ({((i+1)/total*100):.0f}%)", flush=True)

df['text_clean'] = results

print(f"\nSample hasil preprocessing:", flush=True)
for i in range(min(5, len(df))):
    print(f"  Asli   : {df[TEXT_COL].iloc[i][:80]}")
    print(f"  Bersih : {df['text_clean'].iloc[i][:80]}")
    print()

df.to_csv(OUTPUT_PATH, index=False)
print(f"Hasil preprocessing disimpan ke: {OUTPUT_PATH}", flush=True)
print(f"Total: {len(df)} review diproses", flush=True)
