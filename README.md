# Prediksi Nilai Pembelian Pelanggan – Shopping Trends

Proyek Machine Learning (regresi) untuk memprediksi **`Purchase Amount (USD)`** berdasarkan karakteristik pelanggan dan transaksi pada dataset *Customer Shopping Trends*.

## Dataset
- File: `shopping_trends_updated.csv` (3.900 baris, 18 kolom, tanpa missing value dan tanpa duplikat).
- Numerik: `Customer ID`, `Age`, `Purchase Amount (USD)`, `Review Rating`, `Previous Purchases`.
- Kategorikal: `Gender`, `Item Purchased`, `Category`, `Location`, `Size`, `Color`, `Season`, `Subscription Status`, `Shipping Type`, `Discount Applied`, `Promo Code Used`, `Payment Method`, `Frequency of Purchases`.
- **Target:** `Purchase Amount (USD)`.
- `Customer ID` dibuang karena hanya identifier.

## Penjelasan Kode (Notebook)
1. **Import library** – pandas, numpy, seaborn, matplotlib, scikit-learn, joblib, pickle.
2. **Loading & eksplorasi awal** – `head()`, `info()`, `describe(include="all")`.
3. **Missing value** – dicek; numerik diisi median, kategorikal diisi `Unknown` (pada dataset ini tidak ada yang kosong).
4. **Duplikat** – dicek dan dihapus bila ada.
5. **Feature engineering** – `Customer ID` dibuang, target dipisah agar tidak terjadi data leakage.
6. **Outlier** – boxplot dan IQR pada `Age`, `Purchase Amount (USD)`, `Review Rating`, `Previous Purchases`; hanya dicek, tidak dihapus.
7. **Encoding** – One-Hot Encoding (`drop_first=True`) untuk semua kolom kategorikal.
8. **EDA** – histogram target dan fitur numerik, countplot kategori, heatmap korelasi, korelasi fitur terhadap target.
9. **Split data** – 80% train, 20% test (`random_state=42`).
10. **Standardisasi** – `StandardScaler` di-fit hanya pada data train.
11. **Modeling** – Linear Regression (data terstandardisasi), Decision Tree (`max_depth=6`), Random Forest (`n_estimators=200`), Gradient Boosting.
12. **Evaluasi** – MAE, MSE, RMSE, R², plus baseline yang selalu menebak rata-rata.
13. **Feature importance** – 10 fitur terpenting dari Gradient Boosting.
14. **Simpan model** – model terbaik (RMSE terendah) disimpan ke `best_model.joblib` / `best_model.pkl`, scaler ke `scaler.joblib`.

## Hasil
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 20.90 | 23.99 | -0.028 |
| Decision Tree | 21.26 | 24.50 | -0.073 |
| Random Forest | 21.00 | 24.04 | -0.033 |
| Gradient Boosting | 20.79 | 23.81 | -0.013 |
| Baseline (rata-rata) | 20.76 | 23.73 | -0.006 |

Semua model berkinerja sekitar baseline (R² ≈ 0 / sedikit negatif). Artinya fitur yang tersedia hampir tidak punya hubungan prediktif dengan nilai pembelian pada dataset ini. Dataset ini kemungkinan besar sintetis (nilai pembelian tampak acak), sehingga hasil ini wajar dan bukan kesalahan kode. Gradient Boosting hanya sedikit lebih baik dibanding model lain.

## Sumber Dataset
Customer Shopping Trends Dataset (Kaggle). Versi `shopping_trends_updated.csv` memakai kolom `Payment Method` yang sama dengan `Preferred Payment Method` pada `shopping_trends.csv`.

## File Proyek
- `Laporan_Machine_Learning_Dava.ipynb` – notebook utama
- `shopping_trends_updated.csv` – dataset
- `best_model.joblib`, `best_model.pkl`, `scaler.joblib` – model dan scaler tersimpan
- `README.md` – dokumentasi

## Cara Menjalankan
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook Laporan_Machine_Learning_Dava.ipynb
```
Pastikan CSV berada di folder yang sama dengan notebook, lalu jalankan semua cell berurutan.

## Catatan
- Scaling hanya untuk Linear Regression; model berbasis pohon memakai data asli.
- Outlier tidak dihapus karena nilai ekstrem masih valid.
