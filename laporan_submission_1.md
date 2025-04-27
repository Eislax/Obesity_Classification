# Laporan Proyek Machine Learning - Muhammad Rafi Ilham

## Domain Proyek

Masalah obesitas merupakan salah satu tantangan kesehatan global yang signifikan. Obesitas tidak hanya memengaruhi kualitas hidup individu tetapi juga meningkatkan risiko penyakit kronis seperti diabetes, hipertensi, dan penyakit jantung. Oleh karena itu, deteksi dini status obesitas sangat penting untuk mencegah komplikasi lebih lanjut. Dalam proyek ini, kami menggunakan dataset **Obesity Classification** untuk membangun model machine learning yang dapat memprediksi status obesitas seseorang berdasarkan faktor-faktor seperti pola makan, aktivitas fisik, riwayat keluarga, dan gaya hidup lainnya.

**Rubrik/Kriteria Tambahan (Opsional)**:
- **Mengapa Masalah Ini Harus Diselesaikan:**  
  Deteksi dini status obesitas dapat membantu dalam merancang intervensi medis yang tepat waktu, mengurangi risiko penyakit terkait obesitas, dan meningkatkan kualitas hidup individu.
- **Referensi:**  
  Menurut WHO (World Health Organization), prevalensi obesitas di seluruh dunia telah meningkat dua kali lipat sejak tahun 1980. Penelitian oleh [Al-Ghamdi et al. (2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7532674/) menunjukkan bahwa faktor genetik, perilaku makan, dan aktivitas fisik memiliki pengaruh signifikan terhadap status obesitas.

## Business Understanding

### Problem Statements
- **Masalah Utama:** Sulitnya mendeteksi status obesitas secara dini tanpa analisis data yang sistematis.
- **Dampak Masalah:** Kurangnya pemahaman tentang faktor-faktor risiko obesitas dapat menyebabkan penundaan intervensi medis, yang berpotensi meningkatkan risiko penyakit terkait obesitas.

### Goals
- Membangun model machine learning yang dapat memprediksi status obesitas berdasarkan fitur-fitur yang relevan.
- Memberikan wawasan tentang faktor-faktor utama yang memengaruhi status obesitas.

**Rubrik/Kriteria Tambahan (Opsional)**:

### Solution Statements
1. **Baseline Model:** Menggunakan algoritma **Random Forest Classifier** sebagai baseline model untuk memprediksi status obesitas.
2. **Improvement Model:** Melakukan hyperparameter tuning pada **Gradient Boosting Classifier** untuk meningkatkan performa model.
3. **Alternative Algorithm:** Membandingkan performa model dengan **K-Nearest Neighbors (KNN)** untuk memastikan solusi optimal.

Solusi yang diberikan dievaluasi menggunakan metrik evaluasi seperti **Akurasi, Precision, Recall, dan F1-Score**.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Obesity Classification Dataset**, yang dapat diunduh dari [Kaggle](https://www.kaggle.com/datasets/mohithsairamreddy/obesity-classification). Dataset ini mencakup 2111 sampel dengan 17 fitur yang mencakup informasi demografis, kebiasaan makan, aktivitas fisik, dan status obesitas.

### Variabel-variabel pada Obesity Classification Dataset adalah sebagai berikut:
- **Gender:** Jenis kelamin (Female/Male).
- **Age:** Usia individu.
- **Height:** Tinggi badan dalam cm.
- **Weight:** Berat badan dalam kg.
- **Family History with Overweight:** Riwayat keluarga dengan kelebihan berat badan (yes/no).
- **Frequent consumption of high caloric food:** Kebiasaan mengonsumsi makanan tinggi kalori (yes/no).
- **Consumption of alcohol:** Kebiasaan konsumsi alkohol (no/Sometimes/Frequently/Always).
- **Obesity:** Status obesitas (target variabel).

**Rubrik/Kriteria Tambahan (Opsional)**:
- **Exploratory Data Analysis (EDA):**
  - Visualisasi distribusi tinggi dan berat badan menunjukkan hubungan linier antara kedua variabel tersebut.
  - Analisis distribusi status obesitas menunjukkan bahwa sebagian besar individu memiliki status obesitas tipe II dan III.
  - Analisis Distribusi Fitur Numerik: Memahami rentang nilai dan distribusi fitur numerik.
  - Pembulatan dan Identifikasi Nilai Unik: Membulatkan nilai numerik dan menampilkan nilai unik.
  - Mapping Nilai Kategorikal: Mengganti nilai numerik dengan label kategorikal yang lebih deskriptif.
  - Visualisasi Tinggi dan Berat Badan: Membandingkan distribusi tinggi dan berat badan berdasarkan gender.
  - Hubungan Tinggi dan Berat Badan: Menunjukkan hubungan linier antara tinggi dan berat badan.
  - Analisis Status Obesitas: Memahami distribusi status obesitas secara keseluruhan dan berdasarkan gender.
  - Visualisasi Kebiasaan Makan dan Aktivitas Fisik: Memahami pola kebiasaan makan dan aktivitas fisik.

## Data Preparation

Pada tahap ini, dilakukan beberapa teknik data preparation untuk mempersiapkan dataset agar siap digunakan untuk pemodelan.

1. **Penanganan Missing Values:**  
   - Dataset diperiksa untuk menemukan nilai-nilai yang hilang. Tidak ada missing values dalam dataset ini, sehingga langkah ini tidak diperlukan. Namun, jika ada missing values, teknik seperti imputasi atau penghapusan baris/kolom dapat digunakan.

2. **Encoding Variabel Kategorikal:**  
   - **Binary Encoding:** Kolom seperti `Frequent consumption of high caloric food` diubah dari nilai `yes/no` menjadi `1/0`.  
     ```python
     new_yesno = {'yes': 1, 'no': 0}
     data['Smoke'] = data['Smoke'].replace(new_yesno)
     ```
   - **Ordinal Encoding:** Kolom seperti `Frequency of consumption of vegetables` diubah menjadi nilai numerik berdasarkan urutan logis (`Never=0`, `Sometimes=1`, `Always=2`).  
   - **One-Hot Encoding:** Kolom seperti `Transportation used` diubah menjadi beberapa kolom biner untuk menghindari masalah ordinality.  
     ```python
     data = pd.get_dummies(data, columns=['Transportation used'], drop_first=True)
     ```

3. **Feature Scaling:**  
   - Fitur numerik seperti `Height`, `Weight`, dan `Age` dinormalisasi menggunakan `StandardScaler`.  
     ```python
     scaler = StandardScaler()
     data[['Height', 'Weight', 'Age']] = scaler.fit_transform(data[['Height', 'Weight', 'Age']])
     ```
   - **Alasan:** Normalisasi penting untuk menghindari dominasi fitur dengan rentang besar terhadap proses pembelajaran model.

4. **Feature Selection:**  
   - **Metode:** ANOVA F-test digunakan untuk memilih 10 fitur terbaik. ANOVA F-test mengukur hubungan statistik antara setiap fitur independen dan target variabel, dengan membandingkan varians antar kelas.  
   - **Alasan:** Metode ini dipilih karena cocok untuk dataset dengan target kategorikal seperti pada kasus ini.  
   - **Fitur Terpilih:** Fitur yang dipilih meliputi `Height`, `Weight`, `Age`, `Family History with Overweight`, dll., yang memiliki hubungan signifikan dengan status obesitas.

5. **Split Dataset:**  
   - Dataset dibagi menjadi training set (80%) dan testing set (20%) menggunakan `train_test_split`.  
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
     ```
   - **Alasan:** Pembagian ini memastikan bahwa model dapat dievaluasi secara objektif pada data yang belum pernah dilihat. Cross-validation juga dapat digunakan untuk meningkatkan keandalan model.

**Rubrik/Kriteria Tambahan (Opsional):**
- **Proses Data Preparation:**  
  - **Encoding:** Mengubah variabel kategorikal menjadi numerik agar dapat diproses oleh algoritma machine learning.  
  - **Feature Scaling:** Menghindari dominasi fitur numerik dengan rentang besar.  
  - **Feature Selection:** Mengurangi dimensi data untuk meningkatkan efisiensi komputasi dan fokus pada fitur yang relevan.  

Proses data preparation dalam proyek ini mencakup beberapa tahapan penting yang bertujuan untuk memastikan dataset siap digunakan untuk pemodelan machine learning. Pertama, penanganan missing values dilakukan untuk mengatasi data yang tidak lengkap. Meskipun dataset ini tidak memiliki missing values, langkah ini tetap relevan karena data yang hilang dapat menyebabkan bias atau ketidakakuratan dalam analisis. Selanjutnya, encoding variabel kategorikal dilakukan untuk mengonversi nilai teks menjadi numerik, seperti mengganti yes/no dengan 1/0 menggunakan binary encoding, serta menerapkan ordinal encoding dan one-hot encoding untuk kolom dengan tingkatan logis atau tanpa urutan. Encoding diperlukan karena algoritma machine learning hanya dapat memproses data numerik.

Selain itu, feature scaling diterapkan pada fitur numerik seperti Height, Weight, dan Age menggunakan StandardScaler. Langkah ini penting untuk menormalkan rentang nilai fitur, sehingga fitur dengan skala besar tidak mendominasi proses pembelajaran model. Label encoding juga diterapkan pada target variable (Obesity) untuk mengonversi label kategorikal menjadi nilai numerik unik. Terakhir, dataset dibagi menjadi training set dan test set menggunakan train_test_split untuk memastikan bahwa model dapat dievaluasi secara objektif pada data yang belum pernah dilihat.

Alasan utama melakukan tahapan data preparation ini adalah untuk memastikan bahwa dataset berada dalam format yang sesuai untuk pemodelan machine learning. Dataset yang tidak diproses dengan baik dapat menyebabkan hasil yang bias, performa model yang buruk, atau bahkan kegagalan dalam pelatihan model. Dengan melakukan preprocessing yang tepat, kita dapat meningkatkan akurasi model, mengurangi risiko overfitting, dan memastikan bahwa model dapat belajar pola data secara efisien dan adil.

## Modeling

Model machine learning yang digunakan dalam proyek ini adalah:
1. **Random Forest Classifier:** Ensemble method yang stabil dan kuat.
2. **Gradient Boosting Classifier:** Ensemble method dengan kemampuan memperbaiki kesalahan iteratif.
3. **K-Nearest Neighbors (KNN):** Instance-based learning untuk membandingkan performa.

### Parameter yang Digunakan
- **Random Forest:**  
  - `n_estimators=100`: Jumlah pohon dalam ensemble. Nilai ini dipilih karena memberikan keseimbangan antara kompleksitas model dan performa.  
  - `random_state=42`: Untuk memastikan hasil yang konsisten selama pelatihan.
- **Gradient Boosting:**  
  - `learning_rate=0.1`: Mengontrol kontribusi setiap pohon. Nilai ini dipilih karena memberikan hasil yang stabil tanpa overfitting.  
  - `n_estimators=100`: Jumlah pohon dalam ensemble.
- **KNN:**  
  - `n_neighbors=5`: Jumlah tetangga terdekat yang digunakan untuk prediksi. Nilai ini dipilih karena memberikan keseimbangan antara bias dan variance.

**Rubrik/Kriteria Tambahan (Opsional)**:
- **Kelebihan dan Kekurangan Algoritma:**
  - **Random Forest:** Stabil, tetapi kurang sensitif terhadap hubungan non-linear.
  - **Gradient Boosting:** Lebih akurat, tetapi lebih lambat dalam pelatihan.
  - **KNN:** Mudah diimplementasikan, tetapi sensitif terhadap noise.
- **Model Terbaik:** Model **K-Nearest Neighbors (KNN)** dipilih sebagai model terbaik karena memiliki akurasi tertinggi (96%) dibandingkan model lainnya.

## Evaluation

Metrik evaluasi yang digunakan dalam proyek ini adalah:
- **Akurasi:** Proporsi prediksi yang benar.
- **Precision:** Kemampuan model untuk membuat prediksi positif yang benar.
- **Recall:** Kemampuan model untuk mengidentifikasi semua kasus positif.
- **F1-Score:** Rata-rata harmonik antara precision dan recall.

### Hasil Evaluasi
| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest       | 94%      | 94%       | 94%    | 94%      |
| Gradient Boosting   | 95%      | 94%       | 95%    | 95%      |
| K-Nearest Neighbors | 96%      | 95%       | 96%    | 95%      |

**Rubrik/Kriteria Tambahan (Opsional)**:
- **Penjelasan Formula Metrik:**
  - **Akurasi:** $$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Predictions}}$$
  - *Penjelasan* =  Akurasi adalah proporsi prediksi yang benar dibandingkan dengan total jumlah prediksi.
  - **Precision:** $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$
  - *Penjelasan* =  Precision adalah proporsi prediksi positif yang benar dibandingkan dengan total prediksi positif.
  - **Recall:** $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} $$
  - *Penjelasan* =  Recall adalah proporsi kasus positif aktual yang berhasil diidentifikasi oleh model.
  - **F1-Score:** $$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
  - *Penjelasan* =  F1-Score adalah rata-rata harmonik antara precision dan recall.

---
