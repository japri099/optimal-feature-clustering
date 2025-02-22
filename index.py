# Import Libraries
import pandas as pd # Untuk manipulasi data dan pengolahan data tabular
import numpy as np # Untuk operasi numerik
#  Untuk visualisasi
import matplotlib.pyplot as plt
import seaborn as sns
# Untuk pembelajaran mesin dan evaluasi model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.ensemble import IsolationForest
# Untuk membangun model neural network
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
# Untuk mengupload file di Google Colab
from google.colab import files

# STEP 1: Upload Dataset
print("Upload Dataset:")
uploaded = files.upload()
dataset_name = list(uploaded.keys())[0]
data = pd.read_csv(dataset_name) # Membaca file CSV ke dalam DataFrame
print(f"Dataset '{dataset_name}' berhasil diunggah!")

# STEP 2: Data tinjau
# Menampilkan lima baris pertama dari dataset dan informasi tentang kolom-kolom dalam dataset
print("\n5 baris data pertama:")
print(data.head()) # Berasal dari pustaka Pandas dan digunakan untuk menampilkan lima baris pertama dari DataFrame data
print("\nDataset Informasi:")
print(data.info()) # Memberikan informasi tentang kolom-kolom dalam dataset

# STEP 3: Menangani Data yang Hilang
# Memeriksa dan mengisi nilai yang hilang dengan rata-rata kolom jika ada
print("\nMemeriksa data yang hilang...")
missing_count = data.isnull().sum()
# isnull : Mengembalikan DataFrame dengan nilai boolean, di mana True menunjukkan bahwa nilai tersebut adalah NaN (Not a Number)
# sum(): Menghitung jumlah nilai True (nilai yang hilang) untuk setiap kolom dalam DataFrame
print(missing_count)
if missing_count.sum() > 0: # Kondisi ini memeriksa apakah ada nilai yang hilang dalam dataset
    print("Mengisi nilai yang hilang dengan rata-rata...")
    data.fillna(data.mean(), inplace=True) # Jika ada nilai yang hilang, akan diisi dengan rata-rata kolom
    # data.mean(): Menghitung rata-rata untuk setiap kolom numerik dalam DataFrame
    # data.fillna(...): Mengisi nilai NaN dalam DataFrame. Setiap nilai hilang akan diisi dengan rata-rata kolom terkait

# STEP 4: Menangani Outlier
print("\nMenangani outlier menggunakan Isolation Forest...")
# Menggunakan metode Isolation Forest untuk mendeteksi dan menghapus outlier dari dataset
iso = IsolationForest(contamination=0.1, random_state=42) # Mendeteksi outlier dalam dataset
# IsolationForest: Algoritma pembelajaran mesin yang digunakan untuk mendeteksi outlier. Membangun pohon keputusan acak dan mengisolasi titik data. Titik data yang lebih mudah diisolasi dianggap sebagai outlier
# contamination=0.1: Menunjukkan proporsi data yang dianggap sebagai outlier (10% dalam hal ini). Ditentukan berdasarkan pemahaman tentang dataset
# random_state=42: Menetapkan seed untuk memastikan reproduktifitas hasil, sehingga setiap kali kode dijalankan, hasilnya akan sama
outlier_flags = iso.fit_predict(data)
# .fit_predict(data): Melatih model pada dataset dan kemudian memprediksi apakah setiap titik data adalah outlier atau tidak
data = data[outlier_flags == 1]
print(f"Data setelah penghilangan outlier: {data.shape}")
# data[outlier_flags == 1]: Menyaring DataFrame data, hanya menyimpan baris-baris di mana outlier_flags bernilai 1

# Hasil Pemisahan (jika ada)
# Memisahkan kolom 'Outcome' (y) dan sisa kolom sebagai fitur (X)
if 'Outcome' in data.columns: # Jika kolom Outcome ada dalam DataFrame, maka kolom tersebut dipisahkan menjadi variabel target y
    y = data['Outcome']
    X = data.drop(columns=['Outcome'])
else: # Jika kolom Outcome tidak ditemukan, maka y diatur ke None, dan seluruh DataFrame disalin ke X
    y = None
    X = data.copy()

# Menormalkan Data
# Menggunakan MinMaxScaler untuk menormalkan fitur agar berada dalam rentang [0,1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# STEP 5: Model Autoencoder untuk Representasi Fitur
input_dim = X_scaled.shape[1] # Menentukan dimensi input berdasarkan jumlah fitur dalam dataset yang telah dinormalisasi (X_scaled)
encoding_dim = 5  # Dimensi fitur yang berkurang
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder Model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Penghentian untuk mencegah Overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Autoencoder
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, X_val),
                callbacks=[early_stop],
                verbose=1)

# Ekstark Fitur
encoder = Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X_scaled)

# STEP 6: Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_encoded)

# Menambahkan Kluster ke Data
data['Cluster'] = clusters

# Evaluasi Clustering
silhouette_avg = silhouette_score(X_encoded, clusters)
print(f"\nSkor Siluet: {silhouette_avg:.3f}")

# Jika ada kolom 'Outcome', hitung ARI
if y is not None:
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(y, clusters)  # Hitung ARI antara y (label asli) dengan clusters
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

# Visualisasi Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_encoded[:, 0], y=X_encoded[:, 1], hue=clusters, palette="Set2", s=50)
plt.title("Visualisasi Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Cluster", loc='upper right')
plt.show()

# STEP 7: Matriks Korelasi Plot
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Koreksi Matriks")
plt.show()

# STEP 8: Mengevaluasi Model (Jika Ada Hasil)
if y is not None:
    y_pred = kmeans.predict(X_encoded)
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title("Confusion Matrix")
    plt.xlabel("Prediksi")
    plt.ylabel("Sebenarnya")
    plt.show()
else:
    print("Kolom 'Outcome' tidak ditemukan; evaluasi klasifikasi dilewati.")

# STEP 9: Menangani Masalah Model

def detect_model_issues(history):
    # Menganalisis pelatihan dan validasi 'loss' untuk mendeteksi underfitting atau overfitting
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot pelatihan dan validasi 'loss'
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    # Mendeteksi pola underfitting dan overfitting
    if train_loss[-1] > val_loss[-1] * 1.2:  # Overfitting saat kehilangan pelatihan jauh lebih rendah daripada kehilangan validasi
        print("\nOverfitting Terdeteksi!")
    elif val_loss[-1] > train_loss[-1] * 1.5:  # Underfitting ketika kehilangan validasi secara signifikan lebih rendah daripada kehilangan pelatihan
        print("\nUnderfitting Terdeteksi!")
    else:
        print("\nModel tampaknya sudah Seimbang! pelatihan dan validasi 'loss' seimbang.")

# Panggil fungsi untuk menganalisis kinerja pelatihan dan validasi
detect_model_issues(autoencoder.history)

# STEP 10: Statistik Deskriptif untuk Setiap Klaster
print("\nStatistik Deskriptif untuk Setiap Klaster:")
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(cluster_data.describe())

# STEP 11: Outcome Analysis by Cluster
# Menilai kecenderungan diabetes dalam tiap cluster
cluster_outcome_analysis = data.groupby('Cluster')['Outcome'].mean()
print("\nHasil Rata-rata untuk Setiap Klaster (Nilai yang lebih tinggi menunjukkan diabetes yang lebih sering terjadi):")
print(cluster_outcome_analysis)

# Filter data yang berada dalam Cluster 0, 1, dan 2
cluster_0_data = data[data['Cluster'] == 0]
cluster_1_data = data[data['Cluster'] == 1]
cluster_2_data = data[data['Cluster'] == 2]

# Tampilkan beberapa baris pertama data yang tergabung dalam Cluster 0, 1, dan 2
print(cluster_0_data.head())
print(cluster_1_data.head())
print(cluster_2_data.head())

# Tampilkan jumlah data dalam masing-masing Cluster
print(f"Jumlah data di Cluster 0: {cluster_0_data.shape[0]}")
print(f"Jumlah data di Cluster 1: {cluster_1_data.shape[0]}")
print(f"Jumlah data di Cluster 2: {cluster_2_data.shape[0]}")

# STEP 12: Wawasan Tambahan: Fitur Paling Berpengaruh untuk Setiap Klaster
# Memeriksa fitur utama yang dapat menjelaskan perbedaan dalam Outcome per Cluster
feature_importance = X_encoded.mean(axis=0)  # Rata rata dari sample
print("\nRata-rata kepentingan fitur untuk cluster:")
print(feature_importance)

# Kesimpulan tentang kecenderungan diabetes berdasarkan cluster
print("\nKesimpulan:")
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    cluster_outcome = cluster_data['Outcome'].mean()
    print(f"Cluster {cluster}:")
    if cluster_outcome > 0.5:
        print(f"  Kecenderungan besar terhadap diabetes (Outcome = 1). Ciri-ciri: Rata-rata Glucose = {cluster_data['Glucose'].mean():.2f}, BMI = {cluster_data['BMI'].mean():.2f}, dll.")
    else:
        print(f"  Kecenderungan kecil terhadap diabetes (Outcome = 0). Ciri-ciri: Rata-rata Glucose = {cluster_data['Glucose'].mean():.2f}, BMI = {cluster_data['BMI'].mean():.2f}, dll.")
