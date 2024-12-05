import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model dan scaler
model = joblib.load('mobile_price_model.pkl')  # Pastikan file ini ada
scaler = joblib.load('mobile_price_scaler.pkl')  # Pastikan file ini ada

# Sidebar dengan wrap text
st.sidebar.markdown("")
st.sidebar.title("  \nPrediksi Harga Ponsel")
st.sidebar.markdown("- - - - - - - - - - - - - -")
st.sidebar.markdown("Dataset:  \n[Kaggle Mobile Price Classification Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data)  \n\nAuthor:  \nAgung Kurniawan")

# Header
st.title("Mobile Price Prediction App")

# View Dataset
st.subheader("Dataset Overview")
data = pd.read_csv('data/train.csv')  # Pastikan dataset berada di folder yang benar

# Periksa apakah data telah dimuat dengan benar
st.write(data.head())

# Visualization
st.subheader("Visualizations")
# Menggunakan 'price_range' untuk visualisasi
if 'price_range' in data.columns:
    # Membagi kategori harga menjadi dua (Murah, Mahal)
    data['price_category'] = data['price_range'].apply(lambda x: 1 if x == 0 else 2)
    st.bar_chart(data['price_category'].value_counts())  # Menggunakan 'price_category' untuk visualisasi
else:
    st.warning("Kolom 'price_range' tidak ditemukan dalam dataset.")

# Prediction Form
st.subheader("Make a Prediction")

# Input features sesuai dengan yang ada di dataset
battery_power = st.slider("Battery Power", 0, 2000, 1000)
ram = st.slider("RAM", 0, 4000, 2000)
px_height = st.slider("Pixel Height", 0, 2000, 1000)
px_width = st.slider("Pixel Width", 0, 2000, 1000)
internal_memory = st.slider("Internal Memory", 4, 128, 32)

if st.button("Predict"):
    # Membuat fitur untuk prediksi dengan 5 fitur yang digunakan
    features = [[battery_power, ram, px_height, px_width, internal_memory]]
    
    # Menormalisasi fitur menggunakan scaler yang telah dilatih
    features_scaled = scaler.transform(features)
    
    # Melakukan prediksi dengan fitur yang sudah dinormalisasi
    prediction = model.predict(features_scaled)
    
    # Ubah prediksi menjadi dua kategori harga (Murah = 1, Mahal = 2)
    price_category = 1 if prediction[0] == 0 else 2
    
    # Menampilkan hasil prediksi
    price_labels = ["Murah", "Mahal"]
    st.success(f"Prediksi Harga Ponsel: {price_labels[price_category - 1]}")