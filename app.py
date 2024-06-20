import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Tải mô hình đã được đóng gói
with open('naive_bayes_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Tải TfidfVectorizer đã được huấn luyện
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

st.title('Phân loại tin nhắn Spam')

# Nhập một tin nhắn để phân loại
input_text = st.text_area("Nhập tin nhắn của bạn tại đây:")

if st.button('Phân loại'):
    # Chuyển đổi tin nhắn thành dạng vector
    transformed_text = tfidf.transform([input_text]).toarray()  # Chú ý phải dùng [input_text]
    # Dự đoán
    prediction = model.predict(transformed_text)
    # Xuất kết quả
    if prediction[0] == 0:
        st.write('Tin nhắn này không phải là spam')
    else:
        st.write('Tin nhắn này là spam')
