import streamlit as st
import pandas as pd
import pickle
with open('train_model.pkl', 'rb') as f:
    model = pickle.load(f)
st.title("Cảnh báo học vụ")
st.write("Vui lòng nhập thông tin sinh viên")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Độ tuổi", min_value=18, max_value=50, value=20)
        tuition = st.number_input("Nợ học phí (VNĐ)", value=0)
    with col2:
        count_f = st.number_input("Số môn bị điểm F", value=0)
        training_score = st.number_input("Điểm rèn luyện của bạn", value=70)
    submit = st.form_submit_button("Dự đoán")

if submit:
    input_data = [age, tuition, count_f, training_score] + [10]*40 
    prediction = model.predict([input_data])
    probability = model.predict_proba([input_data])

    if prediction[0] == 1:
        st.error(f"CẢNH BÁO: Sinh viên có nguy cơ bị xử lý học vụ! (Xác suất: {probability[0][1]:.2%})")
    else:
        st.success(f"An toàn: Sinh viên có tình trạng học tập biunhf thường. (Xác suất an toàn: {probability[0][0]:.2%})")
