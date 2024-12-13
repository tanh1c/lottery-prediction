import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# pip install [package_name] -i https://pypi.tuna.tsinghua.edu.cn/simple
# venv\Scripts\activate
# Cấu hình cache cho Streamlit
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    def convert_date(date_str):
        date_str = date_str.strip('"')
        weekday, date = date_str.split(', ')
        return pd.to_datetime(date, format='%d/%m/%Y')
    
    df['Date'] = df['Date'].apply(convert_date)
    df['Numbers'] = df['Result'].apply(lambda x: [int(num) for num in x.split('�')[:6]])
    df['Special'] = df['Result'].apply(lambda x: int(x.split('�')[6]))
    return df

@st.cache_data
def create_features_by_type(numbers_array, prediction_type):
    features = []
    for numbers in numbers_array:
        base_features = list(numbers)
        
        if prediction_type in ['match3', 'match4']:
            pairs = [(numbers[i], numbers[j]) for i in range(6) for j in range(i+1, 6)]
            pair_sums = [sum(pair) for pair in pairs]
            pair_diffs = [abs(pair[0] - pair[1]) for pair in pairs]
            additional_features = [
                np.mean(pair_sums),
                np.std(pair_sums),
                np.mean(pair_diffs),
                np.std(pair_diffs)
            ]
            
        elif prediction_type in ['match5', 'jackpot2']:
            sorted_nums = sorted(numbers)
            diffs = np.diff(sorted_nums)
            additional_features = [
                np.mean(diffs),
                np.std(diffs),
                np.min(diffs),
                np.max(diffs),
                np.median(sorted_nums)
            ]
            
        else:  # jackpot1
            sorted_nums = sorted(numbers)
            diffs = np.diff(sorted_nums)
            gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
            additional_features = [
                np.mean(numbers),
                np.std(numbers),
                np.median(numbers),
                np.mean(diffs),
                np.std(diffs),
                max(gaps),
                min(gaps)
            ]
        
        features.append(np.concatenate([base_features, additional_features]))
    return np.array(features)

@st.cache_data
def prepare_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, :6])
    return np.array(X), np.array(y)

def build_model_by_type(seq_length, feature_dim, prediction_type):
    if prediction_type in ['match3', 'match4']:
        model = Sequential([
            LSTM(128, input_shape=(seq_length, feature_dim), return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(6, activation='sigmoid')
        ])
    
    elif prediction_type in ['match5', 'jackpot2']:
        model = Sequential([
            Bidirectional(LSTM(256, return_sequences=True), input_shape=(seq_length, feature_dim)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(128),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(6, activation='sigmoid')
        ])
    
    else:  # jackpot1
        model = Sequential([
            Bidirectional(LSTM(512, return_sequences=True), input_shape=(seq_length, feature_dim)),
            BatchNormalization(),
            Dropout(0.4),
            Bidirectional(LSTM(256, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.4),
            LSTM(128),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(6, activation='sigmoid')
        ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

@st.cache_data
def analyze_trends(df):
    frequent_numbers = {}
    for numbers in df['Numbers']:
        for num in numbers:
            frequent_numbers[num] = frequent_numbers.get(num, 0) + 1
    return sorted(frequent_numbers.items(), key=lambda x: x[1], reverse=True)

def plot_prediction_history(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Numbers'].apply(lambda x: np.mean(x)),
        mode='lines+markers',
        name='Trung bình các số'
    ))
    fig.update_layout(
        title='Xu hướng số qua các kỳ',
        xaxis_title='Ngày',
        yaxis_title='Giá trị trung bình'
    )
    return fig

def main():
    st.set_page_config(
        page_title="Dự Đoán Xổ Số Power 6/55",
        page_icon="🎲",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎲 Dự Đoán Xổ Số Power 6/55")
    
    # Sidebar
    st.sidebar.header("Tùy chọn")
    prediction_type = st.sidebar.selectbox(
        "Chọn loại dự đoán",
        ["Match 3", "Match 4", "Match 5", "Jackpot 2", "Jackpot 1"]
    )
    
    # Thêm các tùy chọn nâng cao
    with st.sidebar.expander("Tùy chọn nâng cao"):
        epochs = st.slider("Số epochs", 10, 200, 50)
        batch_size = st.slider("Batch size", 16, 64, 32)
        seq_length = st.slider(
            "Sequence length",
            5, 20,
            10 if prediction_type in ["Match 3", "Match 4"] else 15
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dự đoán kết quả")
        if st.button("Dự đoán ngay", use_container_width=True):
            with st.spinner('Đang phân tích dữ liệu...'):
                try:
                    # Load và xử lý dữ liệu
                    df = load_data('Data1.csv')
                    numbers = np.array(df['Numbers'].tolist())
                    
                    # Tạo features và dự đoán
                    features = create_features_by_type(
                        numbers,
                        prediction_type.lower().replace(" ", "")
                    )
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    features_scaled = scaler.fit_transform(features)
                    
                    X, y = prepare_sequences(features_scaled, seq_length)
                    model = build_model_by_type(
                        seq_length,
                        features.shape[1],
                        prediction_type.lower().replace(" ", "")
                    )
                    
                    # Training với early stopping
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                    
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Dự đoán và xử lý kết quả
                    last_sequence = features_scaled[-seq_length:]
                    prediction = model.predict(np.array([last_sequence]))
                    
                    numbers = np.clip(np.round(prediction[0] * 55), 1, 55).astype(int)
                    numbers = np.unique(numbers)[:6]
                    numbers.sort()
                    
                    # Hiển thị kết quả
                    st.success("Dự đoán hoàn tất!")
                    st.write("### 🎯 Các số được dự đoán:")
                    
                    # Hiển thị số dự đoán
                    cols = st.columns(6)
                    for i, num in enumerate(numbers):
                        with cols[i]:
                            st.markdown(f"""
                            <div style='background-color: #f0f2f6; 
                                      border-radius: 50%;
                                      width: 60px;
                                      height: 60px;
                                      display: flex;
                                      align-items: center;
                                      justify-content: center;
                                      font-size: 24px;
                                      font-weight: bold;'>
                                {num}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Hiển thị đồ thị training
                    st.write("### 📈 Quá trình training")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        name='Training Loss'
                    ))
                    fig.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        name='Validation Loss'
                    ))
                    fig.update_layout(
                        title='Loss qua các epochs',
                        xaxis_title='Epoch',
                        yaxis_title='Loss'
                    )
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Có lỗi xảy ra: {str(e)}")
    
    with col2:
        st.subheader("Thống kê")
        
        # Hiển thị xác suất trúng
        prob_dict = {
            "Match 3": "1/81 (1.23%)",
            "Match 4": "1/2,208 (0.045%)",
            "Match 5": "1/89,104 (0.001%)",
            "Jackpot 2": "1/4,831,613 (0.000021%)",
            "Jackpot 1": "1/28,989,675 (0.0000034%)"
        }
        
        st.info(f"Xác suất trúng {prediction_type}: {prob_dict[prediction_type]}")
        
        # Hiển thị top số xuất hiện nhiều
        df = load_data('Data1.csv')
        trends = analyze_trends(df)
        
        st.write("### 🔝 Top 10 số xuất hiện nhiều nhất")
        trend_cols = st.columns(2)
        for i, (num, count) in enumerate(trends[:10]):
            col_idx = i % 2
            with trend_cols[col_idx]:
                st.metric(f"Số {num}", f"{count} lần")
        
        # Hiển thị kết quả gần đây
        st.write("### 📅 Kết quả gần đây")
        recent_results = df.head(5)[['Date', 'Result']]
        st.dataframe(recent_results, use_container_width=True)
        
        # Hiển thị đồ thị xu hướng
        st.write("### 📊 Xu hướng")
        st.plotly_chart(plot_prediction_history(df), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>⚠️ <b>Lưu ý quan trọng:</b></p>
        <p>• Đây chỉ là công cụ dự đoán mang tính chất tham khảo</p>
        <p>• Kết quả xổ số là ngẫu nhiên và không thể dự đoán chính xác 100%</p>
        <p>• Vui lòng chơi có trách nhiệm và trong khả năng tài chính</p>
        <p style='font-size: 12px; color: #666;'>
            Phiên bản 1.0 - Cập nhật lần cuối: {}</p>
    </div>
    """.format(datetime.now().strftime("%d/%m/%Y")), unsafe_allow_html=True)

if __name__ == "__main__":
    main() 