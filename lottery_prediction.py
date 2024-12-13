import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

# python lottery_prediction.py
# Đọc và xử lý dữ liệu
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Xử lý cột Date
    def convert_date(date_str):
        # Loại bỏ dấu ngoặc kép
        date_str = date_str.strip('"')
        # Tách thành thứ và ngày tháng
        weekday, date = date_str.split(', ')
        # Chuyển đổi ngày tháng
        return pd.to_datetime(date, format='%d/%m/%Y')
    
    # Chuyển đổi cột Date
    df['Date'] = df['Date'].apply(convert_date)
    
    # Chuyển đổi cột Result thành list các số
    df['Numbers'] = df['Result'].apply(lambda x: [int(num) for num in x.split('�')[:6]])
    
    # Thêm các features mới
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    
    # Map thứ trong tuần
    weekday_map = {
        'T2': 0, 'T3': 1, 'T4': 2, 'T5': 3, 
        'T6': 4, 'T7': 5, 'CN': 6
    }
    
    return df

# Tạo thêm features
def create_features(numbers_array):
    features = []
    for numbers in numbers_array:
        # Tính toán các đặc trưng thống kê
        mean = np.mean(numbers)
        std = np.std(numbers)
        median = np.median(numbers)
        # Tính khoảng cách giữa các số
        diffs = np.diff(sorted(numbers))
        min_diff = np.min(diffs)
        max_diff = np.max(diffs)
        # Tạo feature vector
        feature = np.concatenate([numbers, [mean, std, median, min_diff, max_diff]])
        features.append(feature)
    return np.array(features)

# Chuẩn bị dữ liệu cho LSTM
def prepare_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, :6])  # Chỉ lấy 6 số đầu tiên làm target
    return np.array(X), np.array(y)

# Xây dựng mô hình LSTM nâng cao
def build_advanced_model(seq_length, feature_dim):
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True), input_shape=(seq_length, feature_dim)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(64)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        
        Dense(6, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', 
                 loss='mse',
                 metrics=['mae'])
    return model

# Phân tích xu hướng
def analyze_trends(df):
    frequent_numbers = {}
    for numbers in df['Numbers']:
        for num in numbers:
            frequent_numbers[num] = frequent_numbers.get(num, 0) + 1
    
    # Tìm các số xuất hiện nhiều nhất
    sorted_numbers = sorted(frequent_numbers.items(), key=lambda x: x[1], reverse=True)
    return sorted_numbers

# Hàm chính để train và dự đoán
def predict_lottery():
    # Load và xử lý dữ liệu
    df = load_data('Data1.csv')
    numbers = np.array(df['Numbers'].tolist())
    
    # Tạo features nâng cao
    features = create_features(numbers)
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    # Chuẩn bị sequences
    seq_length = 15  # Tăng độ dài sequence
    X, y = prepare_sequences(features_scaled, seq_length)
    
    # Chia dữ liệu
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Xây dựng và train mô hình
    model = build_advanced_model(seq_length, features.shape[1])
    
    # Thêm early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Train mô hình
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Phân tích xu hướng
    trends = analyze_trends(df)
    print("\nTop 10 số xuất hiện nhiều nhất:")
    for num, count in trends[:10]:
        print(f"Số {num}: {count} lần")
    
    # Dự đoán
    last_sequence = features_scaled[-seq_length:]
    prediction_scaled = model.predict(np.array([last_sequence]))
    
    # Chuyển đổi prediction về dạng số - Sửa phần này
    prediction_with_zeros = np.concatenate([prediction_scaled[0], np.zeros(features.shape[1]-6)])
    prediction_reshaped = prediction_with_zeros.reshape(1, -1)  # Reshape thành mảng 2D
    prediction = scaler.inverse_transform(prediction_reshaped)[0][:6]
    
    # Làm tròn và đảm bảo các số hợp lệ
    prediction = np.clip(np.round(prediction), 1, 55)
    prediction = np.unique(prediction)
    
    # Bổ sung số nếu thiếu
    while len(prediction) < 6:
        # Ưu tiên chọn từ top số xuất hiện nhiều
        for num, _ in trends:
            if num not in prediction:
                prediction = np.append(prediction, num)
                break
    
    prediction = np.sort(prediction[:6])
    
    # Vẽ đồ thị loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    return prediction.astype(int), history

# Chạy dự đoán
if __name__ == "__main__":
    print("Đang phân tích dữ liệu và dự đoán...")
    predicted_numbers, history = predict_lottery()
    
    print("\nDự đoán 6 số cho kỳ quay số tiếp theo:")
    print(predicted_numbers)
    
    print("\nĐộ chính xác của mô hình:")
    print(f"Loss cuối cùng: {history.history['loss'][-1]:.4f}")
    print(f"Validation Loss cuối cùng: {history.history['val_loss'][-1]:.4f}")
    
    print("\nLưu ý: Đây chỉ là dự đoán dựa trên phân tích dữ liệu lịch sử.")
    print("Kết quả xổ số là ngẫu nhiên và không thể dự đoán chính xác 100%.") 