import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

# python lottery_prediction_advanced.py

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    def convert_date(date_str):
        date_str = date_str.strip('"')
        weekday, date = date_str.split(', ')
        return pd.to_datetime(date, format='%d/%m/%Y')
    
    df['Date'] = df['Date'].apply(convert_date)
    df['Numbers'] = df['Result'].apply(lambda x: [int(num) for num in x.split('�')[:6]])
    df['Special'] = df['Result'].apply(lambda x: int(x.split('�')[6]))  # Số đặc biệt
    
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    return df

def create_features_by_type(numbers_array, prediction_type):
    features = []
    for numbers in numbers_array:
        base_features = list(numbers)
        
        if prediction_type in ['match3', 'match4']:
            # Thêm features cho dự đoán match 3,4
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
            # Thêm features cho dự đoán match 5 và jackpot 2
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
            # Features đặc biệt cho jackpot 1
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

def prepare_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, :6])  # Chỉ lấy 6 số đầu tiên làm target
    return np.array(X), np.array(y)

def analyze_trends(df):
    frequent_numbers = {}
    for numbers in df['Numbers']:
        for num in numbers:
            frequent_numbers[num] = frequent_numbers.get(num, 0) + 1
    
    # Tìm các số xuất hiện nhiều nhất
    sorted_numbers = sorted(frequent_numbers.items(), key=lambda x: x[1], reverse=True)
    return sorted_numbers

def build_model_by_type(seq_length, feature_dim, prediction_type):
    if prediction_type in ['match3', 'match4']:
        # Mô hình đơn giản hơn cho match 3,4
        model = Sequential([
            LSTM(128, input_shape=(seq_length, feature_dim), return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(6, activation='sigmoid')
        ])
    
    elif prediction_type in ['match5', 'jackpot2']:
        # Mô hình phức tạp hơn cho match 5 và jackpot 2
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
        # Mô hình phức tạp nhất cho jackpot 1
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

def predict_lottery(prediction_type='match3'):
    print(f"\nĐang tối ưu dự đoán cho {prediction_type}...")
    
    df = load_data('Data1.csv')
    numbers = np.array(df['Numbers'].tolist())
    
    # Tạo features theo loại dự đoán
    features = create_features_by_type(numbers, prediction_type)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    # Điều chỉnh sequence length theo loại dự đoán
    seq_length = 10 if prediction_type in ['match3', 'match4'] else 15
    
    X, y = prepare_sequences(features_scaled, seq_length)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = build_model_by_type(seq_length, features.shape[1], prediction_type)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Điều chỉnh epochs và batch_size theo loại dự đoán
    epochs = 100 if prediction_type in ['match3', 'match4'] else 200
    batch_size = 32 if prediction_type in ['match3', 'match4'] else 16
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Dự đoán và xử lý kết quả
    last_sequence = features_scaled[-seq_length:]
    prediction_scaled = model.predict(np.array([last_sequence]))
    
    prediction_with_zeros = np.concatenate([prediction_scaled[0], np.zeros(features.shape[1]-6)])
    prediction_reshaped = prediction_with_zeros.reshape(1, -1)
    prediction = scaler.inverse_transform(prediction_reshaped)[0][:6]
    
    prediction = np.clip(np.round(prediction), 1, 55)
    prediction = np.unique(prediction)
    
    # Bổ sung số còn thiếu từ phân tích xu hướng
    trends = analyze_trends(df)
    while len(prediction) < 6:
        for num, _ in trends:
            if num not in prediction:
                prediction = np.append(prediction, num)
                break
    
    prediction = np.sort(prediction[:6])
    
    # Vẽ đồ thị loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss During Training ({prediction_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'training_loss_{prediction_type}.png')
    plt.close()
    
    return prediction.astype(int), history

if __name__ == "__main__":
    print("Chọn loại dự đoán:")
    print("1. Match 3 (3 số trùng)")
    print("2. Match 4 (4 số trùng)")
    print("3. Match 5 (5 số trùng)")
    print("4. Jackpot 2 (5 số + số phụ)")
    print("5. Jackpot 1 (6 số trùng)")
    
    choice = input("Nhập lựa chọn của bạn (1-5): ")
    
    prediction_types = {
        '1': 'match3',
        '2': 'match4',
        '3': 'match5',
        '4': 'jackpot2',
        '5': 'jackpot1'
    }
    
    if choice in prediction_types:
        pred_type = prediction_types[choice]
        predicted_numbers, history = predict_lottery(pred_type)
        
        print(f"\nDự đoán 6 số cho {pred_type}:")
        print(predicted_numbers)
        
        print("\nĐộ chính xác của mô hình:")
        print(f"Loss cuối cùng: {history.history['loss'][-1]:.4f}")
        print(f"Validation Loss cuối cùng: {history.history['val_loss'][-1]:.4f}")
        
        if pred_type == 'match3':
            print("\nXác suất trúng Match 3: 1/81 (khoảng 1.23%)")
        elif pred_type == 'match4':
            print("\nXác suất trúng Match 4: 1/2,208 (khoảng 0.045%)")
        elif pred_type == 'match5':
            print("\nXác suất trúng Match 5: 1/89,104 (khoảng 0.001%)")
        elif pred_type == 'jackpot2':
            print("\nXác suất trúng Jackpot 2: 1/4,831,613 (khoảng 0.000021%)")
        else:
            print("\nXác suất trúng Jackpot 1: 1/28,989,675 (khoảng 0.0000034%)")
        
        print("\nLưu ý: Đây chỉ là dự đoán dựa trên phân tích dữ liệu lịch sử.")
        print("Kết quả xổ số là ngẫu nhiên và không thể dự đoán chính xác 100%.")
    else:
        print("Lựa chọn không hợp lệ!") 