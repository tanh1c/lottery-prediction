import pandas as pd
from datetime import datetime
import os

# python lottery_prediction_history.py

def save_prediction():
    # Tạo file nếu chưa tồn tại
    if not os.path.exists('prediction_history.csv'):
        df = pd.DataFrame(columns=['Ngày dự đoán', 'Loại dự đoán', 'Số dự đoán', 'Kết quả thực tế', 'Số trùng'])
        df.to_csv('prediction_history.csv', index=False)
    
    # Nhập thông tin
    print("\n=== NHẬP THÔNG TIN DỰ ĐOÁN ===")
    
    # Ngày dự đoán
    while True:
        try:
            date_str = input("Nhập ngày (dd/mm/yyyy): ")
            date = datetime.strptime(date_str, '%d/%m/%Y')
            break
        except ValueError:
            print("Định dạng ngày không hợp lệ. Vui lòng nhập theo định dạng dd/mm/yyyy")
    
    # Loại dự đoán
    print("\nChọn loại dự đoán:")
    print("1. Match 3 (3 số trùng)")
    print("2. Match 4 (4 số trùng)")
    print("3. Match 5 (5 số trùng)")
    print("4. Jackpot 2 (5 số + số phụ)")
    print("5. Jackpot 1 (6 số trùng)")
    
    pred_types = {
        '1': 'Match 3',
        '2': 'Match 4',
        '3': 'Match 5',
        '4': 'Jackpot 2',
        '5': 'Jackpot 1'
    }
    
    while True:
        pred_type = input("Chọn loại (1-5): ")
        if pred_type in pred_types:
            pred_type = pred_types[pred_type]
            break
        print("Lựa chọn không hợp lệ!")
    
    # Nhập số dự đoán
    while True:
        try:
            numbers_str = input("\nNhập 6 số dự đoán (cách nhau bằng dấu cách): ")
            numbers = [int(x) for x in numbers_str.split()]
            if len(numbers) != 6 or not all(1 <= x <= 55 for x in numbers):
                raise ValueError
            numbers.sort()
            break
        except ValueError:
            print("Vui lòng nhập đúng 6 số từ 1-55!")
    
    # Nhập kết quả thực tế (nếu có)
    while True:
        has_result = input("\nBạn đã có kết quả thực tế chưa? (y/n): ").lower()
        if has_result in ['y', 'n']:
            break
        print("Vui lòng nhập 'y' hoặc 'n'!")
    
    actual_numbers = []
    matching_count = 0
    
    if has_result == 'y':
        while True:
            try:
                result_str = input("Nhập kết quả thực tế (6 số, cách nhau bằng dấu cách): ")
                actual_numbers = [int(x) for x in result_str.split()]
                if len(actual_numbers) != 6 or not all(1 <= x <= 55 for x in actual_numbers):
                    raise ValueError
                actual_numbers.sort()
                matching_count = len(set(numbers) & set(actual_numbers))
                break
            except ValueError:
                print("Vui lòng nhập đúng 6 số từ 1-55!")
    
    # Đọc file hiện tại
    df = pd.read_csv('prediction_history.csv')
    
    # Thêm dự đoán mới
    new_row = {
        'Ngày dự đoán': date.strftime('%d/%m/%Y'),
        'Loại dự đoán': pred_type,
        'Số dự đoán': ' '.join(map(str, numbers)),
        'Kết quả thực tế': ' '.join(map(str, actual_numbers)) if actual_numbers else 'Chưa có',
        'Số trùng': matching_count if actual_numbers else 'Chưa có'
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Lưu lại file
    df.to_csv('prediction_history.csv', index=False)
    
    print("\n=== THỐNG KÊ ===")
    print(f"Tổng số dự đoán đã lưu: {len(df)}")
    if len(df[df['Số trùng'] != 'Chưa có']) > 0:
        print("\nThống kê số trùng:")
        for i in range(7):
            count = len(df[df['Số trùng'] == i])
            if count > 0:
                print(f"Trùng {i} số: {count} lần")
    
    print("\nĐã lưu dự đoán thành công!")

def view_history():
    if not os.path.exists('prediction_history.csv'):
        print("Chưa có lịch sử dự đoán!")
        return
    
    df = pd.read_csv('prediction_history.csv')
    if len(df) == 0:
        print("Chưa có dự đoán nào được lưu!")
        return
    
    print("\n=== LỊCH SỬ DỰ ĐOÁN ===")
    print(df.to_string(index=False))

def update_result():
    if not os.path.exists('prediction_history.csv'):
        print("Chưa có lịch sử dự đoán!")
        return
    
    df = pd.read_csv('prediction_history.csv')
    pending_predictions = df[df['Kết quả thực tế'] == 'Chưa có']
    
    if len(pending_predictions) == 0:
        print("Không có dự đoán nào cần cập nhật kết quả!")
        return
    
    print("\n=== CẬP NHẬT KẾT QUẢ ===")
    print("Các dự đoán chưa có kết quả:")
    print(pending_predictions.to_string(index=True))
    
    while True:
        try:
            idx = int(input("\nChọn số thứ tự cần cập nhật: "))
            if idx not in pending_predictions.index:
                raise ValueError
            break
        except ValueError:
            print("Số th�� tự không hợp lệ!")
    
    while True:
        try:
            result_str = input("Nhập kết quả thực tế (6 số, cách nhau bằng dấu cách): ")
            actual_numbers = [int(x) for x in result_str.split()]
            if len(actual_numbers) != 6 or not all(1 <= x <= 55 for x in actual_numbers):
                raise ValueError
            actual_numbers.sort()
            break
        except ValueError:
            print("Vui lòng nhập đúng 6 số từ 1-55!")
    
    # Tính số trùng
    predicted_numbers = [int(x) for x in df.loc[idx, 'Số dự đoán'].split()]
    matching_count = len(set(predicted_numbers) & set(actual_numbers))
    
    # Cập nhật kết quả
    df.loc[idx, 'Kết quả thực tế'] = ' '.join(map(str, actual_numbers))
    df.loc[idx, 'Số trùng'] = matching_count
    
    # Lưu lại file
    df.to_csv('prediction_history.csv', index=False)
    print("\nĐã cập nhật kết quả thành công!")

def main():
    while True:
        print("\n=== MENU ===")
        print("1. Thêm dự đoán mới")
        print("2. Xem lịch sử dự đoán")
        print("3. Cập nhật kết quả")
        print("4. Thoát")
        
        choice = input("\nChọn chức năng (1-4): ")
        
        if choice == '1':
            save_prediction()
        elif choice == '2':
            view_history()
        elif choice == '3':
            update_result()
        elif choice == '4':
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main() 