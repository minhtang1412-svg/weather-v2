import pandas as pd
import numpy as np

# Đảm bảo kết quả chia tập dữ liệu không bị thay đổi ngẫu nhiên mỗi lần chạy
np.random.seed(42)

# 1. Đọc dữ liệu lịch sử để huấn luyện mô hình
try:
    df_train = pd.read_csv('HaNoi,VietNam 2025-10-01 to 2026-06-02(1) - HaNoi,VietNam 2025-10-01 to 2026-06-02.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file dữ liệu huấn luyện (CSV gốc).")
    exit()

# 2. Đọc file CSV chứa dữ liệu đầu vào cần dự báo
try:
    df_input = pd.read_csv('dulieudauvao.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'dulieudauvao.csv'. Hãy đảm bảo file này tồn tại cùng thư mục.")
    exit()

# 3. Hàm tính log xác suất mật độ Gauss
def log_gaussian_probability(x, mean, std):
    if pd.isna(mean) or pd.isna(std):
        return 0.0
    std = max(std, 1e-9)  # Bộ lọc biên tránh lỗi chia cho số 0
    return -0.5 * np.log(2 * np.pi * std ** 2) - ((x - mean) ** 2 / (2 * std ** 2))

# =========================================================================
# CHIA DỮ LIỆU LỊCH SỬ THÀNH TRAIN (80%) VÀ TEST (20%)
# =========================================================================
# Trộn ngẫu nhiên các dòng dữ liệu để đảm bảo tính khách quan khí hậu
df_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

test_size = int(len(df_shuffled) * 0.2)
test_set = df_shuffled.iloc[:test_size].copy()
train_set = df_shuffled.iloc[test_size:].copy()

# 4. Tính toán các thông số thống kê dựa ĐỘC LẬP trên tập huấn luyện (train_set)
condition_prior = train_set['conditions'].value_counts(normalize=True)
numeric_df = train_set.select_dtypes(include=[np.number]).copy()
numeric_df['conditions'] = train_set['conditions']
condition_stats = numeric_df.groupby('conditions').agg(['mean', 'std'])

mapping_df = train_set.groupby('conditions')[['Weather', 'description']].agg(
    lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else "Unknown"
)
condition_to_weather = mapping_df['Weather'].to_dict()
condition_to_desc = mapping_df['description'].to_dict()

# =========================================================================
# TIẾN HÀNH KIỂM THỬ ĐỊNH LƯỢNG TRÊN TẬP KIỂM TRA (TEST SET)
# =========================================================================
y_true = test_set['conditions'].tolist()  # Nhãn thực tế gốc
y_pred = []                               # Mảng lưu kết quả dự báo của mô hình

for index, row in test_set.iterrows():
    results = {}
    for cond in condition_prior.index:
        log_prob = np.log(condition_prior[cond])
        for feature in test_set.columns:
            if feature in condition_stats.columns.levels[0]:
                mean = condition_stats.loc[cond, (feature, 'mean')]
                std = condition_stats.loc[cond, (feature, 'std')]
                log_prob += log_gaussian_probability(row[feature], mean, std)  
        results[cond] = log_prob
    
    predicted_condition = max(results, key=results.get)
    y_pred.append(predicted_condition)

# Tính độ chính xác tổng thể (Accuracy)
correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
accuracy = (correct_predictions / len(y_true)) * 100

# Khởi tạo Ma trận nhầm lẫn dạng bảng chéo 
labels = list(condition_prior.index)
confusion_matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}

# Đếm tần suất giao thoa kết quả phân loại
for true, pred in zip(y_true, y_pred):
    confusion_matrix[true][pred] += 1

# IN THỐNG KÊ ĐỊNH LƯỢNG RA TERMINAL
print("\n" + "="*70)
print(" ĐÁNH GIÁ ĐỊNH LƯỢNG HIỆU NĂNG MÔ HÌNH ".center(70))
print("="*70)
print(f"[*] Tổng số mẫu đưa vào kiểm thử độc lập: {len(y_true)} dòng")
print(f"[*] Số lượng mẫu dự báo chính xác hoàn toàn: {correct_predictions} dòng")
print(f"[*] Độ chính xác tổng thể hệ thống (Accuracy): {accuracy:.2f}%")
print("-" * 70)
print(" MA TRẬN NHẦM LẪN THỰC NGHIỆM (CONFUSION MATRIX):")
print("-" * 70)

# Định dạng in tiêu đề cột ma trận nhầm lẫn
header_row = f"{'Thực tế \\ Dự đoán':<25}" + "".join([f"{lbl:<20}" for lbl in labels])
print(header_row)

for actual in labels:
    row_str = f"{actual:<25}"
    for predicted in labels:
        row_str += f"{confusion_matrix[actual][predicted]:<20}"
    print(row_str)
print("="*70 + "\n")

print("-" * 80)
print(" BẢNG THỐNG KÊ HIỆU NĂNG CHI TIẾT THEO TỪNG LỚP THỜI TIẾT  ".center(80))
print("-" * 80)

# In tiêu đề cột thẳng hàng
header_b34 = f"{'Nhãn thời tiết mục tiêu':<25}{'Thực tế':<10}{'Dự đoán':<10}{'Precision (%)':<15}{'Recall (%)':<15}{'F1-score (%)':<15}"
print(header_b34)
print("-" * 80)

for label in labels:
    # 1. True Positive (TP): Thực tế là nhãn này và mô hình đoán đúng là nhãn này
    tp = confusion_matrix[label][label]
    
    # 2. Tổng số mẫu Thực tế (Actual): Tổng tất cả các ô trên dòng của nhãn đó (TP + FN)
    total_actual = sum(confusion_matrix[label][predicted] for predicted in labels)
    
    # 3. Tổng số mẫu Dự đoán (Predicted): Tổng tất cả các ô trên cột của nhãn đó (TP + FP)
    total_predicted = sum(confusion_matrix[actual][label] for actual in labels)
    
    # 4. Tính toán Precision và Recall kèm điều kiện bọc biên tránh lỗi chia cho số 0
    precision_val = (tp / total_predicted) * 100 if total_predicted > 0 else 0.0
    recall_val = (tp / total_actual) * 100 if total_actual > 0 else 0.0
    
    # 5. Tính toán F1-score (Trung bình điều hòa giữa Precision và Recall)
    if (precision_val + recall_val) > 0:
        f1_val = (2 * precision_val * recall_val) / (precision_val + recall_val)
    else:
        f1_val = 0.0
        
    # In kết quả định dạng căn lề trái/phải thẳng hàng tuyệt đối với tiêu đề
    print(f"{label:<25}{total_actual:<10}{total_predicted:<10}{precision_val:<15.2f}{recall_val:<15.2f}{f1_val:<15.2f}")

print("=" * 80 + "\n")

# 5. DỰ BÁO VÀ IN KẾT QUẢ CHO TẬP DỮ LIỆU MỚI (df_input)
print("="*70)
print(" BẢNG KẾT QUẢ DỰ BÁO THỜI TIẾT TỰ ĐỘNG CHO DỮ LIỆU MỚI ".center(70))
print("="*70)

for index, row in df_input.iterrows():
    results = {}
    for cond in condition_prior.index:
        log_prob = np.log(condition_prior[cond])
        for feature in df_input.columns:
            if feature in condition_stats.columns.levels[0]:
                mean = condition_stats.loc[cond, (feature, 'mean')]
                std = condition_stats.loc[cond, (feature, 'std')]
                log_prob += log_gaussian_probability(row[feature], mean, std)  
        results[cond] = log_prob
    
    predicted_condition = max(results, key=results.get)
    predicted_weather = condition_to_weather.get(predicted_condition, "Không có dữ liệu")
    predicted_desc = condition_to_desc.get(predicted_condition, "Không có dữ liệu")
    
    print(f"  Dự báo       : {predicted_condition} ({predicted_weather})")
    print(f"  Mô tả chi tiết: {predicted_desc}")
    print("-" * 70)
print("Đã hoàn tất dự báo!\n")