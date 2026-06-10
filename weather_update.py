import pandas as pd
import numpy as np

# 1. Đọc dữ liệu từ file CSV
df = pd.read_csv('HaNoi,VietNam 2025-10-01 to 2026-06-02(1) - HaNoi,VietNam 2025-10-01 to 2026-06-02.csv')

# 2. Định nghĩa hàm tính xác suất Gaussian
def gaussian_probability(x, mean, std):
    if pd.isna(mean) or pd.isna(std):
        return 1.0
    std = max(std, 1e-9) 
    exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# 3. Tính toán các thông số thống kê cần thiết cho Naive Bayes
condition_prior = df['conditions'].value_counts(normalize=True)
numeric_df = df.select_dtypes(include=[np.number])
numeric_df['conditions'] = df['conditions']
condition_stats = numeric_df.groupby('conditions').agg(['mean', 'std'])
mapping_df = df.groupby('conditions')[['Weather', 'description']].agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else "Unknown")
condition_to_weather = mapping_df['Weather'].to_dict()
condition_to_desc = mapping_df['description'].to_dict()
print("-" * 60)
# 4. Nhập dữ liệu mới từ bàn phím (Đã thêm đầy đủ các cột bạn yêu cầu)
new_data = {
    'tempmax': float(input("Nhập nhiệt độ cao nhất (tempmax): ")),
    'tempmin': float(input("Nhập nhiệt độ thấp nhất (tempmin): ")),
    'temp': float(input("Nhập nhiệt độ trung bình (temp): ")),
    'tempwet': float(input("Nhập nhiệt độ bầu ướt (tempwet): ")),
    'humidity': float(input("Nhập độ ẩm (humidity): ")),
    'precip(Rainfall)': float(input("Nhập lượng mưa (precip(Rainfall)): ")),
    'windspeed': float(input("Nhập tốc độ gió (windspeed): ")),
    'cloudcover': float(input("Nhập độ bao phủ mây (cloudcover): ")),
    'visibility': float(input("Nhập tầm nhìn (visibility): "))
}
# 5. Tính toán xác suất tổng hợp cho từng loại conditions
results = {}
for cond in condition_prior.index:
    # 5.1 Khởi tạo xác suất tổng bằng xác suất tiên nghiệm của condition đó
    prob = condition_prior[cond]
    # 5.2 Duyệt qua từng biến bạn đã nhập để nhân dồn xác suất có điều kiện
    for feature, value in new_data.items():
        mean = condition_stats.loc[cond, (feature, 'mean')]
        std = condition_stats.loc[cond, (feature, 'std')]
        prob *= gaussian_probability(value, mean, std)  
    results[cond] = prob
# 6. Chọn loại conditions có xác suất lớn nhất
predicted_condition = max(results, key=results.get)
# 7. Từ conditions suy ra Weather và description
predicted_weather = condition_to_weather.get(predicted_condition, "Không có dữ liệu")
predicted_desc = condition_to_desc.get(predicted_condition, "Không có dữ liệu")
print("-" * 60)
print("KẾT QUẢ DỰ BÁO:")
print(f">> Tình trạng (Condition): {predicted_condition}")
print(f">> Loại thời tiết (Weather): {predicted_weather}")
print(f">> Mô tả chi tiết (Description): {predicted_desc}")