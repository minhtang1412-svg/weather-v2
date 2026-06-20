import pandas as pd
import numpy as np

# 1. Đọc dữ liệu lịch sử để huấn luyện mô hình
try:
    df_train = pd.read_csv('HaNoi,VietNam 2025-10-01 to 2026-06-02(1) - HaNoi,VietNam 2025-10-01 to 2026-06-02.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file dữ liệu huấn luyện (CSV gốc).")
    exit()

# 2. Đọc file CSV chứa dữ liệu đầu vào cần dự báo
try:
    df_input = pd.read_csv('du_lieu_dau_vao_du_bao.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'du_lieu_dau_vao_du_bao.csv'. Hãy đảm bảo file này tồn tại cùng thư mục.")
    exit()

# 3. Hàm tính log xác suất
def log_gaussian_probability(x, mean, std):
    if pd.isna(mean) or pd.isna(std):
        return 0.0
    std = max(std, 1e-9) 
    return -0.5 * np.log(2 * np.pi * std ** 2) - ((x - mean) ** 2 / (2 * std ** 2))

# 4. Tính toán các thông số thống kê từ tập huấn luyện
condition_prior = df_train['conditions'].value_counts(normalize=True)
numeric_df = df_train.select_dtypes(include=[np.number]).copy()
numeric_df['conditions'] = df_train['conditions']
condition_stats = numeric_df.groupby('conditions').agg(['mean', 'std'])

mapping_df = df_train.groupby('conditions')[['Weather', 'description']].agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else "Unknown")
condition_to_weather = mapping_df['Weather'].to_dict()
condition_to_desc = mapping_df['description'].to_dict()

# 5. TỰ ĐỘNG DỰ BÁO VÀ IN KẾT QUẢ RA TERMINAL ĐẸP MẮT
print("\n" + "="*70)
print(" BẢNG KẾT QUẢ DỰ BÁO THỜI TIẾT TỰ ĐỘNG ".center(70))
print("="*70)

for index, row in df_input.iterrows():
    results = {}
    for cond in condition_prior.index:
        log_prob = np.log(condition_prior[cond])
        for feature in df_input.columns:
            # Chỉ tính nếu cột đó có trong tập huấn luyện
            if feature in condition_stats.columns.levels[0]:
                mean = condition_stats.loc[cond, (feature, 'mean')]
                std = condition_stats.loc[cond, (feature, 'std')]
                log_prob += log_gaussian_probability(row[feature], mean, std)  
        results[cond] = log_prob
    
    # Lấy nhãn có xác suất cao nhất
    predicted_condition = max(results, key=results.get)
    predicted_weather = condition_to_weather.get(predicted_condition, "Không có dữ liệu")
    predicted_desc = condition_to_desc.get(predicted_condition, "Không có dữ liệu")
    
    # IN RA MAN HÌNH
    print(f"  Dự báo       : {predicted_condition} ({predicted_weather})")
    print(f"  Mô tả chi tiết: {predicted_desc}")
    print("=" * 100)
print("Đã hoàn tất dự báo!\n")
