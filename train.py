import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load dữ liệu
train_df = pd.read_csv('train.csv')

# 2. Tiền xử lý nhanh
# Chọn các cột số quan trọng và bỏ các cột ID, Text quá phức tạp
features = ['Age', 'Tuition_Debt', 'Count_F', 'Training_Score_Mixed']
# Thêm các cột điểm chuyên cần (Att_Subject_01 đến 40)
att_cols = [col for col in train_df.columns if 'Att_Subject' in col]
features.extend(att_cols)

X = train_df[features].fillna(-1) # Điền giá trị thiếu bằng -1
y = train_df['Academic_Status']

# 3. Huấn luyện
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Serialize mô hình bằng Pickle
with open('train_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Đã lưu mô hình thành công vào file train_model.pkl")