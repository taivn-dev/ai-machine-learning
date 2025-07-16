import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data_path = "./data"

# Read Data
data = pd.read_csv(f"{data_path}/mall_customers.csv")

# Chọn các cột liên quan
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Áp dụng thuật toán KMeans với 5 cụm
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

data['Cluster'] = labels

# Xuất toàn bộ dữ liệu có nhãn cụm vào file CSV mới
data_sorted = data.sort_values(by='Cluster')
# data_sorted.to_csv(f"{data_path}/clustered_customers.csv", index=False)
for cluster_id in sorted(data['Cluster'].unique()):
    # Lọc dữ liệu của cụm hiện tại
    cluster_data = data[data['Cluster'] == cluster_id]
    
    # Tạo tên file dựa trên số cụm
    filename = f"{data_path}/mall_customers_cluster_{cluster_id}.csv"
    
    # Xuất ra file CSV
    cluster_data.to_csv(filename, index=False)
    
    print(f"Đã lưu cụm {cluster_id} vào file: {filename}")


# Vẽ kết quả phân cụm
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title('Phân cụm khách hàng trung tâm thương mại')
plt.xlabel('Thu nhập')
plt.ylabel('Điểm chi tiêu')
plt.show()


# # Xem thống kê tổng quát theo từng cụm
# grouped = data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']].mean()
# print(grouped)

# Hiển thị số lượng khách hàng theo từng cụm
#print(data['Cluster'].value_counts())

# for cluster_id in sorted(data['Cluster'].unique()):
#     print(f"\n--- Cụm {cluster_id} ---")
#     print(data[data['Cluster'] == cluster_id].head())



