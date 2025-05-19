import pandas as pd
import numpy as np
from mf_knn_recommender import MF_KNN_Recommender

# Khởi tạo recommender system
recommender = MF_KNN_Recommender(
    latent_factors=100,  # Tối ưu hóa số lượng latent factors
    steps=250,          # Số bước đào tạo
    k=5,                # Số lượng khuyến nghị mặc định
    alpha=0.001,        # Tỷ lệ học tập
    beta=0.01           # Tham số điều chỉnh
)

# Tải mô hình hoặc đào tạo mới
if not recommender.load_model():
    print("Đang đào tạo mô hình mới...")
    recommender.train('segment_dataset.csv')
    print("Đào tạo hoàn tất!")

# Danh sách user IDs cần khuyến nghị
target_users = [
    'e1feae9083c4c2895ddf6dc80526a85d', 
    'afddf43a03a9941624ed42c0b2c17280', 
    '64ee476500a01beb94df40f97a108c50'
]

print("\n===== KHUYẾN NGHỊ SẢN PHẨM CHO CÁC USER CỤ THỂ =====")
for i, user_id in enumerate(target_users):
    print(f"\n{i+1}. User {user_id}:")
    print("   Khuyến nghị thông thường:")
    recs = recommender.recommend_items_for_user(user_id, top_k=5, show_scores=True)
    
    print("\n   Khuyến nghị đa dạng hóa (lambda=0.7):")
    diverse_recs = recommender.diversify_recommendations(user_id, top_k=5, lambda_diversity=0.7)
    for item in diverse_recs:
        print(f"   - {item}")

print("\n===== SO SÁNH LOẠI KHUYẾN NGHỊ =====")
comparison = {}

for user_id in target_users:
    regular_recs = recommender.recommend_items_for_user(user_id, top_k=5, show_scores=False)
    diverse_recs = recommender.diversify_recommendations(user_id, top_k=5, lambda_diversity=0.7)
    
    # Tính số lượng khuyến nghị khác nhau
    differences = set(diverse_recs) - set(regular_recs)
    
    comparison[user_id] = {
        'regular': regular_recs,
        'diverse': diverse_recs,
        'num_different': len(differences),
        'different_items': list(differences)
    }

# In kết quả so sánh
print("\nSo sánh khuyến nghị thông thường vs. đa dạng hóa:")
for user_id, data in comparison.items():
    print(f"\nUser: {user_id}")
    print(f"  Thông thường: {', '.join(data['regular'])}")
    print(f"  Đa dạng hóa:  {', '.join(data['diverse'])}")
    print(f"  Số sản phẩm khác nhau: {data['num_different']}")
    if data['num_different'] > 0:
        print(f"  Sản phẩm mới: {', '.join(data['different_items'])}")

# Phân tích sâu hơn cho user đầu tiên trong danh sách
focus_user = target_users[0]
print(f"\n===== PHÂN TÍCH CHI TIẾT CHO USER {focus_user} =====")

# Tạo trực quan hóa embedding
print("Đang tạo biểu đồ embedding sản phẩm...")
recommender.plot_item_embeddings(top_n=30, save_path=f"product_embeddings_{focus_user}.png")
print(f"Biểu đồ đã được lưu tại: product_embeddings_{focus_user}.png")

# Thực hiện khuyến nghị với các mức độ đa dạng hóa khác nhau
print("\nTác động của tham số đa dạng hóa:")
lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]

for lambda_val in lambda_values:
    recs = recommender.diversify_recommendations(focus_user, top_k=5, lambda_diversity=lambda_val)
    print(f"  Lambda = {lambda_val:.1f}: {', '.join(recs)}")