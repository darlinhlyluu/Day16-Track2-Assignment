# Báo cáo ngắn

## Họ và tên: Lưu Linh Ly
## MSHV: 2A202600119

Lab này được triển khai trên GCP theo phương án CPU fallback với máy `n2-standard-8`.
Mô hình LightGBM đã được huấn luyện thành công trên dataset `creditcard.csv` gồm `284,807` bản ghi với thời gian train `1.095 giây`, cho thấy workload ML này vẫn chạy rất nhanh trên CPU.
Về chất lượng mô hình, kết quả đạt `AUC-ROC = 0.9778`, `accuracy = 0.9993`, và `F1-score = 0.7980`, thể hiện khả năng phân loại gian lận ở mức tốt.
Về tốc độ suy luận, độ trễ dự đoán cho `1 row` là `0.000798 giây` và throughput với `batch 1000 rows` đạt khoảng `960,820.59 rows/s`, phù hợp để benchmark inference trên CPU.
So với phương án GPU dùng cho LLM/vLLM, phương án CPU không tối ưu cho mô hình ngôn ngữ lớn nhưng vẫn đáp ứng đầy đủ quy trình Terraform IaC, triển khai VM, huấn luyện, inference và theo dõi chi phí.
Tài khoản trial không khả dụng kịp thời để triển khai trong thời gian làm lab nên tôi chọn phương án dùng CPU thay GPU là quota `NVIDIA T4` trên GCP.
