# DS201 - Lab 4: Encoder–Decoder và attention cho dịch máy Anh–Việt

Bài làm nằm trong một notebook: `Lab4_24520253.ipynb` (đã chạy xong trên Kaggle, giữ nguyên output). Báo cáo: `BaoCao.pdf`.

| Bài | Nội dung | Vị trí trong notebook |
|---|---|---|
| 1 | Encoder–Decoder LSTM, 3 lớp mỗi bên, hidden 256, không attention | mục 6 |
| 2 | Như Bài 1, decoder có Bahdanau attention | mục 7, heatmap ở mục 11 |
| 3 | Như Bài 1, decoder có Luong attention (global, general score, input feeding) | mục 8 |

Đánh giá trên test, câu dịch mẫu, heatmap, bảng tổng hợp và kết luận nằm ở mục 9 đến 12.

## Dữ liệu

Dữ liệu không đưa lên repo. small-PhoMT: https://drive.google.com/drive/folders/186OAOuSEYEDVcry7WP5UBdqECXo26QAb (20,000 train, 2,000 dev, 2,000 test). Notebook đọc từ `/kaggle/input` nếu có dataset, nếu không thì tải về thư mục `data/` và kiểm tra SHA-256.

Chỉ giữ cặp câu có cả hai phía dài từ 1 đến 50 token, còn 18,607 / 1,901 / 1,786 cặp.

## Chạy

```bash
pip install torch numpy pandas matplotlib rouge-score sacrebleu jupyter
jupyter nbconvert --to notebook --execute Lab4_24520253.ipynb
```

Hoặc import notebook lên Kaggle (GPU T4, bật Internet) rồi chọn Save & Run All. Cấu hình ở ô `cfg`: embedding 256, hidden 256, 3 lớp, dropout 0.3, batch 32, Adam lr 1e-3, `clip_grad_norm_` 1.0, teacher forcing 0.5, seed 42, tối đa 10 epoch. Trên Kaggle T4 cả notebook chạy khoảng 50 phút; trên CPU chậm hơn nhiều (ước tính 5 đến 6 giờ).

## Kết quả

Chọn epoch theo ROUGE-L dev, tập test chỉ chấm một lần ở cuối. ROUGE-L là F1 trung bình theo câu, tokenizer giữ dấu tiếng Việt. BLEU chỉ để tham khảo.

| Mô hình | ROUGE-L dev tốt nhất | ROUGE-L test | BLEU test | Tham số | Giây/epoch |
|---|---|---|---|---|---|
| LSTM không attention | 0.2298 (epoch 10) | 0.2346 | 2.61 | 7,500,963 | 72.8 |
| LSTM + Bahdanau | 0.2975 (epoch 10) | 0.3138 | 5.83 | 8,025,763 | 108.5 |
| LSTM + Luong | 0.2905 (epoch 10) | 0.3054 | 5.52 | 7,959,971 | 90.7 |

Các hình và bảng số nằm trong `artifacts/`. Checkpoint `.pt` và `metadata.json` không đưa lên repo.

Hai ô phân tích thêm ở mục 11 và 12 (căn chỉnh attention, điểm theo độ dài câu) được chạy lại trên CPU từ checkpoint đã lưu sau khi có kết quả Kaggle. ROUGE-L test tính lại khớp với bản chạy trên GPU.
