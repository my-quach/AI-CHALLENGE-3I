# AI-CHALLENGE-3I
# MEAL at UEH Canteen 🍱 –  ONLY WITH F5! 💻
Link Train CNN + Menu.Json: Link Train CNN + Menu.Json:(https://drive.google.com/drive/folders/1EA9ftrHVHxNf77KtOb5ZV8t8-IeeteV_?usp=sharing)

Hệ thống Nhận Diện Món Ăn Và Tính Tiền Tự Động Tổng quan dự án Đây là một hệ thống ứng dụng AI vào nhận diện và tính tiền món ăn Việt Nam. Hệ thống sử dụng YOLOv10n để phát hiện món ăn trong ảnh và CNN để phân loại tên món. Sau đó đối chiếu với file menu.json để tính giá và hiển thị hóa đơn trực tiếp kèm hình ảnh từng món. Mục đích: Giúp hệ thống tính tiền trong canteen, nhà ăn trở nên linh hoạt, nhanh chóng, chính xác, tối ưu hóa thời gian cho cả người mua và người bán. Chức năng chính: Nhận diện các món ăn trong một bức ảnh.

Cắt ảnh từng món và phân loại tên món bằng mô hình CNN.

Tính tổng tiền hóa đơn dựa trên menu có sẵn.

Hiển thị hóa đơn từng món kèm ảnh và giá tương ứng. Chi tiết các bước thiết lập môi trường Tạo mới một notebook trên Google Colab.

Clone dự án về Colab: !git clone https://github.com/yourusername/vietnamese-food-detection.git
%cd vietnamese-food-detection

Cài đặt các thư viện cần thiết: Tải mô hình YOLOv10n Tải mô hình CNN (.keras), sau đó Upload lên Google Drive và liên kết Colab với Drive. Tải file ảnh mẫu cần nhận diện vào thư mục dự án hoặc Drive. Chạy mô hình YOLOv10n để cắt ảnh vaf script CNN để predict và cuối cùng tính hóa đơn.

Hướng dẫn sử dụng & Ví dụ chạy dự án Nhận diện và tính tiền món ăn từ một bức ảnh Chạy nhận diện bằng YOLOv10n: from ultralytics import YOLO

Load model YOLOv10n
model = YOLO('yolov10n.pt')

Nhận diện và crop món ăn từ ảnh
results = model( source='/content/drive/MyDrive/14.jpg', # Đường dẫn ảnh cần nhận diện save=True, save_crop=True, imgsz=640, conf=0.15) 2. Dự đoán tên món và tính tiền với CNN: from tensorflow.keras.models import load_model from tensorflow.keras.preprocessing import image import numpy as np import os, glob, json, unicodedata from IPython.display import Image as IPyImage, display, HTML

Load model CNN đã train
cnn_model = load_model('/content/drive/MyDrive/pj ai.keras')

Load menu giá tiền
with open('/content/drive/MyDrive/pj ai/menu.json', 'r', encoding='utf-8') as f: menu_raw = json.load(f)

Chuẩn hóa text tiếng Việt
def normalize_text(text): return unicodedata.normalize('NFC', text)

menu = {normalize_text(k): v for k, v in menu_raw.items()}

Danh sách tên lớp
classes = ['canh cải', 'canh chua', 'cá hú kho', 'cơm', 'gà chiên', 'rau muống xào', 'thịt kho', 'thịt kho trứng', 'trứng chiên', 'đậu hủ sốt cà chua']

Duyệt qua từng ảnh crop và dự đoán
cropped = sorted(glob.glob('/content/runs/detect/predict*/crops'), key=os.path.getmtime, reverse=True) latest_cropped = cropped[0]

total = 0for dish_folder in os.listdir(latest_cropped): folder_path = os.path.join(latest_cropped, dish_folder) if not os.path.isdir(folder_path): continue

for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)

    # Load ảnh và dự đoán
    img = image.load_img(img_path, target_size=(200, 200))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = cnn_model.predict(img_array)
    class_idx = np.argmax(prediction)
    dish_name = classes[class_idx]

    dish_name_norm = normalize_text(dish_name)
    price = menu.get(dish_name_norm, 0)
    total += price

    display(IPyImage(img_path, width=250))
    display(HTML(f"<p><b>{dish_name}</b>: {price:,} VND</p>"))
print(f"\n Tổng cộng: {total:,} VND") print("===================") Cài đặt các thư viện cần thiết: from ultralytics import YOLO from tensorflow.keras.models import load_model from tensorflow.keras.preprocessing import image from IPython.display import Image as IPyImage, display, HTML import numpy as np import os, glob, json, unicodedata

Chất lượng chương trình Dự án được xây dựng và tổ chức theo các nguyên tắc lập trình chuẩn và thực hành tốt nhất để đảm bảo: Code rõ ràng, dễ đọc:

Tất cả các khối lệnh được chia theo từng phần rõ ràng: Data Augmentation, Khởi tạo Model, Callback, Training, Dự đoán. Các hàm, biến, và thư mục được đặt tên có ý nghĩa và đồng nhất. Thêm chú thích (comment) giải thích các đoạn mã quan trọng để người đọc dễ hiểu và dễ bảo trì.

Cấu trúc file hợp lý: Tách riêng các tệp dữ liệu (ảnh, model, menu.json) và code. Sử dụng Google Colab để dễ dàng thao tác, kiểm tra và demo. Các file mô hình và menu được lưu vào Google Drive hoặc repository để dễ chia sẻ và tái sử dụng.
