import os
import cv2
import pickle
import face_recognition
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Đường dẫn đến thư mục chứa hình ảnh khuôn mặt
dataset_dir = '/home/admin/Desktop/FACE_V3/dataset'

# Khởi tạo từ điển để lưu trữ encoding và ID
all_face_encodings = {}

def process_image(user_id, image_path):
    try:
        # Đọc hình ảnh và trích xuất encoding khuôn mặt
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        # Nếu tìm thấy encoding, thêm vào từ điển
        if face_encodings:
            return user_id, face_encodings[0]  # Trả về user_id và encoding
    except Exception as e:
        print(f"Không thể xử lý {image_path}: {e}")
    return None  # Trả về None nếu có lỗi

# Hàm chính để xử lý tất cả các ảnh
def process_all_images():
    # Duyệt qua tất cả các thư mục trong dataset_dir
    with ThreadPoolExecutor() as executor:  # Sử dụng đa luồng để xử lý song song
        future_results = []

        for user_id in os.listdir(dataset_dir):
            user_folder = os.path.join(dataset_dir, user_id)

            if os.path.isdir(user_folder):  # Kiểm tra xem có phải là thư mục không
                for image_name in os.listdir(user_folder):
                    image_path = os.path.join(user_folder, image_name)

                    # Chỉ xử lý các tệp hình ảnh
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Nộp công việc xử lý ảnh cho ThreadPoolExecutor
                        future_results.append(executor.submit(process_image, user_id, image_path))

        # Lấy kết quả từ các luồng xử lý
        for future in future_results:
            result = future.result()
            if result:
                user_id, face_encoding = result
                all_face_encodings[user_id] = face_encoding

# Chạy xử lý và lưu kết quả
process_all_images()

# Lưu dataset_faces.dat chứa tất cả các mã hóa khuôn mặt
with open('dataset_faces.dat', 'wb') as dataset_file:
    pickle.dump(all_face_encodings, dataset_file)

print("Tệp dataset_faces.dat đã được lưu thành công!")
