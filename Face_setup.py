import os
import cv2
import pickle
import face_recognition
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Đường dẫn đến thư mục chứa hình ảnh khuôn mặt
dataset_dir = '/home/admin/Desktop/FACE_V3/dataset'

# Khởi tạo từ điển để lưu trữ encoding và ID
all_face_encodings = {}

# Duyệt qua tất cả các thư mục trong dataset_dir
for user_id in os.listdir(dataset_dir):
    user_folder = os.path.join(dataset_dir, user_id)
    
    if os.path.isdir(user_folder):  # Kiểm tra xem có phải là thư mục không
        for image_name in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_name)
            
            # Chỉ xử lý các tệp hình ảnh
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Đọc hình ảnh và trích xuất encoding khuôn mặt
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)

                    # Nếu tìm thấy encoding, thêm vào từ điển
                    if face_encodings:
                        all_face_encodings[user_id] = face_encodings[0]

                except Exception as e:
                    print(f"Không thể xử lý {image_path}: {e}")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = list(all_face_encodings.values())
y = list(all_face_encodings.keys())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVM
clf = svm.SVC(gamma='scale', probability=True)
clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy:.2f}")

# Lưu mô hình đã huấn luyện
with open('face_recognition_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

print("Mô hình đã được lưu thành công!")

# Lưu dataset_faces.dat chứa tất cả các mã hóa khuôn mặt
with open('dataset_faces.dat', 'wb') as dataset_file:
    pickle.dump(all_face_encodings, dataset_file)

print("Tệp dataset_faces.dat đã được lưu thành công!")