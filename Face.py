import cv2
import pickle
from picamera2 import Picamera2
import face_recognition
import numpy as np
import RPi.GPIO as GPIO
import time
import os

# Cấu hình GPIO
GPIO.setmode(GPIO.BCM)
RELAY_PIN = 17
GPIO.setup(RELAY_PIN, GPIO.OUT)

print("Cửa khóa")
GPIO.output(RELAY_PIN, GPIO.LOW)

# Khởi tạo camera với Picamera2
picam2 = Picamera2()
picam2.start()  # Bật camera ngay lập tức

# Tải mô hình Haar Cascade để nhận diện khuôn mặt
haarcascade_path = '/home/admin/Desktop/FACE_V3/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

if face_cascade.empty():
    print("Không thể tải mô hình Haar Cascade.")
    exit()

# Đường dẫn đến dataset_faces.dat
face_data_path = 'dataset_faces.dat'
model_file_path = 'face_recognition_model.pkl'

def load_face_data():
    # Đọc dữ liệu khuôn mặt đã lưu
    try:
        with open(face_data_path, 'rb') as file:
            all_face_encodings = pickle.load(file)
        known_face_encodings = list(all_face_encodings.values())
        known_face_ids = list(all_face_encodings.keys())
        print("Đã tải dữ liệu khuôn mặt.")
        return known_face_encodings, known_face_ids
    except FileNotFoundError:
        print("Tệp dữ liệu khuôn mặt không tồn tại.")
        exit()

def get_file_modification_time(file_path):
    return os.path.getmtime(file_path)

# Khởi tạo lần đầu
known_face_encodings, known_face_ids = load_face_data()
last_model_update_time = get_file_modification_time(face_data_path)

# Ngưỡng khoảng cách tối đa để coi là trùng khớp
face_recognition_threshold = 0.45  # Bạn có thể điều chỉnh ngưỡng này

while True:
    # Kiểm tra xem dataset_faces.dat có được cập nhật không
    current_model_update_time = get_file_modification_time(face_data_path)
    if current_model_update_time != last_model_update_time:
        print("Phát hiện mô hình mới, tải lại dữ liệu khuôn mặt.")
        known_face_encodings, known_face_ids = load_face_data()
        last_model_update_time = current_model_update_time

    # Chụp frame từ Picamera2
    frame = picam2.capture_array()

    # Chuyển đổi frame sang định dạng BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Chuyển frame sang ảnh xám để nhận diện khuôn mặt
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Cắt khuôn mặt từ frame
            face_image = frame[y:y + h, x:x + w]
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Trích xuất đặc trưng khuôn mặt
            face_encoding = face_recognition.face_encodings(face_rgb)

            if len(face_encoding) > 0:  # Kiểm tra xem có encoding không
                # So sánh với các encoding đã lưu trong dataset
                distances = face_recognition.face_distance(known_face_encodings, face_encoding[0])
                best_match_index = np.argmin(distances)  # Lấy chỉ số của khoảng cách nhỏ nhất

                # Kiểm tra khoảng cách có nhỏ hơn ngưỡng không
                if distances[best_match_index] < face_recognition_threshold:
                    dataset_name = known_face_ids[best_match_index]
                else:
                    dataset_name = "Unknown"

                # Nếu khớp, mở khóa cửa
                if dataset_name != "Unknown":
                    print(f"Mở khóa cửa cho người dùng: {dataset_name}")
                    GPIO.output(RELAY_PIN, GPIO.HIGH)
                    time.sleep(5)  # Giữ khóa mở trong 5 giây
                    print("Khóa cửa lại.")
                    GPIO.output(RELAY_PIN, GPIO.LOW)
                else:
                    print("Không khớp. Cửa khóa.")
                    GPIO.output(RELAY_PIN, GPIO.LOW)

                # Hiển thị ID trên khung hình
                cv2.putText(frame_bgr, dataset_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        print("Không phát hiện khuôn mặt. Cửa khóa.")
        GPIO.output(RELAY_PIN, GPIO.LOW)

    # Hiển thị frame với các khuôn mặt được đánh dấu
    cv2.imshow("Camera - Face Detection and Recognition", frame_bgr)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng các cửa sổ
picam2.stop()  # Dừng camera
GPIO.output(RELAY_PIN, GPIO.LOW)  # Đảm bảo cửa khóa khi dừng camera
cv2.destroyAllWindows()
