import cv2
import pickle
from picamera2 import Picamera2
import face_recognition
import numpy as np
import RPi.GPIO as GPIO
import time
import os
import threading
import smtplib
from email.message import EmailMessage
import imghdr

# Khai báo thông tin email
Sender_email = "hatrung2003vt@gmail.com"
Reciever_Email = "duongtuan1008@gmail.com"
pass_sender = "aafj gckj dfsd eddd"

# Hàm gửi email khi phát hiện xâm nhập (Unknown)
def SendEmail(sender, pass_sender, receiver, image_path):
    newMessage = EmailMessage()
    newMessage['Subject'] = "CANH BAO !!!"
    newMessage['From'] = sender
    newMessage['To'] = receiver
    newMessage.set_content('CANH BAO AN NINH - Phát hiện xâm nhập!')

    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f)
        if image_type is None:
            image_type = 'jpeg'
        image_name = os.path.basename(f.name)
        newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender, pass_sender)
        smtp.send_message(newMessage)
    print(f"Email đã được gửi với ảnh đính kèm: {image_path}")

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

# Đọc dữ liệu khuôn mặt đã lưu
try:
    with open('dataset_faces.dat', 'rb') as file:
        all_face_encodings = pickle.load(file)
except FileNotFoundError:
    print("Tệp dữ liệu khuôn mặt không tồn tại.")
    exit()

# Chuyển đổi từ điển thành danh sách các encoding và ID
known_face_encodings = list(all_face_encodings.values())
known_face_ids = list(all_face_encodings.keys())

# Ngưỡng khoảng cách tối đa để coi là trùng khớp
face_recognition_threshold = 0.45  # Bạn có thể điều chỉnh ngưỡng này

# Tạo thư mục lưu ảnh nếu chưa tồn tại
output_dir = "user_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hàm để chụp và lưu ảnh trong luồng riêng
def capture_and_save_image(frame, name):
    # Tạo tên tệp với định dạng "Name_YYYY-MM-DD_HH-MM-SS.jpg"
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    img_filename = f"{output_dir}/{name}_{timestamp}.jpg"
    cv2.imwrite(img_filename, frame)
    print(f"Ảnh đã được lưu: {img_filename}")
    return img_filename  # Trả về đường dẫn ảnh

# Hàm gửi email trong luồng riêng
def send_email_thread(image_path):
    SendEmail(Sender_email, pass_sender, Reciever_Email, image_path)

while True:
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
                # So sánh với các encoding đã lưu
                distances = face_recognition.face_distance(known_face_encodings, face_encoding[0])
                best_match_index = np.argmin(distances)  # Lấy chỉ số của khoảng cách nhỏ nhất

                # Kiểm tra xem khoảng cách có nhỏ hơn ngưỡng không
                if distances[best_match_index] < face_recognition_threshold:
                    name = known_face_ids[best_match_index]
                    print(f"Mở khóa cửa cho người dùng: {name}")
                    GPIO.output(RELAY_PIN, GPIO.HIGH)

                    # Sử dụng luồng riêng để chụp và lưu ảnh
                    threading.Thread(target=capture_and_save_image, args=(frame_bgr, name)).start()

                    print("Khóa cửa lại.")
                    GPIO.output(RELAY_PIN, GPIO.LOW)
                else:
                    name = "Unknown"
                    print("Người dùng không xác định, cửa vẫn khóa.")

                    # Chụp và lưu ảnh cho người không xác định
                    img_filename = capture_and_save_image(frame_bgr, name)

                    # Gửi email trong một luồng riêng
                    threading.Thread(target=send_email_thread, args=(img_filename,)).start()

                # Hiển thị ID trên khung hình
                cv2.putText(frame_bgr, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
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