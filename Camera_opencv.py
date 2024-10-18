import cv2
from picamera2 import Picamera2

# Khởi tạo camera với Picamera2
picam2 = Picamera2()
picam2.start()

# Tải mô hình Haar Cascade để nhận diện khuôn mặt
haarcascade_path = '/home/admin/Desktop/FACE_V3/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

if face_cascade.empty():
    print("Không thể tải mô hình Haar Cascade.")
    exit()

while True:
    # Chụp frame từ Picamera2
    frame = picam2.capture_array()

    # Chuyển đổi frame sang định dạng BGR nếu cần
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Chuyển frame sang ảnh xám để nhận diện khuôn mặt
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong ảnh
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Vẽ hình chữ nhật quanh các khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Hiển thị frame với các khuôn mặt được đánh dấu
    cv2.imshow("Camera - Face Detection", frame_bgr)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng các cửa sổ
picam2.stop()
cv2.destroyAllWindows()
