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
from RPLCD.i2c import CharLCD
# Khởi tạo màn hình LCD với địa chỉ I2C (thường là 0x27 hoặc 0x3F)
lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)
# Khai báo biến log_file là tệp .txt
log_file = "event_log.txt"  # Đường dẫn tệp TXT để lưu nhật ký sự kiện

# Hàm khởi tạo tệp TXT nếu tệp chưa tồn tại
def init_log_file():
    if not os.path.exists(log_file):  # Kiểm tra nếu tệp chưa tồn tại
        with open(log_file, mode='w') as file:
            file.write("Thời gian\t\t\tSự kiện\n")  # Ghi tiêu đề

# Hàm ghi nhật ký sự kiện vào tệp TXT (Chỉ ghi sự kiện, không ghi chi tiết)
def log_event(event_type):
    with open(log_file, mode='a') as file:
        timestamp = time.strftime('%H:%M:%S  %d-%m-%Y')  # Lấy thời gian hiện tại
        file.write(f"{timestamp}  {event_type}\n")  # Ghi sự kiện vào tệp với 2 dấu cách
    print(f"Đã ghi sự kiện: {event_type}")  # Thông báo đã ghi sự kiện
# Bảng bàn phím 4x3
KEYPAD = [
    ['1', '2', '3'],
    ['4', '5', '6'],
    ['7', '8', '9'],
    ['*', '0', '#']
]
# Định nghĩa chân GPIO cho hàng và cột
ROW_PINS = [6, 13, 19, 26]  # Các chân cho hàng R1, R2, R3, R4
COL_PINS = [12, 16, 20]     # Các chân cho cột C1, C2, C3

# Thiết lập GPIO
GPIO.setmode(GPIO.BCM)

# Thiết lập các chân hàng là output
for row in ROW_PINS:
    GPIO.setup(row, GPIO.OUT)
# Thiết lập các chân cột là input với pull-down resistor
for col in COL_PINS:
    GPIO.setup(col, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
# Hàm để đọc mật khẩu từ tệp
def read_password_from_file():
    password_file = "password.txt"
    if os.path.exists(password_file):
        with open(password_file, 'r') as file:
            password = file.readline().strip()  # Đọc dòng đầu tiên và bỏ các ký tự thừa
            return password
    else:
        print("Tệp mật khẩu không tồn tại!")
        return None
# Hàm đọc phím từ bàn phím ma trận với debounce
def read_keypad():
    key_pressed = None
    for i in range(len(ROW_PINS)):
        GPIO.output(ROW_PINS[i], GPIO.HIGH)  # Kích hoạt hàng hiện tại

        for j in range(len(COL_PINS)):
            if GPIO.input(COL_PINS[j]) == GPIO.HIGH:  # Phát hiện nhấn phím
                key_pressed = KEYPAD[i][j]  # Lưu lại phím đã nhấn
                time.sleep(0.5)  # Thêm thời gian nghỉ để tránh xử lý lặp lại phím nhấn
                while GPIO.input(COL_PINS[j]) == GPIO.HIGH:
                    pass  # Đợi đến khi phím được thả ra

        GPIO.output(ROW_PINS[i], GPIO.LOW)

    return key_pressed  # Trả về phím đã nhấn


# Hàm kiểm tra mật khẩu từ bàn phím với debounce
def check_password():
    password = read_password_from_file()  # Đọc mật khẩu từ tệp

    if password is None:
        print("Không thể kiểm tra mật khẩu vì tệp không tồn tại.")
        return

    entered_password = ""
    print("Nhập mật khẩu:")
    lcd.write_string("Nhap mat khau:")
    
    while len(entered_password) < len(password):
        key = read_keypad()

        if key:
            print(f"Phím nhấn: {key}")
            if key == '#':
                print("Dừng nhập.")
                break   
            # Hiển thị "*" để ẩn mật khẩu khi người dùng nhập
            entered_password += key
            masked_password = "*" * len(entered_password)
            time.sleep(0.2)  # Đợi thêm thời gian sau mỗi lần nhấn phím
    time.sleep(2)
    lcd.clear()
    if entered_password == password:
        print("Mật khẩu đúng!")
        threading.Thread(target=unlock_door).start()  # Mở cửa nếu mật khẩu đúng
        log_event("Mật khẩu đúng mở cửa")
    else:
        print("Mật khẩu sai!")
        threading.Thread(target=lock_sound).start()  # Phát âm báo nếu sai
        log_event("Mật khẩu sai cửa khóa")
# Khai báo thông tin email
Sender_email = "hatrung2003vt@gmail.com"
Reciever_Email = "tett43644@gmail.com"
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

GPIO.setmode(GPIO.BCM)
RELAY_PIN = 17
GPIO.setup(RELAY_PIN, GPIO.OUT)
RELAY_SOUND = 27
GPIO.setup(RELAY_SOUND, GPIO.OUT)


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

# Hàm điều khiển relay mở khóa cửa trong 10 giây
def unlock_door():
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # Mở khóa cửa
    print("Relay mở - Cửa đã mở")
    log_event("Mở cửa")    
    # Hiển thị sự kiện lên màn hình LCD
    lcd.write_string("Mo cua")
    time.sleep(2)
    lcd.clear()
    time.sleep(10)  # Đợi 10 giây
    GPIO.output(RELAY_PIN, GPIO.LOW)  # Khóa cửa lại
    print("Cửa đóng lại")
    log_event("Cửa đóng lại")
    lcd.write_string("Cua dong lai")
    time.sleep(2)
    lcd.clear()

def lock_sound():
    GPIO.output(RELAY_SOUND, GPIO.HIGH)  # Phát còi báo
    print("Sai")
    time.sleep(1)  # Đợi 1 giây
    GPIO.output(RELAY_SOUND, GPIO.LOW)  # Tắt còi

# Khai báo từ điển để lưu thời gian nhận diện gần nhất cho mỗi người dùng
last_recognition_time = {}
# Thời gian giãn cách tối thiểu giữa các lần nhận diện (tính bằng giây)
min_recognition_interval = 10  # Ví dụ: 10 giây
# Khai báo thời gian chờ trước khi kết luận "Unknown" (ví dụ 1 giây)
delay_before_unknown = 5  # Ví dụ: 1 giây
# Cờ để xác định xem có phải đã qua thời gian đợi trước khi xác định "Unknown" hay không
unknown_timeout_flag = False
# Hàm kiểm tra xem khuôn mặt có khớp không, trả về True nếu khớp, False nếu không
def check_face_recognition(face_encoding):
    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(distances)
    
    if distances[best_match_index] < face_recognition_threshold:
        return known_face_ids[best_match_index]  # Trả về ID nếu khớp
    else:
        return None  # Trả về None nếu không khớp
# Sửa đổi vòng lặp nhận diện khuôn mặt
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
                name = check_face_recognition(face_encoding[0])

                if name:
                    # Nếu nhận diện được người dùng
                    unknown_timeout_flag = False  # Đặt lại cờ nếu nhận diện đúng người dùng
                    current_time = time.time()
                    if name not in last_recognition_time or (current_time - last_recognition_time[name]) > min_recognition_interval:
                        print(f"Mở khóa cửa cho người dùng: {name}")
                        log_event(f"Cửa mở cho: {name}")
                        last_recognition_time[name] = current_time  # Cập nhật thời gian nhận diện
                        threading.Thread(target=unlock_door).start()
                        # Sử dụng luồng riêng để chụp và lưu ảnh
                        img_filename = capture_and_save_image(frame_bgr, name)
                        
                else:
                    # Đợi trước khi kết luận là "Unknown"
                    if not unknown_timeout_flag:
                        start_unknown_time = time.time()
                        unknown_timeout_flag = True  # Đặt cờ để bắt đầu đợi
                    elapsed_time = time.time() - start_unknown_time
                    # Đợi đến khi thời gian đã trôi qua để xác định là "Unknown"
                    if elapsed_time >= delay_before_unknown:
                        # Kiểm tra lại trước khi thực sự kết luận là "Unknown"
                        name = check_face_recognition(face_encoding[0])
                        if not name:  # Chỉ khi chắc chắn vẫn là "Unknown"
                            name = "Unknown"
                            current_time = time.time()
                            if "Unknown" not in last_recognition_time or (current_time - last_recognition_time["Unknown"]) > min_recognition_interval:
                                last_recognition_time["Unknown"] = current_time  # Cập nhật thời gian nhận diện
                                log_event("Người dùng không xác định")
                                threading.Thread(target=lock_sound).start()
                                img_filename = capture_and_save_image(frame_bgr, name)# Chụp và lưu ảnh cho người không xác định
                                threading.Thread(target=send_email_thread, args=(img_filename,)).start()
                                threading.Thread(target=check_password).start()

                # Hiển thị ID trên khung hình
                cv2.putText(frame_bgr, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Hiển thị frame với các khuôn mặt được đánh dấu
    cv2.imshow("Camera - Face Detection and Recognition", frame_bgr)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng các cửa sổ
picam2.stop()  # Dừng camera
GPIO.output(RELAY_PIN, GPIO.LOW)  # Đảm bảo cửa khóa khi dừng camera
GPIO.cleanup()
cv2.destroyAllWindows()