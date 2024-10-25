from RPLCD.i2c import CharLCD
import time

# Khởi tạo màn hình LCD với địa chỉ I2C (thường là 0x27 hoặc 0x3F)
lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)

def display_hello():
    """Hiển thị thông báo Hello trên màn hình LCD"""
    lcd.clear()  # Xóa màn hình LCD trước khi hiển thị
    lcd.write_string("Hello")  # Hiển thị chữ "Hello"
    time.sleep(5)  # Giữ màn hình hiển thị trong 5 giây

# Gọi hàm để hiển thị Hello
try:
    display_hello()
finally:
    lcd.clear()  # Xóa màn hình sau khi hiển thị xong
