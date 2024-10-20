from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import os
import shutil
import subprocess
import threading
from flask_socketio import SocketIO

app = Flask(__name__)
app.secret_key = 'your_secret_key'
socketio = SocketIO(app)

# Cấu hình SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Model User lưu thông tin người dùng và đường dẫn tới thư mục ảnh
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    folder_path = db.Column(db.String(200), nullable=False)

# Hàm đồng bộ người dùng từ thư mục dataset vào cơ sở dữ liệu
def sync_users_from_dataset():
    dataset_dir = os.path.join(os.getcwd(), 'dataset')

    # Duyệt qua tất cả các thư mục con trong dataset
    for user_folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, user_folder)

        if os.path.isdir(folder_path):
            user_in_db = User.query.filter_by(name=user_folder).first()
            if not user_in_db:
                new_user = User(name=user_folder, folder_path=folder_path)
                db.session.add(new_user)
    
    db.session.commit()

# Hàm chạy Face_setup.py trong nền
def run_face_setup_background():
    try:
        # Chạy Face_setup.py và đợi cho đến khi hoàn tất
        result = subprocess.run(["python", "Face_setup.py"], check=True, capture_output=True, text=True)
        # Khi mô hình cập nhật xong, gửi thông báo cho client
        socketio.emit('model_updated', {'message': "Mô hình đã được cập nhật thành công!"})
        print(result.stdout)  # In thông tin ra console để kiểm tra nếu cần
    except subprocess.CalledProcessError as e:
        # Gửi thông báo lỗi cho client
        socketio.emit('model_updated', {'message': f"Có lỗi khi chạy Face_setup.py: {e.stderr}"})
        print(f"Lỗi chi tiết: {e.stderr}")  # In lỗi chi tiết ra console để kiểm tra

# Route để phục vụ các tệp trong thư mục dataset
@app.route('/dataset/<path:filename>')
def serve_dataset_file(filename):
    dataset_directory = os.path.join(os.getcwd(), 'dataset')
    return send_from_directory(dataset_directory, filename)

# Trang chính - Liệt kê tất cả người dùng
@app.route('/')
def index():
    users = User.query.all()
    return render_template('index.html', users=users)

# Thêm người dùng mới
@app.route('/add', methods=['POST'])
def add_user():
    if request.method == 'POST':
        name = request.form['name']
        images = request.files.getlist('images')

        folder_path = os.path.join('dataset', name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for image in images:
            image_path = os.path.join(folder_path, image.filename)
            image.save(image_path)

        new_user = User(name=name, folder_path=folder_path)
        db.session.add(new_user)
        db.session.commit()

        # Chạy cập nhật mô hình trong nền
        threading.Thread(target=run_face_setup_background).start()

        flash('Thêm người dùng thành công! Mô hình đang được cập nhật...', 'success')
        return redirect(url_for('index'))

# Sửa thông tin người dùng
@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit_user(id):
    user = User.query.get_or_404(id)
    if request.method == 'POST':
        new_name = request.form['name']
        new_images = request.files.getlist('new_images')  # Ảnh thêm mới
        replace_images = request.files.getlist('replace_images')  # Ảnh thay thế toàn bộ

        # Kiểm tra nếu cả hai danh sách ảnh đều rỗng
        if not any(image.filename != '' for image in new_images) and not any(image.filename != '' for image in replace_images):
            flash('Vui lòng chọn ảnh để cập nhật.', 'danger')
            return redirect(url_for('edit_user', id=id))

        # Kiểm tra nếu người dùng cố gắng thực hiện cả hai hành động cùng lúc
        if any(image.filename != '' for image in new_images) and any(image.filename != '' for image in replace_images):
            flash('Chỉ được thêm ảnh mới hoặc thay thế toàn bộ ảnh, không được thực hiện cả hai cùng lúc.', 'danger')
            return redirect(url_for('edit_user', id=id))

        # Đổi tên thư mục nếu người dùng đổi tên
        old_folder_path = user.folder_path
        new_folder_path = os.path.join('dataset', new_name)
        if new_name != user.name:
            os.rename(old_folder_path, new_folder_path)
            user.name = new_name
            user.folder_path = new_folder_path

        # Nếu có ảnh mới để thay thế toàn bộ ảnh cũ
        if replace_images and any(image.filename != '' for image in replace_images):
            # Xóa tất cả ảnh cũ
            for image_file in os.listdir(new_folder_path):
                image_path = os.path.join(new_folder_path, image_file)
                if os.path.isfile(image_path) and image_file.lower().endswith(('jpg', 'jpeg', 'png')):
                    os.remove(image_path)

            # Lưu ảnh thay thế
            for image in replace_images:
                image_path = os.path.join(new_folder_path, image.filename)
                image.save(image_path)
            flash('Thay thế toàn bộ ảnh thành công!', 'success')

        # Nếu có ảnh mới để thêm vào
        if new_images and any(image.filename != '' for image in new_images):
            # Lưu thêm ảnh mới mà không xóa ảnh cũ
            for image in new_images:
                image_path = os.path.join(new_folder_path, image.filename)
                # Đảm bảo không ghi đè ảnh cũ nếu tên ảnh mới trùng tên
                if os.path.exists(image_path):
                    filename, extension = os.path.splitext(image.filename)
                    counter = 1
                    while os.path.exists(image_path):
                        new_filename = f"{filename}_{counter}{extension}"
                        image_path = os.path.join(new_folder_path, new_filename)
                        counter += 1
                # Lưu ảnh với tên không trùng
                image.save(image_path)
            flash('Thêm ảnh mới thành công!', 'success')

        db.session.commit()

        # Chạy cập nhật mô hình trong nền
        threading.Thread(target=run_face_setup_background).start()

        # Giữ người dùng lại ở trang chỉnh sửa
        flash('Sửa thông tin người dùng thành công! Mô hình đang được cập nhật...', 'success')
        return redirect(url_for('edit_user', id=id))  # Điều hướng lại trang edit thay vì về trang chủ

    # Lấy danh sách ảnh từ thư mục người dùng
    image_files = os.listdir(user.folder_path)
    image_files = [f for f in image_files if os.path.isfile(os.path.join(user.folder_path, f)) and f.lower().endswith(('jpg', 'jpeg', 'png'))]

    return render_template('edit.html', user=user, image_files=image_files)


# Thay thế ảnh của người dùng
@app.route('/replace_image/<int:id>/<image_name>', methods=['POST'])
def replace_image(id, image_name):
    user = User.query.get_or_404(id)
    image_path = os.path.join(user.folder_path, image_name)
    new_image = request.files['new_image']

    if new_image and os.path.exists(image_path):
        # Xóa ảnh cũ
        os.remove(image_path)
        # Lưu ảnh mới với cùng tên ảnh cũ
        new_image_path = os.path.join(user.folder_path, image_name)
        new_image.save(new_image_path)
        flash(f'Ảnh đã được thay thế thành công!', 'success')
        flash('Mô hình đang được cập nhật...', 'success')
        # Chạy cập nhật mô hình trong nền
        threading.Thread(target=run_face_setup_background).start()

    else:
        flash(f'Không tìm thấy ảnh để thay thế hoặc không có ảnh mới!', 'danger')

    return redirect(url_for('edit_user', id=id))


# Xóa ảnh của người dùng
@app.route('/delete_image/<int:id>/<image_name>', methods=['POST'])
def delete_image(id, image_name):
    user = User.query.get_or_404(id)
    image_path = os.path.join(user.folder_path, image_name)
    if os.path.exists(image_path):
        os.remove(image_path)
        flash(f'Xóa ảnh {image_name} thành công!', 'success')
        flash('Mô hình đang được cập nhật...', 'success')
        # Chạy cập nhật mô hình trong nền
        threading.Thread(target=run_face_setup_background).start()

    else:
        flash(f'Không tìm thấy ảnh {image_name}', 'danger')
    
    return redirect(url_for('edit_user', id=id))

# Xóa người dùng
@app.route('/delete/<int:id>')
def delete_user(id):
    user = User.query.get_or_404(id)
    try:
        # Xóa thư mục chứa ảnh người dùng
        if os.path.exists(user.folder_path):
            shutil.rmtree(user.folder_path)

        # Xóa người dùng trong database
        db.session.delete(user)
        db.session.commit()

        # Chạy cập nhật mô hình trong nền
        threading.Thread(target=run_face_setup_background).start()

        flash('Xóa người dùng thành công! Mô hình đang được cập nhật...', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Có lỗi xảy ra: {e}', 'danger')
        return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        sync_users_from_dataset()

    socketio.run(app, debug=True)
