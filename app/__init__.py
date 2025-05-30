# Khởi tạo Flask app và register các Blueprint


from flask import Flask



from .home import home_blueprint





def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config') #tải cấu hình từ lớp config từ file config.py

    # Đăng ký các blueprint
    app.register_blueprint(home_blueprint)

    return app

