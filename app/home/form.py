from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, FloatField, IntegerField, RadioField
from wtforms.validators import DataRequired, Optional

class SelectForm(FlaskForm):
    country = SelectField('Chọn quốc gia', validators=[DataRequired()], choices=[])
    fit_mode = SelectField('Chọn kiểu lấy tham số', choices=[
        ('fit', 'Sử dụng r, K fit từ dữ liệu thực tế'),
        ('manual', 'Tự nhập tham số mô phỏng r, K và khoảng năm')
    ])
    r = FloatField('r', validators=[Optional()])
    K = FloatField('K', validators=[Optional()])
    year_start = IntegerField('Từ năm', validators=[Optional()])
    year_end = IntegerField('Đến năm', validators=[Optional()])

    method = RadioField('Chọn phương pháp', choices = 
                         [
                             ('rk4', 'Phương pháp Runge Kutta bậc 4'),
                             ('adam', 'Phương pháp Adam-Bashforth-Moulton bậc 4'),
                             ('rk45', 'Phương pháp Runge–Kutta–Fehlberg (RK45)'),
                             ('all', 'Chọn tất cả')
                         ]
                         )
    compare = RadioField('Chọn kiểu so sánh', choices = 
                         [
                             ('dan-so', 'So sánh dân số'),
                             ('ss-tuyet-doi', 'So sánh sai số tuyệt đối'),
                             ('ss-tuong-doi', 'So sánh sai số tương đối'),
                            ('ss-tuyet-doi-real', 'So sánh sai số tuyệt đối thực tế'),
                             ('ss-tuong-doi-real', 'So sánh sai số tương đối thực tế')
                         ]
                         )
    submit = SubmitField('Xem kết quả')