import os
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use("Agg")  # Dùng backend không cần GUI
import matplotlib.pyplot as plt
import numpy as np
from app import globals

from flask import Blueprint, render_template, request,jsonify
from .form import SelectForm

import io
import base64
from matplotlib.figure import Figure

home_blueprint = Blueprint("home", __name__, template_folder="templates")


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # trở về thư mục gốc project
file_path = os.path.join(BASE_DIR, "dataset", "dataset.csv")
df = pd.read_csv(file_path)

h = 1


def load_df_country(df, country_name):
    try:
        df_country = df[df["Country Name"] == country_name].drop(
            columns=["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
        )
        df_country = df_country.transpose().reset_index()
        df_country.columns = ["Year", "Population"]
        df_country["Year"] = df_country["Year"].astype(int)
        df_country = df_country.sort_values(by="Year")
        df_country.reset_index(drop=True, inplace=True)
        df_country.index = df_country.index + 1

        t = df_country["Year"].values
        P = df_country["Population"].values

        return df_country, t, P
    except Exception as e:
        print("Lỗi khi load country:", e)
        return None


# =============


def fit_r_K(logistic_model, t_index, P_real):
    # Fit tham số r, K
    res, _ = curve_fit(logistic_model, t_index, P_real, p0=[0.02, max(P_real) * 1.5])
    r_fit, K_fit = res

    print(f"Tham số theo mô hình Logistic:")
    print(f"r fit = {r_fit}")
    print(f"K fit = {K_fit}")
    return r_fit, K_fit


def set_r_and_K(r, K):
    globals.r = r
    globals.K = K


def show_fit_chart(r_fit, K_fit, r_input, K_input, t_index, logistic_model):
    t_real = globals.t_real
    
    P_real = globals.P_real

    t_fit = np.linspace(0, t_index[-1], 100)
    P_fit = logistic_model(t_fit, r_fit, K_fit)
    plt.figure(figsize=(15, 9))
    plt.plot(t_real, P_real, "o", label="Dữ liệu thực tế")
    plt.plot(t_fit + t_real.min(), P_fit, "-", label="Mô hình logistic fit")
    if r_input != None and K_input != None:
        t_input = np.linspace(0, t_index[-1], 100)
        P_input = logistic_model(t_input, r_input, K_input)
        plt.plot(t_input + t_real.min(), P_input, "b", label="Dữ liệu mô phỏng")
    plt.xlabel("Năm")
    plt.ylabel("Dân số")
    plt.title("Fit mô hình logistic cho dân số theo quốc gia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Xuất ảnh ra base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()

    return encoded


def show_compare_chart(t_real, t_rk4, P_rk4, P_exact, P_real, label):
    plt.figure(figsize=(15, 9))
    # Vẽ nghiệm RK4 (Logistic)
    plt.plot(t_rk4, P_rk4, "o--", label=label, markersize=4)

    # Vẽ nghiệm chính xác (Logistic)
    plt.plot(t_real, P_exact, "r-", label="Nghiệm chính xác", linewidth=2)

    # Vẽ dân số thực tế
    plt.plot(t_real, P_real,"o", label="Dân số thực tế", markersize=5)

    plt.xlabel("Năm")
    plt.ylabel("Dân số")
    plt.title(
        f"So sánh phương pháp {label} và nghiệm chính xác cho mô hình Logistic"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Xuất ảnh ra base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()

    return encoded

def show_compare_chart_all(t_real, t_rk4,t_adam,t_rkf45, P_rk4, P_exact,P_adam,P_rkf45, P_real, label1, label2 , label3):
    plt.figure(figsize=(15, 9))
    # Vẽ dân số thực tế
    plt.plot(t_real, P_real, "ko", label="Dân số thực tế", markersize=5)
    plt.plot(t_real, P_exact, "gs", label="Nghiệm chính xác", markersize=5, linewidth=2)
    
    # Vẽ nghiệm RK4 (Logistic)
    plt.plot(t_rk4, P_rk4, "r^-", label=label1, markersize=5,linewidth=2)
    plt.plot(t_adam, P_adam, "bv-", label=label2, markersize=5,linewidth=2)
    plt.plot(t_rkf45, P_rkf45, "m+", label="Nghiệm chính xác", markersize=5, linewidth=2)

   

    

    

    plt.xlabel("Năm")
    plt.ylabel("Dân số")
    plt.title(
        f"So sánh tất cả phương pháp và nghiệm chính xác cho mô hình Logistic"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Xuất ảnh ra base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()

    return encoded


def show_table_compare(t_real, P_real, P_rk4, P_exact , label,compare_type):
    # Lấy năm bắt đầu
    start_year = t_real.min()

    # năm thực tế
    years = t_real

    # Tạo DataFrame kết quả
    df_result = pd.DataFrame(
        {
            "Năm": years,
            "Dân số thực tế": P_real,
            f"Dân số tính bằng {label}": P_rk4,
            "Nghiệm chính xác": P_exact,  # P extract trong đk lý tưởng
        }
    )

    # Tính sai số tuyệt đối và tương đối
    # chất lượng phương pháp số, so sánh với nghiệm chính xác,kiểm tra RK4 có sai số bao nhiêu so với lý thuyết
    df_result["Sai số tuyệt đối"] = np.abs(
        df_result[f"Dân số tính bằng {label}"] - df_result["Nghiệm chính xác"]
    )
    df_result["Sai số tương đối"] = (
        df_result["Sai số tuyệt đối"] / df_result["Nghiệm chính xác"] * 100
    )

    # kiểm tra độ khớp với dữ liệu thực tế, cần so sánh với P_real.
    # là bài toán thực nghiệm, xem mô hình có phản ánh đúng dân số ngoài thực tế không
    df_result["Sai số tuyệt đối thực tế"] = np.abs(
        df_result[f"Dân số tính bằng {label}"] - df_result["Dân số thực tế"]
    )
    df_result["Sai số tương đối thực tế"] = (
        df_result["Sai số tuyệt đối thực tế"] / df_result["Dân số thực tế"] * 100
    )

    # Làm tròn cho hiển thị
    df_display = df_result.round(2)
    

    if compare_type == 'dan-so':
        table_compare = show_table_with_color_comparison(
        df=df_result,
        reference_column=f"Dân số thực tế",
        compare_columns=[f"Dân số tính bằng {label}"]
        )
    elif compare_type == 'ss-tuyet-doi':
        table_compare = show_table_with_color_comparison(
        df=df_result,
        reference_column=f"Sai số tuyệt đối",
        compare_columns=[f"Sai số tuyệt đối thực tế"]
        )
    elif compare_type == 'ss-tuong-doi':
        table_compare = show_table_with_color_comparison(
        df=df_result,
        reference_column=f"Sai số tương đối",
        compare_columns=[f"Sai số tương đối thực tế"]
        )
    elif compare_type == 'ss-tuyet-doi-real':
        table_compare = show_table_with_color_comparison(
        df=df_result,
        reference_column=f"Sai số tuyệt đối",
        compare_columns=[f"Sai số tuyệt đối thực tế"]
        )
    else:
        table_compare = show_table_with_color_comparison(
        df=df_result,
        reference_column=f"Sai số tương đối",
        compare_columns=[f"Sai số tương đối thực tế"]
        )
    return table_compare


def show_table_compare_all(t_real, P_real, P_rk4,P_adam,P_rkf45, P_exact , label1, label2, label3, compare_type):
    # Lấy năm bắt đầu
    start_year = t_real.min()

    # năm thực tế
    years = t_real

    # Tạo DataFrame kết quả
    df_result = pd.DataFrame(
        {
            "Năm": years,
            "Dân số thực tế": P_real,
             "Nghiệm chính xác": P_exact,  # P extract trong đk lý tưởng
            f"Dân số tính bằng {label1}": P_rk4,
            f"Dân số tính bằng {label2}": P_adam,
            f"Dân số tính bằng {label3}": P_rkf45
        }
    )

    # Tính sai số tuyệt đối và tương đối
    # chất lượng phương pháp số, so sánh với nghiệm chính xác,kiểm tra RK4 có sai số bao nhiêu so với lý thuyết
    df_result[f"Sai số tuyệt đối theo {label1}"] = np.abs(
        df_result[f"Dân số tính bằng {label1}"] - df_result["Nghiệm chính xác"]
    )
    
    df_result[f"Sai số tuyệt đối theo {label2}"] = np.abs(
        df_result[f"Dân số tính bằng {label2}"] - df_result["Nghiệm chính xác"]
    )
    df_result[f"Sai số tuyệt đối theo {label3}"] = np.abs(
        df_result[f"Dân số tính bằng {label3}"] - df_result["Nghiệm chính xác"]
    )
    df_result[f"Sai số tuyệt đối thực tế theo {label1}"] = np.abs(
        df_result[f"Dân số tính bằng {label1}"] - df_result["Dân số thực tế"]
    )
    df_result[f"Sai số tuyệt đối thực tế theo {label2}"] = np.abs(
        df_result[f"Dân số tính bằng {label2}"] - df_result["Dân số thực tế"]
    )
    df_result[f"Sai số tuyệt đối thực tế theo {label3}"] = np.abs(
        df_result[f"Dân số tính bằng {label3}"] - df_result["Dân số thực tế"]
    )


    df_result[f"Sai số tương đối theo {label1}"] = (
        df_result[f"Sai số tuyệt đối theo {label1}"] / df_result["Nghiệm chính xác"] * 100
    )
    df_result[f"Sai số tương đối theo {label2}"] = (
        df_result[f"Sai số tuyệt đối theo {label2}"] / df_result["Nghiệm chính xác"] * 100
    )
    df_result[f"Sai số tương đối theo {label3}"] = (
        df_result[f"Sai số tuyệt đối theo {label3}"] / df_result["Nghiệm chính xác"] * 100
    )
    # kiểm tra độ khớp với dữ liệu thực tế, cần so sánh với P_real.
    # là bài toán thực nghiệm, xem mô hình có phản ánh đúng dân số ngoài thực tế không
    
    
    
    df_result[f"Sai số tương đối thực tế theo {label1}"] = (
        df_result[f"Sai số tuyệt đối thực tế theo {label1}"] / df_result["Dân số thực tế"] * 100
    )
    df_result[f"Sai số tương đối thực tế theo {label2}"] = (
        df_result[f"Sai số tuyệt đối thực tế theo {label2}"] / df_result["Dân số thực tế"] * 100
    )
    df_result[f"Sai số tương đối thực tế theo {label3}"] = (
        df_result[f"Sai số tuyệt đối thực tế theo {label3}"] / df_result["Dân số thực tế"] * 100
    )
    # Làm tròn cho hiển thị
    df_result = df_result.round(2)
    table_compare = None
    if compare_type == 'dan-so':
        table_compare = show_table_with_color_comparison(
        df=df_result,
        reference_column=f"Dân số tính bằng {label1}",
        compare_columns=[f"Dân số tính bằng {label2}", f"Dân số tính bằng {label3}"]
        )
    elif compare_type == 'ss-tuyet-doi':
        table_compare = show_table_with_color_comparison(
        df=df_result,
        reference_column=f"Sai số tuyệt đối theo {label1}",
        compare_columns=[f"Sai số tuyệt đối theo {label2}", f"Sai số tuyệt đối theo {label3}"]
        )
    elif compare_type == 'ss-tuong-doi':
        table_compare = show_table_with_color_comparison(
        df=df_result,
        reference_column=f"Sai số tương đối theo {label1}",
        compare_columns=[f"Sai số tương đối theo {label2}", f"Sai số tương đối theo {label3}"]
        )
    elif compare_type == 'ss-tuyet-doi-real':
        table_compare = show_table_with_color_comparison(
        df=df_result,
        reference_column=f"Sai số tuyệt đối thực tế theo {label1}",
        compare_columns=[f"Sai số tuyệt đối thực tế theo {label2}", f"Sai số tuyệt đối thực tế theo {label3}"]
        )
    else:
        table_compare = show_table_with_color_comparison(
        df=df_result,
        reference_column=f"Sai số tương đối thực tế theo {label1}",
        compare_columns=[f"Sai số tương đối thực tế theo {label2}", f"Sai số tương đối thực tế theo {label3}"]
        )
    return table_compare


def show_table_with_color_comparison(
    df: pd.DataFrame,
    reference_column: str,
    compare_columns: list
) -> str:
    columns_to_check = [reference_column] + compare_columns

    def color_row(row):
        values = [row[col] for col in columns_to_check]
        is_all_equal = all(val == values[0] for val in values)
        style = "color: green;" if is_all_equal else "color: red;"
        return [style if col in columns_to_check else "" for col in df.columns]

    styled = df.style.apply(color_row, axis=1)
    return styled.to_html(classes="width:100%; max-width:100%", index=False)

def calculator_rk4(t_index, P0, rk4):
    t_rk4 = t_index
    n = len(t_rk4)

    P_rk4 = np.zeros(n)
    P_rk4[0] = P0  # Cần thêm dòng này
    for i in range(n - 1):
        P_rk4[i + 1] = rk4(t_rk4[i], P_rk4[i], h)

    return t_rk4, P_rk4


def calculator_adam(t_index, n, P0, adam4):
    t_adam = t_index
    P_adam = np.zeros(n)
    P_adam[0] = P0
    t_adam, P_adam = adam4(t_adam, P_adam)
    return t_adam, P_adam


# ===========


@home_blueprint.route("/", methods=["GET", "POST"])
def home():
    form = SelectForm()
    # list country de chon
    country_list = df["Country Name"].unique().tolist()
    # Cập nhật choices cho country field
    form.country.choices = [(c, c) for c in country_list]

    selected_country, t_real, P_real = None, None, None
    fit_chart = None
    compare_chart = None
    compare_table = None
    selected_method_label = None
    label_method_map = dict(form.method.choices)
    
    if form.validate_on_submit():
        if form.country.data:
            selected_country = form.country.data
        selected_fit_mode = form.fit_mode.data
        selected_method = form.method.data
        selected_method_label = label_method_map[selected_method]  # Kết quả: 'Runge Kutta bậc 4'

        compare_type = form.compare.data
    
        # load du lieu that tu dataset
        df_country, t_real, P_real = load_df_country(df, selected_country)
        globals.t_real = t_real
        globals.P_real = P_real
        t_index = t_real - t_real[0]
        P0 = P_real[0]  # dân số năm đầu, global
        globals.P0 = P0  # can su dung o models logistic

        from app.models.logistic import logistic_model

        r_input = 0
        K_input = 0
        
        # fit r, K
        r_fit, K_fit = fit_r_K(logistic_model, t_index, P_real)
        if selected_fit_mode == "fit":
            
            # r,K global
            set_r_and_K(r_fit, K_fit)

            from app.models.logistic import rk4, adam4, h, logistic_derivative, rkf45

            # if t_real is not None and P_real is not None:
            fit_chart = show_fit_chart(
                r_fit, K_fit, r_input, K_input, t_index, logistic_model
            )
            n = len(t_index)
            t_rk4, P_rk4 = calculator_rk4(t_index, P0, rk4)
            t_adam, P_adam = calculator_adam(t_index, n, P0, adam4)
            # Tính nghiệm RKF45 tại từng năm nguyên (trùng với dữ liệu thực tế)
            t_rkf45, P_rkf45 = rkf45(logistic_derivative, (t_index[0], t_index[-1]), P0, h_initial=1, tol=1e-6, args=(r_fit, K_fit))
            # Nội suy dân số RKF45 tại các năm nguyên
            from scipy.interpolate import interp1d
            rkf45_interp = interp1d(t_rkf45, P_rkf45, kind='linear', fill_value="extrapolate")
            t_rkf45_nam = t_index
            P_rkf45_nam = rkf45_interp(t_index)
            # Nghiệm chính xác trong điều kiện lý tưởng
            P_exact = logistic_model(t_index, r_fit, K_fit)
        
            if selected_method == 'all':
                label1 = label_method_map['rk4']
                label2 = label_method_map['adam']
                label3 = label_method_map['rk45']
                compare_table = show_table_compare_all(t_real,P_real, P_rk4, P_adam,P_rkf45_nam, P_exact,label1, label2, label3, compare_type )
                compare_chart = show_compare_chart_all(t_real, t_rk4,t_adam,t_rkf45_nam, P_rk4, P_exact, P_adam,P_rkf45_nam, P_real, label1, label2, label3)
            else:
                P_selected_method = P_rk4
                if selected_method =='rk4':
                    P_selected_method = P_rk4
                elif selected_method == 'adam':
                    P_selected_method = P_adam
                else:
                    P_selected_method = P_rkf45_nam
                
                compare_table = show_table_compare(
                        t_real, P_real, P_selected_method, P_exact, selected_method_label, compare_type
                    )
                compare_chart = show_compare_chart(t_real, t_rk4, P_selected_method, P_exact, P_real ,selected_method_label)
        else :
            start_year = form.year_start.data
            end_year = form.year_end.data
            r_input = form.r.data
            K_input = form.K.data
            set_r_and_K(r_input, K_input)
            r,K = globals.r, globals.K
            # Lọc t_real theo khoảng [start_year, end_year]
            if end_year == 2022:
                mask = (t_real >= start_year) & (t_real <= end_year)
                t_filtered = t_real[mask]
                P_filtered = P_real[mask]
            else: 
                # Lọc dữ liệu từ start_year đến 2022
                mask = (t_real >= start_year)
                t_filtered = t_real[mask]
                P_filtered = P_real[mask]

                # Thêm các năm sau 2022 đến end_year
                t_future = np.arange(2023, end_year + 1)
                P_future = [] 
                for year in t_future:
                    P_pred = logistic_model(year, r, K)
                    P_future.append(P_pred)
                
                # Nối dữ liệu quá khứ và tương lai lại
                t_filtered = np.concatenate((t_filtered, t_future))
                P_filtered = np.concatenate((P_filtered, P_future))
      
            t_index = t_filtered - t_filtered[0]
            P0 = P_filtered[0]

            
            globals.P0 = P0
            globals.t_real = t_filtered
            globals.P_real = P_filtered
            print(globals.t_real, globals.P_real)
            
            # if t_real is not None and P_real is not None:
            fit_chart = show_fit_chart(
                r_fit, K_fit, r_input, K_input, t_index, logistic_model
            )

            from app.models.logistic import rk4, adam4, h, logistic_derivative, rkf45
            t_rk4, P_rk4 = calculator_rk4(t_index, P0, rk4)
            n = len(t_index)
        
            t_adam, P_adam = calculator_adam(t_index, n, P0, adam4)
            
            # Tính nghiệm RKF45 tại từng năm nguyên (trùng với dữ liệu thực tế)
            t_rkf45, P_rkf45 = rkf45(logistic_derivative, (t_index[0], t_index[-1]), P0, h_initial=1, tol=1e-6, args=(r, K))
            
            # Nội suy dân số RKF45 tại các năm nguyên
            from scipy.interpolate import interp1d
            rkf45_interp = interp1d(t_rkf45, P_rkf45, kind='linear', fill_value="extrapolate")
            P_rkf45_nam = rkf45_interp(t_index)
            t_rkf45_nam = t_adam
            print((t_rk4))
            print((t_adam))
            print((t_rkf45_nam))
            # Nghiệm chính xác trong điều kiện lý tưởng
            P_exact = logistic_model(t_index, r, K)
            
            if selected_method == 'all':
                label1 = label_method_map['rk4']
                label2 = label_method_map['adam']
                label3 = label_method_map['rk45']
                compare_table = show_table_compare_all(t_filtered,P_filtered, P_rk4, P_adam,P_rkf45_nam, P_exact,label1, label2, label3, compare_type )
                compare_chart = show_compare_chart_all(t_filtered, t_rk4,t_adam,t_rkf45_nam, P_rk4, P_exact, P_adam,P_rkf45_nam, P_filtered, label1, label2, label3)
            else:
                P_selected_method = P_rk4
                if selected_method =='rk4':
                    P_selected_method = P_rk4
                elif selected_method == 'adam':
                    P_selected_method = P_adam
                else:
                    P_selected_method = P_rkf45_nam
                
                compare_table = show_table_compare(
                        t_filtered, P_filtered, P_selected_method, P_exact, selected_method_label, compare_type
                    )
                compare_chart = show_compare_chart(t_filtered, t_rk4, P_selected_method, P_exact, P_filtered ,selected_method_label)

            

    return render_template(
        "home/home.html",
        form=form,
        selected_country=selected_country,
        t_real=t_real,
        P_real=P_real,
        fit_chart=fit_chart,
        compare_table=compare_table,
        compare_chart=compare_chart,
        selected_method_label = selected_method_label,
        compare_type = compare_type
    )

    
