import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from app import globals
df_country =globals.df_country


t_real = globals.t_real

h = 1

#Hàm mô hình logistic
def logistic_model(t, r, K, P0):
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))



def f(t,P):
    
    r = globals.r
    K = globals.K
    return r * P * (1 - P / K)


def rk4(t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6




def adam4(t,P):
    n = len(t)
    h = 1
    for i in range(3):
        P[i+1] = rk4(t[i], P[i], h) # h = 1

    # ---- ABM4 bắt đầu từ bước thứ 4 ----
    for i in range(3, n-1):
        f_n = f(t[i], P[i])
        f_n1 = f(t[i-1], P[i-1])
        f_n2 = f(t[i-2], P[i-2])
        f_n3 = f(t[i-3], P[i-3])

        # Predictor: Adams-Bashforth 4
        P_predict = P[i] + (h/24)*(55*f_n - 59*f_n1 + 37*f_n2 - 9*f_n3)

        # Corrector: Adams-Moulton 4
  
        f_predict = f(t[i+1], P_predict)
        P[i+1] = P[i] + (h/24)*(9*f_predict + 19*f_n - 5*f_n1 + f_n2)

    return t,P

def logistic_derivative(t, P, r, K):
    return r * P * (1 - P / K)
def rkf45(f, t_span, y0, h_initial=1.0, tol=1e-6, args=()):
    t0, tf = t_span
    t = [t0]
    y = [y0]
    h = h_initial

    # Các hệ số RKF45 (Butcher tableau)
    a = np.array([
        [0, 0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0, 0, 0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
    ])
    b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])  # RK4
    b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]) 

    while t[-1] < tf:
        # Tính các hệ số k
        k = np.zeros(6)
        k[0] = h * f(t[-1], y[-1], *args)
        for i in range(1, 6):
            t_inter = t[-1] + a[i, i-1] * h
            y_inter = y[-1] + np.sum(a[i, :i] * k[:i])
            k[i] = h * f(t_inter, y_inter, *args)

        # Tính nghiệm xấp xỉ bậc 4 và bậc 5
        y4 = y[-1] + np.sum(b4 * k)
        y5 = y[-1] + np.sum(b5 * k)

        # Ước lượng sai số
        error = np.abs(y5 - y4)

        # Điều chỉnh bước thời gian
        if error < tol or h < 1e-8:
            t.append(t[-1] + h)
            y.append(y5)  # Chọn nghiệm bậc 5 (chính xác hơn)
            # Tăng h nếu sai số nhỏ
            if error == 0:
                h = h * 2
            else:
                h = 0.9 * h * (tol / error) ** 0.2
        else:
            # Giảm h nếu sai số lớn
            h = 0.9 * h * (tol / error) ** 0.25

        # Đảm bảo không vượt quá tf
        h = min(h, tf - t[-1])

    return np.array(t), np.array(y)