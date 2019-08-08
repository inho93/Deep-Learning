# coding: utf-8
import numpy as np


# 편미분을 통한 기울기
# 행렬로 각 변수의 기울기 계산
def numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 똑같은 0행렬

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / 2 * h

        x[idx] = tmp_val  # 값 복원

    return grad


# 편미분으로 기울기 산출 수치 미분
def numerical_gradient(f, x):
    grad = np.zeros_like(x)  # x와 똑같은 0행렬
    if x.ndim == 1:
        return numerical_gradient_no_batch(f, x)
    else:
        for idx, x in enumerate(x):
            grad[idx] = numerical_gradient_no_batch(f, x)
        return grad


# 최소값을 향해 가는 경사 하강법
def gradient_descent(f, x, lr=0.01, step_num=100):
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr * grad
    return x