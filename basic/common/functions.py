# coding: utf-8
import numpy as np


# 평균 제곱 오차 MSE (Mean Squared Error)
def mean_squared_error(t, y):
    return np.sum((t - y)**2).mean()


# 평균 제곱근 오차 RMSE ( Root Mean Squared Error )
def root_mean_squared_error(t, y):
    return np.sqrt(mean_squared_error(t, y))


# 교차 엔트로피 오차 CEE(Cross Entropy Error)
# label의 값이 one_hot encoding일 때만 사용이 가능.
def cross_entropy_error(t, y):
    delta = 1e-7  # log 0을 막기위해
    cost = -np.sum(t * np.log(y + delta))
    return cost


# 미니배치 교차 엔트로피 오차 CEE(Cross Entropy Error)
# 기존 CEE에서 행렬 처리랑 평균값 계산만 추가한거
def cross_entropy_error_avg(t, y):
    batch_size = y.shape[0]
    cost = 0
    if batch_size == 1:
        return cross_entropy_error(t, y)
    else:
        for idx in range(0, batch_size):
            cost += cross_entropy_error(t[idx], y[idx]) / batch_size
        return cost


# 활성화 함수 softmax함수
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 오버플로 대책
    y = np.exp(x) / np.sum(np.exp(x))
    return y


# sigmoid 함수
def sigmoid(x):
    exp_x = np.exp(-x)
    y = 1 / (1 + exp_x)
    return y


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)