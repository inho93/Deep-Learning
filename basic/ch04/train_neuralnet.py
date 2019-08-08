# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
from basic.ch04.two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist


# data 세팅
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print('x_train.shape :', x_train.shape)
print('t_train.shape :', t_train.shape)

# 하이퍼 파라미터
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0]
batch_size = 100 # 미니배치 크기
learning_rate = 0.1

# 저장할 리스트 선언
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1 에폭당 반복 수
iter_per_epoch = int(train_size/batch_size)

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)

# 더미 테스트
x = np.random.rand(1, 784)
t = np.random.rand(1, 10)
grads = net.numerical_gradient(x, t) # 테스트용 기울기 계산
print(grads['W1'].shape)

# 예측 -> 손실 함수 값 구함 -> 함수를 최솟값으로 -> 각 매개변수별 편미분 -> 기울기를 최솟값(경사 하강법) -> 매개변수 반영
for step in range(iters_num):
    # Mini-Batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산 수치미분
    #grad = net.numerical_gradient(x_batch, t_batch)
    # 오차 역전파
    grad = net.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -=learning_rate * grad[key]

    # 학습 과정 기록
    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1 epoch 당 정확도 계산
    if (step+1) % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("Step:{:04d}, Train Acc: {:.5f}, Test Acc:{:5f}".format(step+1, train_acc, test_acc))
