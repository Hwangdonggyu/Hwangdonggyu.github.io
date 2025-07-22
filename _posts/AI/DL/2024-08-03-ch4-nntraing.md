---
title: 밑바닥부터 시작하는 딥러닝_CH4 신경망 학습
date: 2024-07-16 11:26:00 +0900
categories: [AI,DL]
tags: [AI,DL]
math: true
---

## 데이터 주도 학습

기계학습은 데이터가 생명이다. 데이터에서 답을 찾고 데이터에서 패턴을 발견하고 데이터로 이야기를 만드는 것이 바로 기계학습이다. 그래서 기계학습 중심에는 **데이터**가 존재한다.

## 훈련 데이터와 시험 데이터

기계학습 문제는 데이터를 **훈련 데이터(training data)**와 **시험 데이터(test data)**로 나눠 학습과 실험을 수행하는 것이 일반적이다. 우선 훈련 데이터만 사용하여 학습하면서 최적의 매개변수를 찾는다. 그런 다음 시험 데이터를 사용하여 앞서 훈련한 모델의 실력을 평가하는 것이다.

### 왜 훈련 데이터와 시험 데이터를 나눠야 할까?

그것은 우리가 원하는 것은 범용적으로 사용할 수 있는 모델이기 때문이다. 이 **범용 능력**을 제대로 평가하기 위해 훈련 데이터와 시험 데이터를 분리하는 것이다.

범용 능력은 아직 보지 못한 데이터(훈련 데이터에 포함되지 않는 데이터)로도 문제를 올바르게 풀어내는 능력이다. 이 범용 능력을 획득하는 것이 기계학습의 최종 목표이다.

데이터셋 하나로만 매개변수의 학습과 평가를 수행하면 올바른 평가가 될 수 없다. 수중의 데이터셋은 제대로 맞히더라도 다른 데이터셋에는 엉망인 일도 벌어진다. 참고로 한 데이터셋에만 지나치게 최적화된 상태를 **오버피팅(Overfitting)**이라고 한다. 오버피팅 피하기는 기계학습의 중요한 과제이기도 하다.

## 손실 함수

신경망은 ‘하나의 지표’를 기준으로 최적의 매개변수 값을 탐색한다. 신경망 학습에서 사용하는 지표는 **손실함수(loss function)**라고 한다. 이 손실 함수는 임의의 함수를 사용할 수도 있지만 일반적으로는 오차제곱합과 교차 엔트로피 오차를 사용한다.

### 오차제곱합

가장 많이 쓰이는 손실 함수는 **오차제곱합(Sum Squares for error, SSE)**이다. 수식으로는 다음과 같다.

$$
E = \frac{1}{2}\sum_k(y_k-t_k)^2
$$

여기서 $y_k$는 신경망의 출력(신경망이 추정한 값), $t_k$는 정답 레이블, k는 데이터의 차원 수를 나타낸다. “손글씨 숫자 인식” 예에서 $y_k$와$t_k$는 다음과 같은 원소 10개짜리 데이터이다.

```python
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
```

이 배열들의 원소는 첫 번째 인덱스부터 순서대로 숫자 ‘0’, ’1’, ‘2’, …일 때의 값이다. 여기에서 신경망 출력 y는 소프트맥스 함수의 출력이다. t와 같이 한 원소만 1로 하고 그 외는 0으로 나타내는 표기법을 **원-핫 인코딩**이라 한다.

오차제곱합은 각 원소의 출력(추정 값)과 정답 레이블(참 값)의 차($y_k - t_k)$를 제곱한 후, 그 총합을 구한다. 

```python
def sum_squares_error(y, t):
	return 0.5 * np.sum((y-t)**2)
```

오차제곱합을 파이썬으로 구현해봤다.

### 교차 엔트로피 오차

또 다른 손실 함수로서 **교차 엔트로피 오차(cross entropy error, CEE)**도 자주 이용한다. 교차 엔트로피 수식은 다음과 같다.

$$
E = -\sum_kt_klogy_k
$$

여기에서 log는 밑이 e인 자연로그이다. $y_k$는 신경망의 출력, $t_k$는 정답 레이블이다. 또 $t_k$는 정답에 해당하는 인덱스의 원소만 1이고 나머지는 0이다(원-핫 인코딩). 그래서 위의 식은 실질적으로 정답일 때의 추정($t_k$가 1일 때의 $y_k$)의 자연로그를 계산하는 식이 된다.

![](https://velog.velcdn.com/images/acadias12/post/38b9c1b3-29df-41c3-8aeb-f3ba31d59281/image.png)


위의 그림에서 볼 수 있듯이 x가 1일 때 y는 0이 되고 x가 0에 가까워질수록 y값은 점점 작아진다.

```python
def cross_entropy_error(y, t):
	delta = 1e-7
	return -np.sum(t * np.log(y + delta))
```

코드 마지막을 보면 np.log를 계산할 때 아주 작은 값인 delta를 더했다. 이는 np.log() 함수에 0을 입력하면 마이너스 무한대를 뜻하는 -inf가 되어 더 이상 계산을 진행할 수 없게 되기 때문이다.

## 미니배치 학습

기계학습 문제는 훈련 데이터를 사용해 학습한다. 더 구체적으로 말하면, 훈련 데이터에 대한 손실 함수의 값을 구하고, 그 값을 최대한 줄여주는 매개변수를 찾아 낸다. 이렇게 하려면 모든 훈련 데이터를 대상으로 손실 함수를 구해야한다.

훈련 데이터 모두에 대한 손실 함수의 합을 구하는 방법을 생각해보자. 예를 들어 교차 엔트로피 오차는 밑의 식처럼 된다.

$$
E = -\frac{1}{N}\sum_n\sum_kt_{nk}logy_{nk}
$$

이때 데이터가 N개라면 $t_{nk}$는 n번째 데이터의 k번째 값을 의미한다. 마지막에 N으로 나눔으로써 ‘평균 손실 함수’를 구하는 것이다. 이렇게 평균을 구해 사용하면 훈련 데이터 개수와 관계없이 언제든 통일된 지표를 얻을 수 있다.

---

수 많은 데이터를 대상으로 일일이 손실 함수를 계산하는 것은 현실적이지 않다. 이런 경우 데이터 일부를 추려 전체의 ‘근사치’로 이용할 수 있다. 신경망 학습에서도 훈련 데이터로부터 일부만 골라 학습을 수행한다. 이 일부를 **미니배치(mini-batch)**라고 한다. 가령 60,000장의 훈련 데이터 중에서 100장을 무작위로 뽑아 그 100장만 사용하여 학습하는 것이다. 이러한 학습 방법을 **미니배치 학습**이라고 한다.

```python
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```

np.random.choice()함수를 통해 훈련 데이터에서 무작위로 10장만 빼낼 수 있다.

## 왜 손실 함수를 설정하는가?

‘정확도’라는 지표를 놔두고 ‘손실 함수의 값’이라는 우회적인 방법을 택하는 이유는 뭘까? 이 의문은 신경망 학습에서의 ‘미분’의 역할에 주목한다면 해결된다. (미분 part는 생략…)

신경망 학습에서는 최적의 매개변수 (가중치와 편향)를 탐색할 때 손실 함수의 값을 가능한 한 작게 하는 매개변수를 찾는다. 이때 매개변수의 미분( 정확히는 기울기)을 계산하고, 그 미분 값을 단서로 매개변수 값을 서서히 갱신하는 과정을 반복한다.

가중치 매개변수의 손실 함수의 미분이란 ‘가중치 매개변수의 값을 아주 조금 변화 시켰을 때, 손실 함수가 어떻게 변하나’라는 의미이다. 만약 이 미분 값이 음수면 그 가중치 매개변수를 양의 방향으로 변화시켜 손실 함수의 값을 줄일 수 있다. 반대로, 미분 값이 양수면 가중치 매개변수를 음의 방향으로 변화시켜 손실 함수의 값을 줄일 수 있다. 그러나 미분 값이 0이면 가중치 매개변수를 어느 쪽으로 움직여도 손실 함수의 값은 줄어들지 않는다. 그래서 가중치 매개변수의 갱신은 거기서 멈춘다.

**정확도를 지표를 삼아서는 안 되는 이유는 미분 값이 대부분의 장소에서 0이 되어 매개변수를 갱신할 수 없기 때문이다.**

### 손실 함수를 지표로 삼는다면

정확도를 지표로 삼으면 연속적인 변화보다 불연속적인 띄엄띄엄한 값으로 바뀌어버린다. 반면, 손실 함수를 지표로 삼았다면 매개변수의 값이 조금 변하면 그에 반응하여 손실 함수의 값도 연속적으로 변화한다.

![](https://velog.velcdn.com/images/acadias12/post/ad238946-3f99-4974-8963-2fcf8cacc06e/image.png)


계단 함수와 시그모이드 함수를 비교해보면, 계단 함수는 한순간만 변화를 일으키지만, 시그모이드 함수의 미분(접선)은 위의 그림과 같이 출력(세로축의 값)이 연속적으로 변하고 곡선의 기울기도 연속적으로 변한다. 즉, 시그모이드 함수의 미분은 어느 장소라도 0이 되지는 않는다. 이는 신경망 학습에서 중요한 성질로, 기울기가 0이 되지 않는 덕분에 신경망이 올바르게 학습할 수 있는 것이다.

## 학습 알고리즘 구현하기

신경망 학습의 절차는 다음과 같다

- **전제**
    - 신경망에서는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 ‘학습’이라 한다. 신경망 학습은 다음과 같이 4단계로 수행한다.
- **1단계 - 미니배치**
    - 훈련 데이터 중 일부를 무작위로 가져온다. 이렇게 선별한 데이터를 미니배치라 하며, 그 미니배치의 손실 함수 값을 줄이는 것이 목표이다.
- **2단계 - 기울기 산출**
    - 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구한다. 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시한다.
- **3단계 - 매개변수 갱신**
    - 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.
- **4단계 - 반복**
    - 1~3단계를 반복한다.

이것이 신경망 학습이 이뤄지는 순서이다. 이는 경사 하강법으로 매개변수를 갱신하는 방법이며, 이때 데이터를 미니배치로 무작위로 선정하기 때문에 **확률적 경사 하강법(stochastic gradient descent, SGD)**이라고 부른다.

### 2층 신경망 클래스 구현하기

이 클래스의 이름은 TwoLayerNet이다.소스 코드는 ch04/two_layer_net.py에 있다.

```python
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		#가중치 초기화
		self.params = {}
		self.params['W1'] = weight_init_std * \
												np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * \
												np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)
		
	def predict(self, x):
		W1, W2 = self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'], self.params['b2']
		
		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)
		
		return y
		
	# x : 입력 데이터, t : 정답 레이블
	def loss(self, x, t):
		y = self.predict(x)
		
		return cross_entropy_error(y, t)
		
	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)
		
		accuracy = np.sum(y == t) / float(x.shape[0])
		return accuracy
		
	# x : 입력 데이터, t : 정답 레이블
	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t)
		
		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
		
		return grads
```

### 중요한 변수 및 메서드 정리

TwoLayerNet 클래스가 사용하는 변수

- params
    - 신경망의 매개변수를 보관하는 딕셔너리 변수(인스턴스 변수)
    - params[’W1’]은 1번째 층의 가중치, params[’b1’]은 1번째 층의 편향
    - params[’W2’]은 2번째 층의 가중치, params[’b2’]은 2번째 층의 편향
- grads
    - 기울기 보관하는 딕셔너리 변수(numerical_gradient() 메서드의 반환 값)
    - grads[’W1’]은 1번째 층의 가중치의 기울기, grads[’b1’]은 1번째 층의 편향의 기울기
    - grads[’W2’]은 2번째 층의 가중치의 기울기, grads[’b2’]은 2번째 층의 편향의 기울기

TwoLayerNet 클래스의 메서드

- __init__(self, input_size, hidden_size, output_size)
    - 초기화를 수행한다. 인수는 순서대로 입력층의 뉴런 수, 은닉층의 뉴런 수, 출력층의 뉴런 수
- predict(self, x)
    - 예측(추론)을 수행한다. 인수 x는 이미지 데이터
- loss(self,x,t)
    - 손실 함수의 값을 구한다. 인수 x는 이미지 데이터, t는 정답 레이블
- accuracy(self,x,t)
    - 정확도를 구한다.
- numerical_gradient(self,x,t)
    - 가중치 매개변수의 기울기를 구한다.
- gradient(self,x,t)
    - 가중치 매개변수의 기울기를 구한다.

## 미니배치 학습 구현하기

```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

```

여기에서는 미니배치 크기를 100으로 했다. 즉, 매번 60,000개의 훈련 데이터에서 임의로 100개의 데이터 (이미지 데이터와 정답 레이블 데이터)를 추려낸다. 그리고 그 100개의 미니배치를 대상으로 확률적 경사 하강법을 수행해 매개변수를 갱신한다. 경사법에 의한 갱신 횟수(반복 횟수)를 10,000번으로 설정하고, 갱신할 때마다 훈련 데이터에 대한 손실 함수를 계산하고, 그 값을 배열에 추가한다.

## 시험 데이터로 평가하기

**에폭(epoch)**은 하나의 단위이다. 1에폭은 학습에서 훈련 데이터를 모두 소진했을 때의 횟수에 해당한다. 예를 들어, 훈련 데이터 10,000개를 100개의 미니배치로 학습할 경우, 확률적 경사 하강법을 100회 반복하면 모든 훈련 데이터를 ‘소진’한게 된다. 이 경우 100회가 1에폭이 된다.

신경망 학습의 원래 목표는 범용적인 능력을 익히는 것이다. 범용 능력을 평가하려면 훈련 데이터에 포함되지 않은 데이터를 사용해 평가해봐야 한다. 이를 위해 다음 구현에서는 학습 도중 정기적으로 훈련 데이터와 시험 데이터를 대상으로 정확도를 기록한다. 여기에서는 1에폭별로 훈련 데이터와 시험 데이터에 대한 정확도를 기록한다.

```python
    train_acc_list = []
		test_acc_list = []
		
		# 1에폭당 반복 수
		iter_per_epoch = max(train_size / batch_size, 1)
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
```

위의 코드에서 평가하는 부분만 가져온 것이다.

![](https://velog.velcdn.com/images/acadias12/post/8718a6e3-0188-4074-b549-25f4f84fe6f6/image.png)



위의 그림에서는 훈련 데이터에 대한 정확도를 실선으로, 시험 데이터에 대한 정확도를 점선으로 그렸다. 보다시피 에폭이 진행될수록(학습이 진행될수록) 훈련 데이터와 시험 데이터를 사용하고 평가한 정확도가 모두 좋아지고 있다. 또, 두 정확도에는 차이가 없음을 알 수 있다. 다시 말해 이번 학습에서는 오버피팅이 일어나지 않았다.

## 정리

- 기계학습에서 사용하는 데이터셋은 훈련 데이터와 시험 데이터로 나눠 사용한다.
- 훈련 데이터로 학습한 모델의 범용 능력을 시험 데이터로 평가한다.
- 신경망 학습은 손실 함수를 지표로, 손실 함수의 값이 작아지는 방향으로 가중치 매개변수를 갱신한다.
- 가중치 매개변수를 갱신할 때는 가중치 매개변수의 기울기를 이용하고, 기울어진 방향으로 가중치의 값을 갱신하는 작업을 반복한다.
- 아주 작은 값을 주었을 때의 차분으로 미분하는 것을 수치 미분이라고 한다.
- 수치 미분을 이용해 가중치 매개변수의 기울기를 구할 수 있다.
- 수치 미분을 이용한 계산에는 시간이 걸리지만, 그 구현은 간단하다. 한편, 다음 장에서 구현하는 (다소 복잡한) 오차역전파법은 기울기를 고속으로 구할 수 있다.