---
title: 오차 역전파, 활성화 함수, 손실 함수, 옵티마이저
date: 2024-02-05 11:04:00 +0900
categories: [AI,DL]
tags: [AI,DL]
math: true
---

![](https://velog.velcdn.com/images/acadias12/post/7d5e1cee-a1c0-42ec-9a50-d0b00b5b9647/image.png)
## 심층 신경망(다중 퍼셉트론) 개요

![](https://velog.velcdn.com/images/acadias12/post/7e4cce80-19c2-4205-88db-993a4736c2f3/image.png)

### 심층 신경망(다중 퍼셉트론) 이해

기본 퍼셉트론은 간단한 문제만 해결이 가능. 보다 복잡한 문제의 해결을 위해서 **은닉층(Hidden Layer)**가 포함된 다중 퍼셉트론으로 심층 신경망을 구성

![](https://velog.velcdn.com/images/acadias12/post/15a7f1d5-20cf-422c-a855-1036510e809f/image.png)


### 심층 신경망(다중 퍼셉트론) 구조

![](https://velog.velcdn.com/images/acadias12/post/2e47363c-2990-4cc8-944c-996a85aa1d99/image.png)


위의 구조를 추상화할 수 있지만, 파라미터가 많아지고 식 자체가 굉장히 복잡해진다. 이것을 경사하강법으로 어떻게 구현할까? → 역전파(Backpropagation)

## 심층 신경망의 학습법 개요

1. Feed Forward 수행.
2. Backpropaagation을 수행하면서 Weight Update.
3. 1,2과정을 iteration 수행.

![](https://velog.velcdn.com/images/acadias12/post/e5f5a994-14d1-49e3-8ec9-9caa543b4117/image.png)


### Backpropagation(역전파) 개요

출력층부터 역순으로 Gradient를 전달하여 전체 Layer의 가중치를 Update하는 방식.

### 미분의 연쇄 법칙(Chain Rule) - 합성 함수의 미분

![](https://velog.velcdn.com/images/acadias12/post/56be1b28-7684-46e4-8670-6a8322c52163/image.png)


고등학교때 합성함수의 미분과 동일하다. → 합성 함수를 구성하는 **각 함수의 미분의 곱**으로 나타낼 수 있음.

![](https://velog.velcdn.com/images/acadias12/post/047244e3-7cc8-455d-a17e-eab50c2fee92/image.png)

미분의 연쇄 법칙을 이용한다면 복잡한 고차원 식을 보다 간편하게 미분할 수 있다. [미분의 연쇄 법칙은 양파를 까는 일과 같다](https://blog.naver.com/alwaysneoi/100171733834?viewType=pc) → 잘 설명된 블로그

### 퍼셉트론 회귀 손실 함수의 편미분

![](https://velog.velcdn.com/images/acadias12/post/608a4df9-48f2-4cf6-9a46-13d2db20df5f/image.png)


$Z =y_i(w_0+w_1*x_i)$로 치환하고 손실 함수 미분을 진행한다.

### 미분의 연쇄 법칙 - 의존 변수들의 순차적인 변화율

![](https://velog.velcdn.com/images/acadias12/post/cc53f31e-9902-4681-b882-b58a9e6322f4/image.png)


- Chain Rule은 변수가 여러 개일때 어떤 변수에 대한 다른 변수의 변화율을 알아내기 위해 사용됨
- 변수 y가 변수 u에 의존하고, 다시 변수 u가 변수 x에 의존한다고 하면 x에 대한 y의 변화율은 u에 대한 y의 변화율과 x에 대한 u의 변화율을 곱하여 계산할 수 있음.

### 미분의 연쇄 법칙 의의

아무리 개별 변수가 복잡하게 구성된 함수의 미분이라도 해당 함수가 (미분가능한) 내표 함수의 연속적인 결합으로 되어 있다면 연쇄 법칙으로 쉽게 미분 가능.

### 심층 신경망은 합성 함수의 연쇄 결합

![](https://velog.velcdn.com/images/acadias12/post/85750621-2fa4-4b6b-a63c-3537e24d8fa6/image.png)


- Input = X
- 첫번째 은닉층 출력 = $O_1 = F_1(W^1X)$
- 두번째 은닉층 출력 = $O_2=F_2(W^2O_1)$
- 최종 출력 Output = $F_3(W^3O_2)$
    
    = $F_3(W^3F_2(W^2O_1))$
    
    = $F_3(W^3F_2(W^2F1(W^1X)))$
    

### 간단한 신경망 Backpropagation

![](https://velog.velcdn.com/images/acadias12/post/d2797128-0767-444a-aa91-5dd678b684bc/image.png)


Input이 들어오게 되면 Loss함수를 계산하게 되고 이것들을 순차적으로 미분값을 구하면서 계산하는 방식

### Upstream Gradient와 Local Gradient

![](https://velog.velcdn.com/images/acadias12/post/f7d5b971-0107-438c-a723-e876b12eb12f/image.png)


Upstream Gradient에 Local Gradient를 곱해서 Gradient를 구할 수 있음.

### 여러 뉴런이 있는 신경망의 Backpropagation

![](https://velog.velcdn.com/images/acadias12/post/063c9803-d702-48ee-b15a-5ebe8b1a6b14/image.png)


$\theta_{21}$은 도착이 2번 출발이 1번이다. 즉, 번호 순서는 도착 → 출발 순이다.

![](https://velog.velcdn.com/images/acadias12/post/95ce2d60-fcc3-42c8-97f9-84ec3c8a9e7e/image.png)


푸른색, 오렌지색의 형광펜 친 부분의 연쇄 미분 과정이다. $\theta_{11}$이 푸른색 방향만 영향을 준 것이 아니라 오렌지색 방향도 영향을 줬기 때문에 같이 고려를 해야 한다. → 오렌지색 + 푸른색

![](https://velog.velcdn.com/images/acadias12/post/6ef72f3f-3283-4630-abdf-0404594e2d33/image.png)

## 활성화 함수(Activation Function)

![](https://velog.velcdn.com/images/acadias12/post/9a87a4fc-ea7d-4226-ab81-82085a596bde/image.png)


**파란색 선이 각 함수의 그래프이고 노란색 선이 함수를 미분한 그래프이다.**

z = 입력값이다.

**Pytorch에서의 활성화 함수 사용법**

import torch → 라이브러리 함수

- Sigmoid Function : torch.sigmoid(z)
- Hyperbolic Tangent(tanh): torch.tanh(z)
- Rectified Linear Unit(ReLU) : torch.relu(z)

### 활성화 함수(Activation Function)의 주요 목적

**활성화 함수의 주요 목적은 딥러닝 네트웤에 비선형성을 적용하기 위함**

![](https://velog.velcdn.com/images/acadias12/post/850d0032-f0fb-4727-8c41-aafab5598ca5/image.png)


선형 활성화 함수는 선형 레벨의 판별 기준을 제공하고 비선형 활성화 함수는 복잡한 함수를 근사할 수 있도록 만들어줌.

### 활성화 함수(Activation Function) 함수의 적용

![](https://velog.velcdn.com/images/acadias12/post/6efd5c7f-45a1-41e7-a91f-c9b49123604b/image.png)


- **ReLU** : 은닉층에 사용됨
- **Sigmoid** : 이진 분류(Binary Classfication)시 마지막 Classfication 출력층에 사용
- **Softmax** : 멀티 분류 시(Multi Classification)시 마지막 Classfication 출력층에 사용

### Sigmoid 함수 특성

시그모이드는 은닉층(Hidden Layer)의 활성화 함수로는 Gradient Vanishing 등의 이슈로 더 이상 사용되지 않음. 이진 분류의 최종 Classification Layer의 Activation 함수로 주로 사용.

![](https://velog.velcdn.com/images/acadias12/post/0880186f-ed1d-4442-8c4c-304a1fc6233b/image.png)


입력값이 양이나 음으로 크게 커지거나 작아지면 출력값의 변화가 거의 없고 미분 값이 0이 가까워짐 → Gradient Vanishing 문제가 발생

### Gradient Vanishing 문제

![](https://velog.velcdn.com/images/acadias12/post/e27e152c-071e-4e7e-8a9e-fe5ba4f02de3/image.png)

미분 값이 0이 되어버려 w(가중치)가 업데이트가 안되는 문제가 생김.

### 회귀를 이용한 분류에서 사용되는 Sigmoid

![](https://velog.velcdn.com/images/acadias12/post/ec621b26-19f1-4435-870e-e16a7d53c1ee/image.png)


0 또는 1 값을 반환하는 시그모이드의 특성으로 인해 이진 분류의 확률 값을 기반으로 최종 분류 예측을 적용하는데 Sigmoid Function이 사용된다.

### Hyperbolic Tangent(tanh)

![](https://velog.velcdn.com/images/acadias12/post/2a7cfab3-7712-4786-9388-f109066c851f/image.png)



Tanh는 Sigmoid와 달리 -1과 1 사이의 값을 출력하여 평균이 0이 될 수 있지만 여전히 입력값이 양(또는 음)으로 큰 값이 입력될 경우에는 출력값의 변화가 미비함.

### ReLU(Rectified Linear Unit)

![](https://velog.velcdn.com/images/acadias12/post/dc2bf223-f4db-4ec2-819e-145f8d9a02cc/image.png)


대표적인 은닉층의 활성함수(Activation Function).

입력값이 0보다 작을 때 출력은 0, 0보다 크면 입력값을 출력( output = 0 if input ≤ 0, output = input if input > 0)

다양한 유형의 변형이 존재(Leaky ReLU, ELU 등)

### 소프트맥스 함수(Softmax activation)

![](https://velog.velcdn.com/images/acadias12/post/91050b11-b33f-462e-92cb-ce093a5175b9/image.png)


Sigmoid와 유사하게 Score값을 확률값 0~1로 변환 하지만, 차이점은 소프트맥스 개별 출력값의 총 합이 1이 되도록 매핑해 주는 것임.

여러개의 타겟값을 분류하는 Multi Classification의 최종 활성화 함수로 사용됨.

### Tensorflow Playground에서 실습 해보기

[Tensorflow — Neural Network Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.74637&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

## 손실/비용/목적 함수

손실 함수 = 비용 함수 = 목적함수

loss function = cost function = objective function

### 손실 함수(Loss Function) 개요

![](https://velog.velcdn.com/images/acadias12/post/d31f7bde-7e3d-460a-b48b-7a9edf218ad8/image.png)


네트워크 모델이 얼마나 학습 데이터에 잘 맞고 있는지, 우리가 학습을 잘 하고 있는지 알려주는 함수.

### 손실 함수(Loss Function)의 역할

![](https://velog.velcdn.com/images/acadias12/post/7132b4a0-7aa7-4284-bbd9-5e17c772252d/image.png)


손실 함수는 학습 과정이 올바르게 이뤄질 수 있도록 적절한 가이드를 제공할 수 있어야한다.

**회귀는 주로 MSE(Mean Squared Error), 분류는 주로 Cross Entropy(binary, categorical)를 이용한다.**

### Softmax/Sigmoid와 Cross Entropy Loss

![](https://velog.velcdn.com/images/acadias12/post/352aa023-9323-4b82-8e69-d0a096ba0368/image.png)


Multi Class일 경우 Categorical CE Loss 또는 Softmax CE Loss

![](https://velog.velcdn.com/images/acadias12/post/fc8d8f81-c026-4870-ae33-91a3236fb278/image.png)


Binary Class일 경우 Binary CE Loss

![](https://velog.velcdn.com/images/acadias12/post/0da4193f-c44e-4cab-950d-5d000d1c6dba/image.png)


### Cross Entropy 개요(Softmax의 경우)

![](https://velog.velcdn.com/images/acadias12/post/ac43a71f-62d2-4bd8-8033-aa005dc9f89b/image.png)


-를 붙여주는 이유는 log 0.1은 -1이되기 때문에 -를 붙여준다

### Cross Entropy 특성

실제 클래스 값에 해당하는 Softmax의 결과 값에만 Loss를 부여함.

아주 잘못된 예측 결과에는 매우 높은 Loss 부여

![](https://velog.velcdn.com/images/acadias12/post/097d67fd-2f12-4186-ae88-0a1a35e7b80e/image.png)


$Cross\_Entropy=-log(2번째예측값)= -log(0.9) = 0.105$

### Cross Entropy를 통한 Loss

![](https://velog.velcdn.com/images/acadias12/post/895d2a35-6213-4eb5-881e-f6619649a6db/image.png)


첫번째 예시에서는 Sample 1에서 3번째 값이 제일 크지만 1번째 값으로 잘못 예측함. 나머지는 다 맞게 예측 → 예측 정확도 66.6%

두번째 예시에서는 첫번째와 마찬가지로 예측 정확도는 66.6% CE는 다름

### Cross Entropy와 Squared Error 비교

Squared Error 기반은 일반적으로 잘못된 예측에 대해서 상대적으로 CE보다 높은 비율의 패널티가 부여되어 Loss값의 변화가 상대적으로 심함. 이때문에 CE에 비해 최적 수렴이 어려움. 또한 아주 잘못된 예측에 대해서는 CE보다 낮은 비율의 패널티가 부여됨.

![](https://velog.velcdn.com/images/acadias12/post/ac85d14d-149a-4501-9fa1-40a8dc21c091/image.png)


### Cross Entropy - Sigmoid일 경우

![](https://velog.velcdn.com/images/acadias12/post/ea5a5b28-9783-424c-bcf0-a1696cba16cc/image.png)


Sigmoid가 최종 출력층에 적용이 되는 경우에는 반드시 이진 분류여야 한다.

이진 분류(Binary Classification) → True or False ( 0 or 1)

- $-(y_i-log(\hat{y_i}))$이 부분은 Sigmoid가 적용된 값임. 이 부분은 실제값이 0이면 사라짐
- $(1-y_i) * log(1-\hat{y_i})$이 부분은 실제값이 1이면 사라짐

### Softmax/Sigmoid와 Cross Entropy Loss

![](https://velog.velcdn.com/images/acadias12/post/629713c0-1086-4281-a333-083b4fc19b81/image.png)


## Optimizer

![](https://velog.velcdn.com/images/acadias12/post/ce34feed-b647-4f69-b923-5851e3ebde69/image.png)


- 보다 최적으로 Gradient Descent를 적용
- 최소 Loss로 보다 빠르고 안정적으로 수렴 할 수 있는 기법 적용 필요.

### 주요 Optimizer들

Momentum → Gradient 조정

AdaGrad(Adaptive Gradient), RMSprop → learning rate(학습률 조정)

 ADAM(Adaptive Moment Estimation) → Gradient, learning rate 조정

### Momentum

가중치를 계속 Update 수행 시마다 이전의 Gradient들의 값을 일정 수준 반영 시키면서 신규 가중치로 Update 적용.

![](https://velog.velcdn.com/images/acadias12/post/f64ade94-5b3b-42fc-ad93-52e1de9b4054/image.png)


### Momentum 효과

![](https://velog.velcdn.com/images/acadias12/post/c5ffe774-1883-4abc-855c-54f732530cef/image.png)


SDG(Stochastic Gradient Descent)의 경우는 random한 데이터를 기반으로 Gradient를 계산하므로 최소점을 찾기 위한 최단 스텝 형태로 가중치가 Update되지 못하고 지그재그 형태의 Update가 발생하기 쉬움. Momentum을 통해서 이러한 지그재그 형태의 Update를 일정 수준 개선 가능.

### AdaGrad(Adaptive Gradient)

![](https://velog.velcdn.com/images/acadias12/post/6cb7ee7c-c067-48f0-9476-254ecab64be2/image.png)

가중치 별로 서로 다른 Learning Rate를 동적으로 적용. 그동안 적게 변화된 가중치는 보다 큰 Learning Rate를 적용하고, 많이 변화된 가중치는 보다 작은 Learning Rate를 적용

→ Gradient의 제곱은 언제나 양이기 때문에 iteration시마다 $s_t$의 값이 계속 증가하면서 Learning Rate 값이 아주 작게 변환되는 문제점이 발생

### RMSProp

![](https://velog.velcdn.com/images/acadias12/post/a1ffd36e-e802-4859-8043-8639d540fb49/image.png)

지나치게 Learning Rate가 작아지는 것을 막기 Gradient 제곱값을 단순히 더하지 않고 지수 가중 평균법(Exponentially Weighted Average)로 구함.

![](https://velog.velcdn.com/images/acadias12/post/4924b8cc-1460-44df-bb23-ae12e896b5ac/image.png)


지수 가중 평균 계수는 Learning Rate 변경에 적용될 현 시점 iteration 시의 Gradient 제곱값이 커지지 않고 동시에 오랜 과거의 Gradient 값의 영향을 줄일 수 있도록 설정. 보통은 0.9를 적용!

### Adam(Adaptive Momentum Estimation)

![](https://velog.velcdn.com/images/acadias12/post/07546c72-a9d7-4a75-a0d7-27bd8ed6eac5/image.png)


- RMSProp과 같이 개별 Weight에 서로 다른 Learning Rate(iteration을 반복할 수록 작아짐)을 적용함과 동시에 Gradient에 Momentum 효과를 부여
- Learning Rate를 적용하는 부분은 RMS와 유사. Weight에 변경된 Gradient를 적용하는 부분은 Momentum과 유사하나 Momentum 적용 시 과거 Gradient의 영향을 감소시키도록 지수 가중 평균을 적용.

### 학습률 최적화 유형

- Optimizer 방식: weight update 시에 Learning Rate을 동적 변경
- Learning Rate Scheduler 방식 : epochs 시마다 성능 평가 지표 등에 따라 동적으로 학습율을 변경하는 방식

출처 : 권철민 CNN 완벽 가이드