---
title: "[논문 리뷰] RobustSAM: Segment Anything Robustly on Degraded Image"
date: 2025-07-22 16:30:00 +0900
categories: [AI,PaperReview]
tags: [AI,PaperReview]
math: true
---


## Abstract

Segment Anything Model(SAM)은 image segmentation에서 혁신적인 접근 방식으로 등장했으며, zero-shot segmentation과 유연한 prompting system으로 호평받고있다.

그럼에도 불구하고 낮은 품질의 이미지에 대해 성능 저하에 대한 challenge가 존재한다. 이 문제를 해결하기 위해 낮은 품질을 가진 이미지에서도 좋은 성능을 보여주는 RobustSAM을 소개한다.

- RobustSAM은 pre-trained SAM 모델을 활용하며 이에 해당하는 파라미터 증가와 요구되는 계산은 적은편이다.
- RobustSAM의 추가적인 파라미터들은 8개의 GPU로 30시간 정도 최적화하면 된다. 이는 연구 실험에서 실행가능성과 실용성을 설명한다.
- 저자들은 또한 688K의 different degradations 이미지-마스크 쌍의 RobustSAM을 훈련하고 평가한 RobustSeg dataset을 소개한다.
- 광범위한 RobustSAM의 실험 결과는 under zero-shot conditions에서 우수한 성능을 보여준다.
- RobustSAM의 방법은 single image dehazing and deblurring과 같은 SAM-based downstream tasks의 개선된 성능을 보여준다.

**Downstream Task란?**

사전학습(pretraining)된 모델을 활용해서 해결하고자 하는 실제 목적의 과제(task)를 의미.

- 예를 들면 SAM은 다양한 Segmentation을 위한 Foundation 모델인데 이를 활용한 Downstream task는 의학 영상 분할, 위성 이미지 분석 등의 실제 할 일.

### Introduction(짧게 요약)

- 저자들은 SAM 모델을 기반으로 다양한 degradation 이미지에 대해 성능 개선을 한 RobustSAM을 소개하고있음.
- 저자들은 Robust-Seg dataset을 만들었음. 이 데이터들은 총 688K의 이미지와 마스크 쌍이고 각각 다른 degradation을 가짐.

## Method

RobustSAM은 SAM의 핵심 장점인 zero-shot segmentation 능력은 유지하면서, 이미지 품질이 떨어졌을 때도 잘 동작하도록 개선한 모델임.

기존처럼 fin-tune하거나 복잡한 구조를 적용하는 것 없이, 간결하고 의도적인 향상을 적용함.

![](https://velog.velcdn.com/images/acadias12/post/2e89f38a-a0ab-47c1-bcaa-329ff64af8b6/image.png)


RobustSAM의 key contribution은 아래 두 가지 모듈임. 아래 두 모듈은 degradation-invariant에 강건한 정보를 추출하며, 이 정보들은 원래 SAM이 깨끗한 이미지에서 추출한 정보와 잘 정렬되도록 설계되어 있음.

- **AntiDegradation Output Token Generation (AOTG)**
- **AntiDegradation Mask Feature Generation (AMFG)**

---

이것은  15 types of synthetic degradation 증강을 통해 clear-degraded image pairs을 생성한다.

다른 loss들은 clear와 degraded feature, ground truth와 예측된 segmentation 사이의 일관성을 강화하기 위해 적용된다.

비록 RobustSAM은 합성된 degradation를 사용해 학습했지만, RobustSAM은 실제 사진에서도 잘 동작한다.

**Training**

- RobustSAM을 훈련하기위해, clear input image에 degradation augmentation을 적용하고 그 적용된 degraded image를 RobustSAM에 넣는다.
- 처음에, 모델은 degraded 이미지로부터 feature를 추출하기 위해 Image Encoder를 활용한다.
- SAM framewrok와 달리, RobustSAM은 output token을 finetune한다 → **Robust Output Token(ROT)**라 부름.
- 이 ROT는 prompt 토큰과 Image Encoder가 추출한 feature들과 함께 원래 SAM layer를 통해 처리되며, 그 결과 degrade 상태에서 mask feature $F_{MFD}$와 마스크별 Robust Output Token $T_{RO}$를 생성한다.
- **AOTG block**은 degradation에 강건한 정보를 추출하기 위해 $T_{RO}$을 처리하고 이 과정에서 $\hat{T}_{RO}$로 변함.
- 이와 동시에, **AMFG block**은 Image Encoder의 초기층과 마지막층에서 추출된 mask와 complementary feature( $F_{MFD}$와 $F_{CFD}$)를 정제하여, degradation와 관련된 정보를 제거하고, 정제정제된 feature($\hat{F}_{MFD}$와 $\hat{F}_{CFD}$)를 생성한다.
- Feature Fusion block은 이 정제된 feature들을 RobustSAM의 마지막 robust mask feature와 결합하여 segmentation quality를 개선한다.
- 병렬로, 원래 clear image는 clear version의 complementary feature $F_{CFC}$, mask feature $F_{MFC}$ 그리고 output token $T_{OC}$를 추출하기 위해 SAM에 의해 처리된다.
- clear feature들과 RobustSAM이 생성한 정제된 feature들 사이의 일관된 loss들은 undegraded image output과 정렬을 보장한다. →(이때 정렬은 비슷하다 라고 해석하면 됌.)
- RobustSAM은 degraded된 입력에 대해 나온 segmentation 결과를 정답과 비교하여 segmentation loss로 학습하고, 학습 시에는 15가지 degradations과 함께 원본 이미지를 그대로 사용한는 identity mapping도 포함하여 깨끗한 이미지에 대해서도 성능 저하 없이 잘 동작하도록 설계되었음.

**inference**

- 추론동안, RobustSAM은 segmentation mask를 생성하도록 사용되었음.

**Anti-Degradation Mask Feature Generation**

![](https://velog.velcdn.com/images/acadias12/post/62b01699-3bf7-4d97-a1fe-52bc3d40a637/image.png)


- 그림에 나온 것 처럼, input feature들은 Instance Normalization(IN)에 의해 처음 처리됌.
- 이전 연구들에 영감을 받아, IN의 목적은  image degradation와 관련된 변화를 표준화 하는 것임.
- 직관적으로, 이것은 style 속성을 제거하면서 핵심 내용은 유지한다. → style은 degradation이라 생각
- 이 단계는 이미지 각각의 distortions의 영향을 완화하기 위해 필수적이고, content의 안정성을 보장한다.
- 이것과 병렬로, RobustSAM은 Batch Normalization(BN)을 적용한다. BN은 IN과정에서 발생할 수 있는 세부 정보 소실 문제를 해결하주기 때문에 매우 중요하다.
- RobustSAM은 이 BT와 IN에 의해 각각 생성된 feature들을 합친다.
- Attention mechanism은 attention map을 생성하기 위해 이 합쳐진 feature들을 계산한다. 이 attention map은 각 feature 종류의 중요도를 동적으로 가중치로 반영하며, 그 결과, BN과 IN의 장점을 모두 통합한 feature set을 생성한다.
- 손실될 수 있는 semantic 정보를 보충하기위해, 이 feature set은 channel dimension에 속한 original input features와 합쳐진다.
- 추가적으로, RobustSAM은 channel attention을 통합하는데 이는 squeeze-and-excitation 방식과 유사하며, feature 통합을 적응적으로 정제하는데 사용됨.
- 저자들은 통합된 feature들을 향상시키기 위해 **Fourier Degradation Suppression module**을 도입했음. 이 모듈은 Fourier 변환을 이용하여, feature들을 공간 도메인에서 주파수 도메인으로 변환함으로써 동작함.
- 이 기법은 주파수의 진폭 성분을 활용하여, 이미지 degradation와 관련된 style 정보를 포착하고, 1x1 convolutionㅇㄹ 적용해 degradation를 분리하고 제거하는데 집중함. 한편, phase 성분은 그대로 유지하여 이미지의 구조적인 일관성을 보존함.
- inverse Fourier transform은 정제된 feature들을 공간적 도메인으로 가져온다. 이 과정은 degradations된 image style로 간주하고, degradation-invariant feature를 생성하여 강건한 segmentation을 가능하게 한다.
- 이 모듈은 이전 모듈들이 생성한 두 feature, $F_{CFD}$와 $F_{MFD}$에 적용된다.
- 이 정제된 feature들과 깨끗한 이미지를 입력했을 때 SAM에 의해 추출된 feature들의 일관성을 유지하도록 보장하기 위해 **Mask Feature Consistency Loss ( $L_{MFC}$)**를 사용함.

$$
L_{MFC} = || \hat{F}_{CFD}-F_{CFC}||_2 + ||\hat{F}_{MFD}-F_{MFC}||_2
$$

- 각 부분의 $L_{MFC}$를 최소화함으로써, RobustSAM는 정제된 feature들이 깨끗한 이미지 조건에서 추출된 feature들과 일관성을 유지하도록 보장한다.
- 이를 통해, 다양한 degradation 상황에서도 feature들의 강건성과 일관성이 보장된다.

**Anti-Degradation Output Token Generation**

![](https://velog.velcdn.com/images/acadias12/post/f85fb1cd-bf6f-4718-b529-b73a704467d5/image.png)


- Anti-Degradation Output Token Generation module은 degradation와 관련된 정보를 제거하기 위해 마스크 당 Robust Output Token( $T_{RO}$)를 정제하기위해 노력한다.
- 전통적인 mask feature들과 달리, $T_{RO}$는 텍스처 정보가 덜 포함된 classification boundaries의 명확성을 보장하기 위해 기능한다.
- 그러므로, 우리는 degradation에 민감한 정보를 걸러내기 위해 lightweight 모듈만 사용하는 것으로도 충분하다는 것을 발견했다.
- 그림의 오른쪽에 나타난 것 처럼, 이 모듈을 여러 층의 IN을 적용한 뒤, 하나의 MLP층을 이어서 사용한다.
- 이 전략은 model이 degradation된 입력으로 부터 강건한 mask 정보를 되찾는 것을 보장하면서, 계산 효율성을 유지하기 위함을 목표로 한다.
- 정제된 token $\hat{T}_{RO}$는 깨끗한 입력 이미지를 사용했을 때 원래 SAM이 생성한 Output token $T_{OC}$와 비교되어, Token Consistency Loss $L_{TC}$를 계산하는데 사용된다.

$$
L_{TC}=||\hat{T}_{RO}-T_{OC}||_2
$$

- 이 Loss는 clear image 조건에서 추출된 정제된 token이 일관되게 유지하도록 보장한다.
- MLP를 통해 처리된 후, output은 마지막 mask를 생성하기 위해 Robust Mask Feature와 결합된다.

---

### Overall Loss

- overall loss function은 Mask Feature Consistency Loss ( $L_{MFC}$), Token Consistency Loss ($L_{TC}$) 그리고 Segmentation Loss ($L_{Seg}$)을 통합하여, 모델 학습을 위한 종합적인 Loss를 구성한다.

$$
L_{Overall} = L_{MFC} + \lambda_1L_{TC} + \lambda_2L_{seg}
$$

- $L_{seg}$는 Dice와 Focal loss를 통합한 segmentation loss이다.

$$
L_{seg} = L_{dice}(P,G) + \lambda_3L_{Focal}(P,G)
$$

- P는 예측된 mask이고, G는 ground truth mask이고, $\lambda_1 - \lambda_3$은 Loss 항에 가중치를 부여하기 위한 hyperparameter이다.
- 이 합쳐진 loss function은 모델의 degradation에 대한 강건성을 강화시키면서 segmentation 품질을 향상시킨다.

### Implementation Details

대략 688,000 image 장의 Robust-Seg dataset을 사용하여 훈련 및 평가를 진행.

![](https://velog.velcdn.com/images/acadias12/post/b94bf0d3-2349-4fd7-af27-e027187c40d0/image.png)

![](https://velog.velcdn.com/images/acadias12/post/a2d47ccf-52d6-42db-91ce-35e1a47d9a62/image.png)

![](https://velog.velcdn.com/images/acadias12/post/2aec6566-616e-4242-95b4-9ceeafbd6ca6/image.png)
