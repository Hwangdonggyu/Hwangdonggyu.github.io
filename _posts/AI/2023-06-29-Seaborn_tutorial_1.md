---
title: AI 시각화 연습_1
date: 2023-06-30 00:01:00 +0900
categories: [AI]
tags: [AI]
---

![](https://velog.velcdn.com/images/acadias12/post/596daee7-7a2a-41b1-a038-da5f749d38a3/image.jpeg)

오늘은 AI를 공부해보면서 Seaborn을 이용한 시각화 연습한 것 중 Relational 그래프를 다뤄보려한다.

## Seaborn이란?
> Seaborn은 파이썬 데이터 시각화 라이브러리이고, Matplotlib을 기반으로 한 통계 그래픽스 패키지이다.

## Relational 그래프
> #### 두 가지 변수의 관계를 나타내는 그래프이다.
- scatterplot : 산점도
- lineplot : 라인
- relplot : scatterplot와 lineplot을 합친 그래프


## 모듈 및 데이터 셋

### 모듈
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### 데이터 셋


```python
#tips 데이터
tips = sns.load_dataset("tips")
tips
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>239</th>
      <td>29.03</td>
      <td>5.92</td>
      <td>Male</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>240</th>
      <td>27.18</td>
      <td>2.00</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>241</th>
      <td>22.67</td>
      <td>2.00</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>242</th>
      <td>17.82</td>
      <td>1.75</td>
      <td>Male</td>
      <td>No</td>
      <td>Sat</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>243</th>
      <td>18.78</td>
      <td>3.00</td>
      <td>Female</td>
      <td>No</td>
      <td>Thur</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>244 rows × 7 columns</p>
</div>


```python
# fmri 데이터 셋
fmri = sns.load_dataset("fmri")
fmri
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>timepoint</th>
      <th>event</th>
      <th>region</th>
      <th>signal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s13</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.017552</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s5</td>
      <td>14</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.080883</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s12</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.081033</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s11</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.046134</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s10</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.037970</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1059</th>
      <td>s0</td>
      <td>8</td>
      <td>cue</td>
      <td>frontal</td>
      <td>0.018165</td>
    </tr>
    <tr>
      <th>1060</th>
      <td>s13</td>
      <td>7</td>
      <td>cue</td>
      <td>frontal</td>
      <td>-0.029130</td>
    </tr>
    <tr>
      <th>1061</th>
      <td>s12</td>
      <td>7</td>
      <td>cue</td>
      <td>frontal</td>
      <td>-0.004939</td>
    </tr>
    <tr>
      <th>1062</th>
      <td>s11</td>
      <td>7</td>
      <td>cue</td>
      <td>frontal</td>
      <td>-0.025367</td>
    </tr>
    <tr>
      <th>1063</th>
      <td>s0</td>
      <td>0</td>
      <td>cue</td>
      <td>parietal</td>
      <td>-0.006899</td>
    </tr>
  </tbody>
</table>
<p>1064 rows × 5 columns</p>
</div>


## Relational 그래프 정리

### scatterplot

> scatterplot은 산점도를 나타내는 그래프이다.

```python
#scatterplot : 산점도 scatter(x, y, data)
sns.scatterplot(x='total_bill', y='tip', data=tips)
```

```
<Axes: xlabel='total_bill', ylabel='tip'>
```

![](https://velog.velcdn.com/images/acadias12/post/636bfc74-cede-49f9-980a-450826dd75dd/image.png)


```python
#hue : 의미에 따라 점의 색을 변경
#style: 모양 구분
sns.scatterplot(x='total_bill', y= 'tip', data = tips, hue = 'day', style = 'time')
```

```
<Axes: xlabel='total_bill', ylabel='tip'>
```

![](https://velog.velcdn.com/images/acadias12/post/c787daaa-65d6-4a9c-8807-a6ba43bc42c5/image.png)

### lineplot

> lineplot은 line을 나타내는 그래프이며, 데이터가 연속적일 경우 주로 사용한다. lineplot(x,y,data)

```python
sns.lineplot(x='timepoint', y='signal', data=fmri)
```

```
<Axes: xlabel='timepoint', ylabel='signal'>
```

![](https://velog.velcdn.com/images/acadias12/post/30812429-4f7c-45ad-b69f-a2fac5441741/image.png)

```python
# 위의 그래프에서 색이 칠해져 있는 부분은 신뢰구간(confidene interval)로 ci 파라미터로 조절 가능
# hue와 style 옵션을 사용할 수 있음

sns.lineplot(x='timepoint',y='signal',data=fmri,hue='event',style='event',ci=None)
```

```
<Axes: xlabel='timepoint', ylabel='signal'>
```

![](https://velog.velcdn.com/images/acadias12/post/4e2a937a-23bc-4c9e-8f60-d28a88c9990a/image.png)


### relplot

> relplot은 scatterplot과 lineplot을 합쳐 놓은 그래프이다.
- kind 파라미터에 scatter나 line으로 형식 선택 가능 (default=scatter)
- scatterplot과 lineplot은 AxeSubplot을 반환하지만, relplot은 FaceGrid를 반환
- FaceGrid를 반환하는 경우 여러 그래프를 한 번에 그릴수 있음
- hue와 style 옵션 모두 사용 가능


```python

#scatter
sns.relplot(x='total_bill',y='tip',kind='scatter',hue ='time',data=tips)

```

```
<seaborn.axisgrid.FacetGrid at 0x164067f70>
```

![](https://velog.velcdn.com/images/acadias12/post/087d0f04-02b1-443b-b374-6dab887723c8/image.png)

```python
#line
sns.relplot(x='timepoint',y='signal',kind='line',hue ='event', style ='event',ci = None,data=fmri)
```

```
<seaborn.axisgrid.FacetGrid at 0x16337f880>
```

![](https://velog.velcdn.com/images/acadias12/post/15461c69-cd41-4d6b-9f25-330377c7b7f9/image.png)


## 느낀점
AI를 입문한지 얼마 되지 않아 Seaborn과 pandas를 다루는게 미숙하고 어렵지만 계속 공부하고 사용해보면서 익숙해져봐야겠다. Seaborn을 통해 데이터 셋을 한 후 Relational 그래프를 통해 시각화 한 것을 보니 좀 신기하고 흥미로웠다. 더 열심히 공부해봐야겠다.





