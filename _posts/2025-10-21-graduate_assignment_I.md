---
title: "공공통계 프로그래밍 II - 과제 1"
excerpt: "산점도를 이용한 데이터 시각화"
classes: wide

categories:
  - Statistics
---

# 공공통계 프로그래밍 II - 과제 1

## 라이브러리 불러오기
- 한글 세팅 추가


```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
```


```python
import matplotlib.font_manager as fm

font_path = '../../../../usr/share/fonts/truetype/nanum/NanumSquareR.ttf'
# font_path = '../../../../usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf'
font_prop = fm.FontProperties(fname=font_path)

plt.rc('font', family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 20

# plt.rcParams['xtick.major.size'] = 0
# plt.rcParams['ytick.major.size'] = 0

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# plt.style.use('seaborn-v0_8')

plt.title('한글 테스트')
plt.text(0.5, 0.5, '속', ha='center', va='center')
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_3_0.png)
    


## 데이터셋 통합하기


```python
data_path = './data/'
years = range(1995, 2005)

df_list = []

for year in years:
    file_path = f'{data_path}csat{year}.dta'
    
    temp_df = pd.read_stata(file_path)
    temp_df['year'] = year
    
    df_list.append(temp_df)
    
df_total = pd.concat(df_list, ignore_index=True)
```


```python
df_total.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7648411 entries, 0 to 7648410
    Data columns (total 6 columns):
     #   Column   Dtype  
    ---  ------   -----  
     0   lea      object 
     1   retaker  int8   
     2   female   int8   
     3   kor_ss   float32
     4   eng_ss   float32
     5   year     int64  
    dtypes: float32(2), int64(1), int8(2), object(1)
    memory usage: 189.6+ MB



```python
df_total.head()
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
      <th>lea</th>
      <th>retaker</th>
      <th>female</th>
      <th>kor_ss</th>
      <th>eng_ss</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>강원</td>
      <td>0</td>
      <td>1</td>
      <td>73.477730</td>
      <td>66.827797</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>강원</td>
      <td>0</td>
      <td>1</td>
      <td>97.580811</td>
      <td>72.456558</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>강원</td>
      <td>0</td>
      <td>0</td>
      <td>97.192047</td>
      <td>85.878998</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>강원</td>
      <td>0</td>
      <td>0</td>
      <td>71.922691</td>
      <td>63.363941</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>강원</td>
      <td>0</td>
      <td>1</td>
      <td>69.978897</td>
      <td>77.219360</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_total.tail()
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
      <th>lea</th>
      <th>retaker</th>
      <th>female</th>
      <th>kor_ss</th>
      <th>eng_ss</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7648406</th>
      <td>부산</td>
      <td>0</td>
      <td>0</td>
      <td>124.745102</td>
      <td>100.022659</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>7648407</th>
      <td>경북</td>
      <td>0</td>
      <td>0</td>
      <td>115.726868</td>
      <td>110.985092</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>7648408</th>
      <td>서울</td>
      <td>0</td>
      <td>0</td>
      <td>96.688370</td>
      <td>90.156479</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>7648409</th>
      <td>경기</td>
      <td>0</td>
      <td>1</td>
      <td>82.660011</td>
      <td>82.482780</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>7648410</th>
      <td>대전</td>
      <td>0</td>
      <td>0</td>
      <td>112.720787</td>
      <td>113.177574</td>
      <td>2004</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_total[df_total['year']==2001].describe()
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
      <th>retaker</th>
      <th>female</th>
      <th>kor_ss</th>
      <th>eng_ss</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>835531.000000</td>
      <td>835531.000000</td>
      <td>835531.000000</td>
      <td>835452.000000</td>
      <td>835531.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.286492</td>
      <td>0.467686</td>
      <td>100.000008</td>
      <td>100.000000</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.452122</td>
      <td>0.498955</td>
      <td>20.000000</td>
      <td>20.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-4.339706</td>
      <td>40.896797</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>89.597305</td>
      <td>83.307175</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>106.597511</td>
      <td>102.838264</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>115.533516</td>
      <td>117.905113</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>126.431076</td>
      <td>130.181793</td>
      <td>2001.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_sample = df_total.sample(frac=0.1, random_state=2025)

df_sample.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 764841 entries, 3943821 to 7399459
    Data columns (total 6 columns):
     #   Column   Non-Null Count   Dtype  
    ---  ------   --------------   -----  
     0   lea      764841 non-null  object 
     1   retaker  764841 non-null  int8   
     2   female   764841 non-null  int8   
     3   kor_ss   764841 non-null  float32
     4   eng_ss   764764 non-null  float32
     5   year     764841 non-null  int64  
    dtypes: float32(2), int64(1), int8(2), object(1)
    memory usage: 24.8+ MB


## 문제 1-1
재학생 수능 응시생을 토대로 언어영역 표준점수의 연도별 추세를 서울 지역과 비서울 지역으로 구분해서 산점도로 제시해 보세요.


```python
## 재학생 응시생만 따로 추출
temp1 = df_total[df_total['retaker']==0]
temp2 = df_sample[df_sample['retaker']==0]

## 연도, 지역, 언어점수만 따로 추출
temp1 = temp1[['year', 'lea', 'kor_ss', 'retaker']]
temp2 = temp2[['year', 'lea', 'kor_ss', 'retaker']]

## 지역 구분: 서울 VS 비서울
temp1['region'] = temp1['lea'].apply(lambda x: '서울' if x=='서울' else '비서울')
temp2['region'] = temp2['lea'].apply(lambda x: '서울' if x=='서울' else '비서울')

## 연도별 지역 구분 변수 생성
temp1['year_region'] = temp1['year'].astype(str) + '_' + temp1['region']
temp2['year_region'] = temp2['year'].astype(str) + '_' + temp2['region']
```


```python
# sorted(temp['year_region'].unique()) # 심플하지만 비서울, 서울 순서

x_axis_order = []
years = range(1995, 2005)

for year in years:
    x_axis_order.append(f'{year}_서울')
    x_axis_order.append(f'{year}_비서울')
    
# x_axis_order
```


```python
fig, ax = plt.subplots(figsize=(20,10))

sns.stripplot(data=temp2, x='year_region', y='kor_ss',
              hue='region',
              jitter=True,
              palette='Set2',
              alpha=0.5,
              ax=ax,
              order=x_axis_order,
              hue_order=['서울', '비서울'],
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='x', rotation=45)

ax.set_title('연도별/지역별 재학생 언어영역 표준점수 산점도 (1995-2004)', fontsize=22)
ax.set_xlabel('연도 및 지역', fontsize=18)
ax.set_ylabel('언어영역 표준점수', fontsize=18)

ax.legend(title='지역 구분', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
        #   shadow=True,
          )

plt.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_14_0.png)
    



```python
fig, ax = plt.subplots(figsize=(20,10))

sns.boxplot(data=temp1, x='year_region', y='kor_ss',
            hue='region',
            palette='Set2',
            showfliers=False,
            legend=False,
            ax=ax,
            order=x_axis_order,
            hue_order=['서울', '비서울'],
            )

sns.stripplot(data=temp2, x='year_region', y='kor_ss',
              hue='region',
              jitter=True,
            #   palette='dark:black',
              palette='Set2',
              alpha=0.5,
              ax=ax,
              order=x_axis_order,
              hue_order=['서울', '비서울'],
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='x', rotation=45)

ax.set_title('연도별/지역별 재학생 언어영역 표준점수 산점도+박스플롯 (1995-2004)', fontsize=22)
ax.set_xlabel('연도 및 지역', fontsize=18)
ax.set_ylabel('언어영역 표준점수', fontsize=18)

ax.legend(title='지역 구분', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
        #   shadow=True,
          )

plt.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_15_0.png)
    


## 문제 1-2
재수생 이상 수능 응시생을 토대로 언어영역 표준점수의 연도별 추세를 서울 지역과 비서울 지역으로 구분해서 산점도로 제시해 보세요.


```python
## 재수 이상의 응시생만 따로 추출
temp1 = df_total[df_total['retaker']!=0]
temp2 = df_sample[df_sample['retaker']!=0]

## 연도, 지역, 언어점수만 따로 추출
temp1 = temp1[['year', 'lea', 'kor_ss', 'retaker']]
temp2 = temp2[['year', 'lea', 'kor_ss', 'retaker']]

## 지역 구분: 서울 VS 비서울
temp1['region'] = temp1['lea'].apply(lambda x: '서울' if x=='서울' else '비서울')
temp2['region'] = temp2['lea'].apply(lambda x: '서울' if x=='서울' else '비서울')

## 연도별 지역 구분 변수 생성
temp1['year_region'] = temp1['year'].astype(str) + '_' + temp1['region']
temp2['year_region'] = temp2['year'].astype(str) + '_' + temp2['region']
```


```python
fig, ax = plt.subplots(figsize=(20,10))

sns.stripplot(data=temp2, x='year_region', y='kor_ss',
              hue='region',
              jitter=True,
              palette='Set2',
              alpha=0.5,
              ax=ax,
              order=x_axis_order,
              hue_order=['서울', '비서울'],
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='x', rotation=45)

ax.set_title('연도별/지역별 재수 이상 수험생 언어영역 표준점수 산점도 (1995-2004)', fontsize=22)
ax.set_xlabel('연도 및 지역', fontsize=18)
ax.set_ylabel('언어영역 표준점수', fontsize=18)

ax.legend(title='지역 구분', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

plt.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_18_0.png)
    



```python
fig, ax = plt.subplots(figsize=(20,10))

sns.boxplot(data=temp1, x='year_region', y='kor_ss',
            hue='region',
            palette='Set2',
            showfliers=False,
            legend=False,
            ax=ax,
            order=x_axis_order,
            hue_order=['서울', '비서울'],
            )

sns.stripplot(data=temp2, x='year_region', y='kor_ss',
              hue='region',
              jitter=True,
              palette='Set2',
              alpha=0.5,
              ax=ax,
              order=x_axis_order,
              hue_order=['서울', '비서울'],
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='x', rotation=45)

ax.set_title('연도별/지역별 재수 이상 수험생 언어영역 표준점수 산점도+박스플롯 (1995-2004)', fontsize=22)
ax.set_xlabel('연도 및 지역', fontsize=18)
ax.set_ylabel('언어영역 표준점수', fontsize=18)

ax.legend(title='지역 구분', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

plt.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_19_0.png)
    


## 문제 1-3
재학생 수능 응시생을 토대로 영어영역 표준점수의 연도별 추세를 서울 지역과 비서울 지역으로 구분해서 산점도로 제시해 보세요


```python
## 재학생 응시생만 따로 추출
temp1 = df_total[df_total['retaker']==0]
temp2 = df_sample[df_sample['retaker']==0]

## 연도, 지역, 영어점수만 따로 추출
temp1 = temp1[['year', 'lea', 'eng_ss', 'retaker']]
temp2 = temp2[['year', 'lea', 'eng_ss', 'retaker']]

## 지역 구분: 서울 VS 비서울
temp1['region'] = temp1['lea'].apply(lambda x: '서울' if x=='서울' else '비서울')
temp2['region'] = temp2['lea'].apply(lambda x: '서울' if x=='서울' else '비서울')

## 연도별 지역 구분 변수 생성
temp1['year_region'] = temp1['year'].astype(str) + '_' + temp1['region']
temp2['year_region'] = temp2['year'].astype(str) + '_' + temp2['region']
```


```python
fig, ax = plt.subplots(figsize=(20,10))

sns.stripplot(data=temp2, x='year_region', y='eng_ss',
              hue='region',
              jitter=True,
              palette='Set2',
              alpha=0.5,
              ax=ax,
              order=x_axis_order,
              hue_order=['서울', '비서울'],
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='x', rotation=45)

ax.set_title('연도별/지역별 재학생 영어영역 표준점수 산점도 (1995-2004)', fontsize=22)
ax.set_xlabel('연도 및 지역', fontsize=18)
ax.set_ylabel('영어영역 표준점수', fontsize=18)

ax.legend(title='지역 구분', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

plt.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_22_0.png)
    



```python
fig, ax = plt.subplots(figsize=(20,10))

sns.boxplot(data=temp1, x='year_region', y='eng_ss',
            hue='region',
            palette='Set2',
            showfliers=False,
            legend=False,
            ax=ax,
            order=x_axis_order,
            hue_order=['서울', '비서울'],
            )

sns.stripplot(data=temp2, x='year_region', y='eng_ss',
              hue='region',
              jitter=True,
              palette='Set2',
              alpha=0.5,
              ax=ax,
              order=x_axis_order,
              hue_order=['서울', '비서울'],
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='x', rotation=45)

ax.set_title('연도별/지역별 재학생 영어영역 표준점수 산점도+박스플롯 (1995-2004)', fontsize=22)
ax.set_xlabel('연도 및 지역', fontsize=18)
ax.set_ylabel('영어영역 표준점수', fontsize=18)

ax.legend(title='지역 구분', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

plt.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_23_0.png)
    


## 문제 1-4
재수생 이상 수능 응시생을 토대로 영어영역 표준점수의 연도별 추세를 서울 지역과 비서울 지역으로 구분해서 산점도로 제시해 보세요.


```python
## 재수 이상의 응시생만 따로 추출
temp1 = df_total[df_total['retaker']!=0]
temp2 = df_sample[df_sample['retaker']!=0]

## 연도, 지역, 언어점수만 따로 추출
temp1 = temp1[['year', 'lea', 'eng_ss', 'retaker']]
temp2 = temp2[['year', 'lea', 'eng_ss', 'retaker']]

## 지역 구분: 서울 VS 비서울
temp1['region'] = temp1['lea'].apply(lambda x: '서울' if x=='서울' else '비서울')
temp2['region'] = temp2['lea'].apply(lambda x: '서울' if x=='서울' else '비서울')

## 연도별 지역 구분 변수 생성
temp1['year_region'] = temp1['year'].astype(str) + '_' + temp1['region']
temp2['year_region'] = temp2['year'].astype(str) + '_' + temp2['region']
```


```python
fig, ax = plt.subplots(figsize=(20,10))

sns.stripplot(data=temp2, x='year_region', y='eng_ss',
              hue='region',
              jitter=True,
              palette='Set2',
              alpha=0.5,
              ax=ax,
              order=x_axis_order,
              hue_order=['서울', '비서울'],
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='x', rotation=45)

ax.set_title('연도별/지역별 재수 이상 수험생 영어영역 표준점수 산점도 (1995-2004)', fontsize=22)
ax.set_xlabel('연도 및 지역', fontsize=18)
ax.set_ylabel('영어영역 표준점수', fontsize=18)

ax.legend(title='지역 구분', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

plt.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_26_0.png)
    



```python
fig, ax = plt.subplots(figsize=(20,10))

sns.boxplot(data=temp1, x='year_region', y='eng_ss',
            hue='region',
            palette='Set2',
            showfliers=False,
            legend=False,
            ax=ax,
            order=x_axis_order,
            hue_order=['서울', '비서울'],
            )

sns.stripplot(data=temp2, x='year_region', y='eng_ss',
              hue='region',
              jitter=True,
              palette='Set2',
              alpha=0.5,
              ax=ax,
              order=x_axis_order,
              hue_order=['서울', '비서울'],
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='x', rotation=45)

ax.set_title('연도별/지역별 재수 이상 수험생 영어영역 표준점수 산점도+박스플롯 (1995-2004)', fontsize=22)
ax.set_xlabel('연도 및 지역', fontsize=18)
ax.set_ylabel('영어영역 표준점수', fontsize=18)

ax.legend(title='지역 구분', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

plt.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_27_0.png)
    


## 문제 2-1
연도별 재수생 이상인 학생의 수 추세를 산점도로 제시하십시오


```python
## 재수 이상의 응시생만 따로 추출
temp1 = df_total[df_total['retaker']!=0]
temp2 = df_sample[df_sample['retaker']!=0]
```


```python
dict_retaker = temp1['year'].value_counts().to_dict()
list_years = sorted(dict_retaker.keys())
list_retakers = [ dict_retaker[year] for year in list_years ]
```


```python
import matplotlib.ticker as ticker
```


```python
fig, ax = plt.subplots(figsize=(14, 8))

sns.lineplot(x=list_years, y=list_retakers,
             marker='o', markersize=10,
             linewidth=1.5,
             ax=ax
             )

ax.set_ylim(0, 300000)
ax.set_yticks(np.arange(0, 300001, 20000))
ax.set_xticks(np.arange(1995, 2005, 1))

ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
# ax.tick_params(axis='x', rotation=45)

ax.set_title('연도별 재수 이상 응시생 수 산점도 및 추세선 (1995-2004)', fontsize=22)
ax.set_ylabel('재수 이상 응시생 수 (단위: 명)', fontsize=18)
ax.set_xlabel('연도', fontsize=18)

formatter = ticker.StrMethodFormatter('{x:,.0f}')
ax.yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_32_0.png)
    


## 문제 2-2
연도별 재수생 이상인 학생의 비율 추세를 산점도로 제시하십시오


```python
# df_total.groupby(by=['year', 'retaker'], as_index=False)[['year','retaker']].value_counts(normalize=True)
# df_total['retaker'].value_counts(normalize=True)
df_retaker_ratio = df_total.groupby('year')['retaker'].mean() * 100
df_retaker_ratio = pd.DataFrame(df_retaker_ratio)
```


```python
fig, ax = plt.subplots(figsize=(14, 8))

sns.lineplot(data=df_retaker_ratio, x='year', y='retaker',
             marker='o', markersize=10,
             linewidth=1.5,
             ax=ax,
             )

# ax.set_ylim(0, 300000)
ax.set_yticks(np.arange(0, 51, 5))
ax.set_xticks(np.arange(1995, 2005, 1))

ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)
# ax.tick_params(axis='x', rotation=45)

ax.set_title('연도별 재수 이상 응시생 비율 산점도 및 추세선 (1995-2004)', fontsize=22)
ax.set_ylabel('재수 이상 응시생의 비율 (단위: %)', fontsize=18)
ax.set_xlabel('연도', fontsize=18)

# formatter = ticker.StrMethodFormatter('{x:,.0f}')
# ax.yaxis.set_major_formatter(formatter)

fig.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_35_0.png)
    


## 문제 2-3
연도별 재수생 이상인 학생의 비율 추세를 여성과 남성으로 구분해서 산점도로 제시하십시오


```python
# df_retaker_male = df_total.loc[ (df_total['retaker']!=0)&(df_total['female']!=0) , ]
# df_retaker_male = df_retaker_male.reset_index(drop=True)

# df_retaker_female = df_total.loc[ (df_total['retaker']!=0)&(df_total['female']==0) , ]
# df_retaker_female = df_retaker_female.reset_index(drop=True)

# df_retaker_male.groupby('year')['retaker'].mean()
```




    year
    1995    1.0
    1996    1.0
    1997    1.0
    1998    1.0
    1999    1.0
    2000    1.0
    2001    1.0
    2002    1.0
    2003    1.0
    2004    1.0
    Name: retaker, dtype: float64




```python
df_retaker_ratio = pd.DataFrame(df_total.groupby(['year', 'female'], as_index=False)['retaker'].mean())
df_retaker_ratio['retaker'] = df_retaker_ratio['retaker'] * 100

df_retaker_ratio_m = df_retaker_ratio.loc[ df_retaker_ratio['female']==0 , ]
df_retaker_ratio_f = df_retaker_ratio.loc[ df_retaker_ratio['female']==1 , ]
```


```python
df_retaker_ratio_m
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
      <th>year</th>
      <th>female</th>
      <th>retaker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1995</td>
      <td>0</td>
      <td>36.037625</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1996</td>
      <td>0</td>
      <td>36.103768</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997</td>
      <td>0</td>
      <td>33.598017</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1998</td>
      <td>0</td>
      <td>30.407536</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1999</td>
      <td>0</td>
      <td>27.923749</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2000</td>
      <td>0</td>
      <td>28.108000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2001</td>
      <td>0</td>
      <td>29.942779</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2002</td>
      <td>0</td>
      <td>25.663626</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2003</td>
      <td>0</td>
      <td>27.572845</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2004</td>
      <td>0</td>
      <td>29.173193</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(14, 8))

sns.lineplot(data=df_retaker_ratio_m, x='year', y='retaker',
             marker='o', markersize=10,
             linewidth=1.5,
             label='남성',
             ax=ax,
             )

sns.lineplot(data=df_retaker_ratio_f, x='year', y='retaker',
             marker='o', markersize=10,
             linewidth=1.5,
             label='여성',
             ax=ax,
             )

ax.set_yticks(np.arange(0, 51, 5))
ax.set_xticks(np.arange(1995, 2005, 1))

ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

ax.set_title('연도/성별에 따른 재수 이상 응시생 비율 산점도 및 추세선 (1995-2004)', fontsize=22)
ax.set_ylabel('재수 이상 응시생의 비율 (단위: %)', fontsize=18)
ax.set_xlabel('연도', fontsize=18)

ax.legend(title='성별', loc='upper right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

fig.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_40_0.png)
    



```python
df_retaker_ratio = pd.DataFrame(df_total.groupby(['year', 'female'], as_index=False)['retaker'].mean())
df_retaker_ratio['retaker_percentage'] = df_retaker_ratio['retaker'] * 100

df_retaker_ratio['gender'] = df_retaker_ratio['female'].apply(lambda x: '여성' if x == 1 else '남성')
df_retaker_ratio
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
      <th>year</th>
      <th>female</th>
      <th>retaker</th>
      <th>retaker_percentage</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1995</td>
      <td>0</td>
      <td>0.360376</td>
      <td>36.037625</td>
      <td>남성</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1995</td>
      <td>1</td>
      <td>0.320271</td>
      <td>32.027091</td>
      <td>여성</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1996</td>
      <td>0</td>
      <td>0.361038</td>
      <td>36.103768</td>
      <td>남성</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1996</td>
      <td>1</td>
      <td>0.322232</td>
      <td>32.223198</td>
      <td>여성</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1997</td>
      <td>0</td>
      <td>0.335980</td>
      <td>33.598017</td>
      <td>남성</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1997</td>
      <td>1</td>
      <td>0.285877</td>
      <td>28.587735</td>
      <td>여성</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1998</td>
      <td>0</td>
      <td>0.304075</td>
      <td>30.407536</td>
      <td>남성</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1998</td>
      <td>1</td>
      <td>0.256174</td>
      <td>25.617365</td>
      <td>여성</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1999</td>
      <td>0</td>
      <td>0.279237</td>
      <td>27.923749</td>
      <td>남성</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1999</td>
      <td>1</td>
      <td>0.231773</td>
      <td>23.177336</td>
      <td>여성</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2000</td>
      <td>0</td>
      <td>0.281080</td>
      <td>28.108000</td>
      <td>남성</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2000</td>
      <td>1</td>
      <td>0.257407</td>
      <td>25.740702</td>
      <td>여성</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2001</td>
      <td>0</td>
      <td>0.299428</td>
      <td>29.942779</td>
      <td>남성</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2001</td>
      <td>1</td>
      <td>0.271769</td>
      <td>27.176878</td>
      <td>여성</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2002</td>
      <td>0</td>
      <td>0.256636</td>
      <td>25.663626</td>
      <td>남성</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2002</td>
      <td>1</td>
      <td>0.230839</td>
      <td>23.083858</td>
      <td>여성</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2003</td>
      <td>0</td>
      <td>0.275728</td>
      <td>27.572845</td>
      <td>남성</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2003</td>
      <td>1</td>
      <td>0.250123</td>
      <td>25.012291</td>
      <td>여성</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2004</td>
      <td>0</td>
      <td>0.291732</td>
      <td>29.173193</td>
      <td>남성</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2004</td>
      <td>1</td>
      <td>0.255041</td>
      <td>25.504083</td>
      <td>여성</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(14,8))

sns.lineplot(data=df_retaker_ratio, x='year', y='retaker_percentage',
             hue='gender',
            #  style='gender',
            marker='o', markersize=10,
            linewidth=1.5,
            )

ax.set_yticks(np.arange(0, 51, 5))
ax.set_xticks(np.arange(1995, 2005, 1))

ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

ax.set_title('연도/성별에 따른 재수 이상 응시생 비율 산점도 및 추세선 (1995-2004)', fontsize=22)
ax.set_ylabel('재수 이상 응시생의 비율 (단위: %)', fontsize=18)
ax.set_xlabel('연도', fontsize=18)

ax.legend(title='성별', loc='upper right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

fig.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_42_0.png)
    


## 문제 3-1
재학생 수능 응시생을 토대로 언어영역 표준점수의 연도별 추세를 여성과 남성으로 구분해서 산점도로 제시해 보세요


```python
df_first = df_total[df_total['retaker']==0]
df_first['gender'] = df_first['female'].apply(lambda x: '여성' if x == 1 else '남성')
# df_first.groupby(['year', 'gender'], as_index=False)[['year', 'gender']].count()
df_first['year_gender'] = df_first['year'].astype(str) + '_' + df_first['gender']
```

    /tmp/ipykernel_871/3476366203.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_first['gender'] = df_first['female'].apply(lambda x: '여성' if x == 1 else '남성')
    /tmp/ipykernel_871/3476366203.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_first['year_gender'] = df_first['year'].astype(str) + '_' + df_first['gender']



```python
x_axis_order = []
years = range(1995, 2005)

for year in years:
    x_axis_order.append(f'{year}_여성')
    x_axis_order.append(f'{year}_남성')
    
# x_axis_order
```


```python
fig, ax = plt.subplots(figsize=(20,10))

sns.stripplot(data=df_first, x='year_gender', y='kor_ss',
              hue='gender',
              jitter=True,
              palette='Set1',
              alpha=0.5,
              ax=ax,
              order=x_axis_order,
              hue_order=['여성', '남성'],
              )

# ax.set_ylim(-10, 180)
# ax.set_yticks(np.arange(0, 181, 25))
# ax.tick_params(axis='y', labelsize=14)
# ax.tick_params(axis='x', labelsize=14)
# ax.tick_params(axis='x', rotation=45)

# ax.set_title('연도별/지역별 재학생 영어영역 표준점수 산점도 (1995-2004)', fontsize=22)
# ax.set_xlabel('연도 및 지역', fontsize=18)
# ax.set_ylabel('영어영역 표준점수', fontsize=18)

# ax.legend(title='지역 구분', loc='lower right',
#           title_fontsize=18, fontsize=16,
#           frameon=True,
#           )

fig.tight_layout()
plt.show()
```

    /tmp/ipykernel_871/2296194278.py:28: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
      fig.tight_layout()
    /home/drkail/uv_projects/statistics/.venv/lib/python3.12/site-packages/IPython/core/pylabtools.py:170: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
      fig.canvas.print_figure(bytes_io, **kw)



    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_46_1.png)
    



```python
df_first = df_total[df_total['retaker']==0].copy()
df_first['gender'] = df_first['female'].apply(lambda x: '여성' if x==1 else '남성')
```


```python
fig, ax = plt.subplots(figsize=(20,10))

sns.stripplot(data=df_first, x='year', y='kor_ss',
              hue='gender',
              jitter=True,
              palette='Set1',
              alpha=0.3,
              ax=ax,
              dodge=True,
              hue_order=['여성', '남성']
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

ax.set_title('성별 및 연도별 재학생 언어영역 표준점수 산점도 (1995-2004)', fontsize=22)
ax.set_xlabel('연도', fontsize=18)
ax.set_ylabel('언어영역 표준점수', fontsize=18)

ax.legend(title='성별', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

fig.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_48_0.png)
    



```python
df_sample = df_first.sample(n=20000, random_state=2025)

fig, ax = plt.subplots(figsize=(20, 10))

sns.boxplot(data=df_first, x='year', y='kor_ss',
            hue='gender',
            ax=ax,
            palette='Pastel1',
            showfliers=False,
            hue_order=['여성', '남성']
            )

sns.stripplot(data=df_sample, x='year', y='kor_ss',
              hue='gender',
              legend=False,
              ax=ax,
              palette='Pastel1',
              jitter=True,
              alpha=0.4,
              dodge=True,
              hue_order=['여성', '남성']
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

ax.set_title('성별 및 연도별 재학생 언어영역 표준점수 산점도 및 박스플롯 (1995-2004)', fontsize=22)
ax.set_xlabel('연도', fontsize=18)
ax.set_ylabel('언어영역 표준점수', fontsize=18)

ax.legend(title='성별', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

fig.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_49_0.png)
    


## 문제 3-2
재수생 이상 수능 응시생을 토대로 언어영역 표준점수의 연도별 추세를 여성과 남성으로 구분해서 산점도로 제시해 보세요


```python
df_retaker = df_total[df_total['retaker']!=0].copy()
df_retaker['gender'] = df_retaker['female'].apply(lambda x: '여성' if x==1 else '남성')
df_sample = df_retaker.sample(n=20000, random_state=2025)
```


```python
fig, ax = plt.subplots(figsize=(20, 10))

sns.boxplot(data=df_retaker, x='year', y='kor_ss',
            hue='gender',
            ax=ax,
            palette='Pastel1',
            showfliers=False,
            hue_order=['여성', '남성']
            )

sns.stripplot(data=df_sample, x='year', y='kor_ss',
              hue='gender',
              legend=False,
              ax=ax,
              palette='Pastel1',
              jitter=True,
              alpha=0.4,
              dodge=True,
              hue_order=['여성', '남성']
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

ax.set_title('성별 및 연도별 재수 이상 응시생 언어영역 표준점수 산점도 및 박스플롯 (1995-2004)', fontsize=22)
ax.set_xlabel('연도', fontsize=18)
ax.set_ylabel('언어영역 표준점수', fontsize=18)

ax.legend(title='성별', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

fig.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_52_0.png)
    


## 문제 3-3
재학생 수능 응시생을 토대로 영어영역 표준점수의 연도별 추세를 여성과 남성으로 구분해서 산점도로 제시해 보세요


```python
df_first = df_total[df_total['retaker']==0].copy()
df_first['gender'] = df_first['female'].apply(lambda x: '여성' if x==1 else '남성')
df_sample = df_first.sample(n=20000, random_state=2025)

fig, ax = plt.subplots(figsize=(20, 10))

sns.boxplot(data=df_first, x='year', y='eng_ss',
            hue='gender',
            ax=ax,
            palette='Pastel1',
            showfliers=False,
            hue_order=['여성', '남성']
            )

sns.stripplot(data=df_sample, x='year', y='eng_ss',
              hue='gender',
              legend=False,
              ax=ax,
              palette='Pastel1',
              jitter=True,
              alpha=0.4,
              dodge=True,
              hue_order=['여성', '남성']
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

ax.set_title('성별 및 연도별 재학생 영어영역 표준점수 산점도 및 박스플롯 (1995-2004)', fontsize=22)
ax.set_xlabel('연도', fontsize=18)
ax.set_ylabel('영어영역 표준점수', fontsize=18)

ax.legend(title='성별', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

fig.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_54_0.png)
    


## 문제 3-4
재수생 이상 수능 응시생을 토대로 영어영역 표준점수의 연도별 추세를 여성과 남성으로 구분해서 산점도로 제시해 보세요


```python
df_retaker = df_total[df_total['retaker']!=0].copy()
df_retaker['gender'] = df_retaker['female'].apply(lambda x: '여성' if x==1 else '남성')
df_sample = df_retaker.sample(n=20000, random_state=2025)

fig, ax = plt.subplots(figsize=(20, 10))

sns.boxplot(data=df_retaker, x='year', y='eng_ss',
            hue='gender',
            ax=ax,
            palette='Pastel1',
            showfliers=False,
            hue_order=['여성', '남성']
            )

sns.stripplot(data=df_sample, x='year', y='eng_ss',
              hue='gender',
              legend=False,
              ax=ax,
              palette='Pastel1',
              jitter=True,
              alpha=0.4,
              dodge=True,
              hue_order=['여성', '남성']
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

ax.set_title('성별 및 연도별 재수 이상 응시생 영어영역 표준점수 산점도 및 박스플롯 (1995-2004)', fontsize=22)
ax.set_xlabel('연도', fontsize=18)
ax.set_ylabel('영어영역 표준점수', fontsize=18)

ax.legend(title='성별', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

fig.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_56_0.png)
    


## 문제 4-1
언어영역 표준점수의 연도별 추세를 재수생 이상인 학생과 재학생으로 구분해서 산점도로 제시해 보세요


```python
df_temp = df_total.copy()
df_temp['retaker'] = df_temp['retaker'].map({0: '재학생', 1: '재수 이상 응시생'})
df_sample = df_temp.sample(n=20000, random_state=2025)

fig, ax = plt.subplots(figsize=(20,10))

sns.boxplot(data=df_temp, x='year', y='kor_ss',
            hue='retaker',
            ax=ax,
            palette='Set1',
            showfliers=False,
            )

sns.stripplot(data=df_sample, x='year', y='kor_ss',
              hue='retaker',
              palette='Set1',
              ax=ax,
              alpha=0.2,
              dodge=True,
              jitter=True,
              legend=False,
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

ax.set_title('연도별/재수 여부별 언어영역 표준점수 산점도 및 박스플롯 (1995-2004)', fontsize=22)
ax.set_xlabel('연도', fontsize=18)
ax.set_ylabel('언어영역 표준점수', fontsize=18)

ax.legend(title='재수 여부', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

fig.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_58_0.png)
    


## 문제 4-2
영어영역 표준점수의 연도별 추세를 재수생 이상인 학생과 재학생으로 구분해서 산점도로 제시해 보세요


```python
df_temp = df_total.copy()
df_temp['retaker'] = df_temp['retaker'].map({0: '재학생', 1: '재수 이상 응시생'})
df_sample = df_temp.sample(n=20000, random_state=2025)

fig, ax = plt.subplots(figsize=(20,10))

sns.boxplot(data=df_temp, x='year', y='eng_ss',
            hue='retaker',
            ax=ax,
            palette='Set1',
            showfliers=False,
            )

sns.stripplot(data=df_sample, x='year', y='eng_ss',
              hue='retaker',
              palette='Set1',
              ax=ax,
              alpha=0.2,
              dodge=True,
              jitter=True,
              legend=False,
              )

ax.set_ylim(-10, 180)
ax.set_yticks(np.arange(0, 181, 25))
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='x', labelsize=14)

ax.set_title('연도별/재수 여부별 영어영역 표준점수 산점도 및 박스플롯 (1995-2004)', fontsize=22)
ax.set_xlabel('연도', fontsize=18)
ax.set_ylabel('영어영역 표준점수', fontsize=18)

ax.legend(title='재수 여부', loc='lower right',
          title_fontsize=18, fontsize=16,
          frameon=True,
          )

fig.tight_layout()
plt.show()
```


    
![png](../../assets/images/graduate_assignment/공공통계II_과제1_60_0.png)
    



```python

```
