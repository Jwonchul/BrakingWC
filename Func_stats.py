from unoUtil.readDataFile import *
from unoUtil.unoPostProcessing import *
from unoUtil.vehicleTest import makeStandardDF
from unoUtil.gpsProcessing import GPStoDistance2, changeVBOXCoord
import os
from unoUtil.unoPostProcessing import SP_MovingAverage
from unoUtil.vehicleTest._HndlTest_ import SP_CurveFit_MF
from unoUtil.unoAI.fitMinimizeOptimization import MinimizeEstimator
from unoUtil.unoAI._DynamicModel_ import TireMagicFormula
from unoUtil.readDataFile.ReadDataFile import ReadDataFile
import itertools
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from unoUtil.readDataFile.ReadDataFile import ReadDataFile
from unoUtil.vehicleTest._BRKTest_ import *
from unoUtil.unoPostProcessing import SP_MovingAverage
import os
from scipy.stats import skew, kurtosis
from scipy.stats import skewnorm
import re
import pickle
from scipy import stats


from scipy import stats
import seaborn as sns


def cohens_d_compare(mean1, std1, n1, mean2, std2, n2):
    """
    Cohen's d는 두 집단 간 평균 차이가 얼마나 큰지를 표준편차 기준으로 정규화해서 보여주는 효과 크기(effect size) 지표,
    d < 0.2 : 작은 효과 (small) / d = 0.5 : 중간 효과 (medium) / d > 0.8 : 큰 효과 (large)
    """
    # Pooled standard deviation 계산
    s_pooled = np.sqrt(((n1 - 1)*(std1**2) + (n2 - 1)*(std2**2)) / (n1 + n2 - 2))
    # Cohen's d 계산
    d = (mean1 - mean2) / s_pooled
    return d

def rank_compare(series, tolerance=0.015):
    """
    같은 Rank 그룹의 평균과 비교하여 tolerance 이내면 같은 Rank,
    아니면 Rank 증가하는 방식으로 Rank 부여
    """
    series = series.sort_values(ascending=False).reset_index()
    values = series[series.columns[1]].values
    index = series['index'].values

    ranks = []
    current_rank = 1
    current_group_values = [values[0]]
    ranks.append(current_rank)

    for i in range(1, len(values)):
        avg = np.mean(current_group_values)
        if abs(values[i] - avg) >= tolerance:
            current_rank += len(current_group_values)
            current_group_values = [values[i]]
        else:
            current_group_values.append(values[i])
        ranks.append(current_rank)

    return pd.Series(ranks, index=index)

def stats_iqr(series, percentage=25, **kwargs):
    """
    Q1 (제1사분위수): 하위 25% / Q3 (제3사분위수): 상위 75%
    remove 활성화 하면 이상치 상한과 하한으로 cut
    chname 설정으로 칼럼 이름 변경 가능
    """
    remove = kwargs.get('remove', False)
    chname = kwargs.get('chname', 'series')

    q1 = series.quantile(percentage/100)
    q3 = series.quantile((1-percentage/100))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # 시리즈 이름 기반으로 컬럼명 구성
    prefix = chname
    result = pd.DataFrame({f'{prefix}_min': [series.min()],
                           f'{prefix}_max': [series.max()],
                           f'{prefix}_mean': [series.mean()],
                           f'{prefix}_q1': [q1],
                           f'{prefix}_q3': [q3],
                           f'{prefix}_iqr': [iqr],
                           f'{prefix}_lower_bound': [lower],
                           f'{prefix}_upper_bound': [upper]})

    if remove:
        return result, series[(series >= lower) & (series <= upper)]

    return result

def stats_tdist(series, confidence=95, **kwargs):
    """
    t분포 계산
    chname 설정으로 칼럼 이름 변경 가능
    """
    from scipy import stats

    plot = kwargs.get("plot", False)
    chname = kwargs.get('chname', 'series')

    # series = series[['Dist_0','Dist_1','Dist_2','Dist_3']]
    n = len(series)
    df = n - 1 # 자유도
    mean = np.mean(series)
    std_dev = np.std(series, ddof=1)

    # 신뢰수준 설정 (예: 95%)
    # confidence = 0.95
    t_crit = stats.t.ppf((1 + (confidence/100)) / 2, df)  # 양측 t-값

    # 표준 오차
    se = std_dev / np.sqrt(n)

    # 신뢰구간 계산
    margin_of_error = t_crit * se
    lower = mean - margin_of_error
    upper = mean + margin_of_error

    # 시리즈 이름 기반으로 컬럼명 구성
    prefix = chname
    result = pd.DataFrame({f'{prefix}_mean': [mean],
                           f'{prefix}_df': [df],
                           f'{prefix}_se': [se],
                           f'{prefix}_lower': [lower],
                           f'{prefix}_upper': [upper]})

    if plot:
        x = np.linspace(mean - 4 * se, mean + 4 * se, 300)
        y = stats.t.pdf((x - mean) / se, df)  # 중심을 sample mean에 맞추기 위해 변환

        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='t-distribution', color='blue')
        plt.axvline(lower, color='red', linestyle='--', label='95% lower')
        plt.axvline(upper, color='red', linestyle='--', label='95% upper')
        plt.axvline(mean, color='green', linestyle=':', label='Sample Mean')

        # 색칠된 신뢰구간 영역
        x_fill = np.linspace(lower, upper, 200)
        y_fill = stats.t.pdf((x_fill - mean) / se, df)
        plt.fill_between(x_fill, y_fill, color='red', alpha=0.2)

        plt.title("t-Distribution with 95% Confidence Interval")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    return result

def slip_stats(df, left_col, right_col, prefix):

    tmp = df.loc[(~df["testNum"].isna()) & (df["filterAcc"] < 0) &
                 (df[left_col]  > 0) & (df[right_col] > 0) & (df["VelHorizontal"] > 5)]
    slip = pd.Series(np.concatenate([tmp[left_col].values, tmp[right_col].values]))
    x, y = AN_Histogram(slip, bandwidth=0.3, bin=1000)
    peak  = x[np.argmax(y)]

    ## figure
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.plot(x, y, label=prefix)
    plt.axvline(peak, color='red', linestyle='--',label=f"MaxSlip={peak:.2f}")
    plt.axvline(slip.mean(), color='yellow', linestyle='--', label=f"mean={slip.mean():.2f}")
    plt.axvline(slip.median(), color='green', linestyle='--', label=f"median={slip.median():.2f}")
    plt.xlim(-5, 30)
    plt.ylabel('Pdf')
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.subplot(122)
    plt.plot(slip, marker='o',linestyle='')
    plt.axhline(peak, color='red', linestyle='--',label=f"MaxSlip={peak:.2f}")
    plt.axhspan(peak-slip.std(ddof=0)/2, peak+slip.std(ddof=0)/2, color='orange', alpha=0.5,
                label=f"stdev={slip.std(ddof=0):.2f}")
    plt.ylabel('Slip')
    plt.grid(True)
    plt.legend(loc='upper right')

    return {f"{prefix}_MaxSlip": peak,
            f"{prefix}_Skew":    skew(slip),
            f"{prefix}_Kurtosis":kurtosis(slip)}

# def slip_stats_distribution(data,**kwargs):
#     margin = kwargs.get('margin', 3)
#     weights = kwargs.get('weights', None)
#     plot = kwargs.get("plot", False)
#
#     peak_col = [col for col in data.columns if 'MaxSlip' in col][0]
#     skew_col = [col for col in data.columns if 'Skew' in col][0]
#     kurt_col = [col for col in data.columns if 'Kurtosis' in col][0]
#
#     margin = 3
#     x_min = data[peak_col].min() - margin * (1 + data[skew_col].max())
#     x_max = data[peak_col].max() + margin * (1 + data[skew_col].max())
#
#     data = list(data[[peak_col, skew_col, kurt_col]].itertuples(index=False, name=None))
#
#     x = np.linspace(x_min, x_max, 1000)
#     individual_pdfs = []
#     num_distributions = len(data)
#
#     if weights is None:
#         weights = [1 / num_distributions] * num_distributions
#     else:
#         # Normalize weights to sum to 1
#         total = sum(weights)
#         weights = [w / total for w in weights]
#
#     plt.figure(figsize=(10, 6))
#     colors = plt.cm.tab10.colors
#
#     # Plot individual distributions
#     for i, (peak, skew_val, kurt_val) in enumerate(data):
#         a = skew_val
#         dist = skewnorm(a, loc=peak, scale=1)
#         y = dist.pdf(x)
#         y = y ** (kurt_val / 10)  # Approximate kurtosis effect
#         individual_pdfs.append(y)
#         if plot:
#             plt.plot(x, y, label=f'Distribution {i+1}', color=colors[i % len(colors)], alpha=0.6)
#
#     # Compute and plot mixture distribution
#     mixture_pdf = np.zeros_like(x)
#     for y, w in zip(individual_pdfs, weights):
#         mixture_pdf += w * y
#     if plot:
#         plt.plot(x, mixture_pdf, label='Mixture Distribution', color='black', linewidth=2)
#         plt.title("Mixture of Skewed Distributions with Kurtosis")
#         plt.xlabel("Value")
#         plt.ylabel("Density (simulated)")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()

def slip_stats_distribution(df, **kwargs):
    margin = kwargs.get('margin', 3)
    weights = kwargs.get('weights', None)
    plot = kwargs.get("plot", False)

    # 컬럼 이름 추출
    def find_column(keyword):
        matches = [col for col in df.columns if keyword.lower() in col.lower()]
        if not matches:
            raise ValueError(f"'{keyword}'가 포함된 컬럼을 찾을 수 없습니다.")
        return matches[0]

    peak_col = find_column('MaxSlip')
    skew_col = find_column('Skew')
    kurt_col = find_column('Kurtosis')

    # x 범위 설정
    x_min = df[peak_col].min() - margin * (1 + df[skew_col].max())
    x_max = df[peak_col].max() + margin * (1 + df[skew_col].max())
    x = np.linspace(x_min, x_max, 1000)

    # 가중치 설정
    num_distributions = len(df)
    if weights is None:
        weights = [1 / num_distributions] * num_distributions
    else:
        total = sum(weights)
        weights = [w / total for w in weights]
        if len(weights) != num_distributions:
            raise ValueError("weights의 길이는 데이터 개수와 같아야 합니다.")

    # 개별 분포 계산 및 시각화
    individual_pdfs = []
    colors = plt.cm.tab10.colors
    mixture_pdf = np.zeros_like(x)

    for idx in range(num_distributions):
        peak = df.iloc[idx][peak_col]
        skew_val = df.iloc[idx][skew_col]
        kurt_val = df.iloc[idx][kurt_col]

        dist = skewnorm(skew_val, loc=peak, scale=1)
        y = dist.pdf(x) ** (kurt_val / 10)
        individual_pdfs.append(y)
        mixture_pdf += weights[idx] * y

        if plot:
            plt.plot(x, y, label=f'Distribution {idx+1}', color=colors[idx % len(colors)], alpha=0.6)

    # 혼합 분포 그리기
    if plot:
        plt.plot(x, mixture_pdf, label='Mixture Distribution', color='black', linewidth=2)
        plt.title("Mixture of Skewed Distributions with Kurtosis")
        plt.xlabel("Value")
        plt.ylabel("Density (simulated)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    return x, mixture_pdf

def copy_and_sum_lists(A, B):

    result_list = []

    for a_df, b_df in zip(A, B):
        if len(a_df) == 1:
            a_expanded = pd.concat([a_df] * len(b_df), ignore_index=True)
            b_expanded = b_df.reset_index(drop=True)
        elif len(b_df) == 1:
            a_expanded = a_df.reset_index(drop=True)
            b_expanded = pd.concat([b_df] * len(a_df), ignore_index=True)
        else:
            raise ValueError("둘 중 하나는 반드시 1행이어야 합니다.")

        result_df = pd.concat([a_expanded, b_expanded], axis=1)
        result_list.append(result_df)

    return pd.concat(result_list, ignore_index=True)