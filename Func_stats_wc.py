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

# def stats_iqr(series, percentage=25, **kwargs):
#     """
#     Q1 (제1사분위수): 하위 25% / Q3 (제3사분위수): 상위 75%
#     remove 활성화 하면 이상치 상한과 하한으로 cut
#     chname 설정으로 칼럼 이름 변경 가능
#     """
#     remove = kwargs.get('remove', False)
#     chname = kwargs.get('chname', 'series')
#
#     q1 = series.quantile(percentage/100)
#     q3 = series.quantile((1-percentage/100))
#     iqr = q3 - q1
#     lower = q1 - 1.5 * iqr
#     upper = q3 + 1.5 * iqr
#
#     # 시리즈 이름 기반으로 컬럼명 구성
#     prefix = chname
#     result = pd.DataFrame({f'{prefix}_min': [series.min()],
#                            f'{prefix}_max': [series.max()],
#                            f'{prefix}_mean': [series.mean()],
#                            f'{prefix}_q1': [q1],
#                            f'{prefix}_q3': [q3],
#                            f'{prefix}_iqr': [iqr],
#                            f'{prefix}_lower_bound': [lower],
#                            f'{prefix}_upper_bound': [upper]})
#
#     if remove:
#         return result, series[(series >= lower) & (series <= upper)]
#
#     return result
#
# def stats_tdist(series, confidence=95, **kwargs):
#     """
#     t분포 계산
#     chname 설정으로 칼럼 이름 변경 가능
#     """
#     from scipy import stats
#
#     plot = kwargs.get("plot", False)
#     chname = kwargs.get('chname', 'series')
#
#     # series = series[['Dist_0','Dist_1','Dist_2','Dist_3']]
#     n = len(series)
#     df = n - 1 # 자유도
#     mean = np.mean(series)
#     max = np.max(series)
#     min = np.min(series)
#     std_dev = np.std(series, ddof=1)
#
#     # 신뢰수준 설정 (예: 95%)
#     # confidence = 0.95
#     t_crit = stats.t.ppf((1 + (confidence/100)) / 2, df)  # 양측 t-값
#
#     # 표준 오차
#     se = std_dev / np.sqrt(n)
#
#     # 신뢰구간 계산
#     margin_of_error = t_crit * se
#     lower = mean - margin_of_error
#     upper = mean + margin_of_error
#
#     # 시리즈 이름 기반으로 컬럼명 구성
#     prefix = chname
#     result = pd.DataFrame({f'{prefix}_mean': [mean],
#                            f'{prefix}_max': [max],
#                            f'{prefix}_min': [min],
#                            f'{prefix}_df': [df],
#                            f'{prefix}_se': [se],
#                            f'{prefix}_lower': [lower],
#                            f'{prefix}_upper': [upper]})
#
#     if plot:
#         x = np.linspace(mean - 4 * se, mean + 4 * se, 300)
#         y = stats.t.pdf((x - mean) / se, df)  # 중심을 sample mean에 맞추기 위해 변환
#
#         plt.figure(figsize=(8, 4))
#         plt.plot(x, y, label='t-distribution', color='blue')
#         plt.axvline(lower, color='red', linestyle='--', label='95% lower')
#         plt.axvline(upper, color='red', linestyle='--', label='95% upper')
#         plt.axvline(mean, color='green', linestyle=':', label='Sample Mean')
#
#         # 색칠된 신뢰구간 영역
#         x_fill = np.linspace(lower, upper, 200)
#         y_fill = stats.t.pdf((x_fill - mean) / se, df)
#         plt.fill_between(x_fill, y_fill, color='red', alpha=0.2)
#
#         plt.title("t-Distribution with 95% Confidence Interval")
#         plt.xlabel("Value")
#         plt.ylabel("Density")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#
#     return result

# class StatsAnalyzer:
#     def __init__(self, series, chname='series'):
#         """
#         chname 설정으로 칼럼 이름 변경 가능
#         """
#         self.series = series
#         self.chname = chname
#
#     def iqr(self, percentage=25, remove=False):
#         q1 = self.series.quantile(percentage / 100)
#         q3 = self.series.quantile((1 - percentage / 100))
#         iqr = q3 - q1
#         lower = q1 - 1.5 * iqr
#         upper = q3 + 1.5 * iqr
#
#         result = pd.DataFrame({f'{self.chname}_mean': [self.series.mean()],
#                                f'{self.chname}_max': [self.series.max()],
#                                f'{self.chname}_min': [self.series.min()],
#                                f'{self.chname}_q1': [q1],
#                                f'{self.chname}_q3': [q3],
#                                f'{self.chname}_iqr': [iqr],
#                                f'{self.chname}_lower': [lower],
#                                f'{self.chname}_upper': [upper]})
#
#         if remove:
#             return result, self.series[(self.series >= lower) & (self.series <= upper)]
#
#         return result
#
#     def tdist(self, confidence=95, remove=False):
#
#         n = len(self.series)
#         df = n - 1
#         mean = self.series.mean()
#         std_dev = self.series.std(ddof=1)
#         t_crit = stats.t.ppf((1 + confidence / 100) / 2, df)
#         se = std_dev / np.sqrt(n)
#         margin = t_crit * se
#         lower = mean - margin
#         upper = mean + margin
#
#         result = pd.DataFrame({f'{self.chname}_mean': [mean],
#                                f'{self.chname}_max': [self.series.max()],
#                                f'{self.chname}_min': [self.series.min()],
#                                f'{self.chname}_df': [df],
#                                f'{self.chname}_se': [se],
#                                f'{self.chname}_lower': [lower],
#                                f'{self.chname}_upper': [upper]})
#
#         if remove:
#             return result, self.series[(self.series >= lower) & (self.series <= upper)]
#
#         return result
#
#     def zdist(self, confidence=95, remove=False):
#
#         n = len(self.series)
#         mean = self.series.mean()
#         std_dev = self.series.std(ddof=1)
#         z_crit = stats.norm.ppf((1 + confidence / 100) / 2)
#         se = std_dev / np.sqrt(n)
#         margin = z_crit * se
#         lower = mean - margin
#         upper = mean + margin
#
#         result = pd.DataFrame({f'{self.chname}_mean': [mean],
#                                f'{self.chname}_max': [self.series.max()],
#                                f'{self.chname}_min': [self.series.min()],
#                                f'{self.chname}_se': [se],
#                                f'{self.chname}_z_crit': [z_crit],
#                                f'{self.chname}_lower': [lower],
#                                f'{self.chname}_upper': [upper]})
#         if remove:
#             return result, self.series[(self.series >= lower) & (self.series <= upper)]
#
#         return result

class StatsAnalyzer:
    def __init__(self, data, colname, chname=None):
        self.data = data
        self.colname = colname
        self.series = data[colname]
        self.chname = chname if chname else colname

    def iqr(self, percentage=25, remove=False):
        q1 = self.series.quantile(percentage / 100)
        q3 = self.series.quantile((1 - percentage / 100))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        result = pd.DataFrame({f'{self.chname}_smean': [self.series.mean()],
                               f'{self.chname}_smax': [self.series.max()],
                               f'{self.chname}_smin': [self.series.min()],
                               f'{self.chname}_q1': [q1],
                               f'{self.chname}_q3': [q3],
                               f'{self.chname}_iqr': [iqr],
                               f'{self.chname}_slower': [lower],
                               f'{self.chname}_supper': [upper]})

        if remove:
            return result, self.series[(self.series >= lower) & (self.series <= upper)]
        return result

    def tdist(self, confidence=95, remove=False):
        from scipy import stats
        n = len(self.series)
        df = n - 1
        mean = self.series.mean()
        std_dev = self.series.std(ddof=1)
        t_crit = stats.t.ppf((1 + confidence / 100) / 2, df)
        se = std_dev / np.sqrt(n)
        margin = t_crit * se
        lower = mean - margin
        upper = mean + margin

        result = pd.DataFrame({f'{self.chname}_smean': [mean],
                               f'{self.chname}_smax': [self.series.max()],
                               f'{self.chname}_smin': [self.series.min()],
                               f'{self.chname}_df': [df],
                               f'{self.chname}_se': [se],
                               f'{self.chname}_slower': [lower],
                               f'{self.chname}_supper': [upper]})
        if remove:
            return result, self.series[(self.series >= lower) & (self.series <= upper)]

        return result

    def zdist(self, confidence=95, remove=False):
        from scipy import stats
        n = len(self.series)
        mean = self.series.mean()
        std_dev = self.series.std(ddof=1)
        z_crit = stats.norm.ppf((1 + confidence / 100) / 2)
        se = std_dev / np.sqrt(n)
        margin = z_crit * se
        lower = mean - margin
        upper = mean + margin

        result = pd.DataFrame({f'{self.chname}_smean': [mean],
                               f'{self.chname}_smax': [self.series.max()],
                               f'{self.chname}_smin': [self.series.min()],
                               f'{self.chname}_se': [se],
                               f'{self.chname}_z_crit': [z_crit],
                               f'{self.chname}_slower': [lower],
                               f'{self.chname}_supper': [upper]})
        if remove:
            return result, self.series[(self.series >= lower) & (self.series <= upper)]

        return result

    @staticmethod
    def compute_stats(df, group_cols, colname, method='iqr', **kwargs):
        chname = kwargs.pop('chname', method)
        def apply_fn(group):
            analyzer = StatsAnalyzer(group, colname, chname=chname)
            if method == 'iqr':
                stats_df = analyzer.iqr(**kwargs)
            elif method == 'tdist':
                stats_df = analyzer.tdist(**kwargs)
            elif method == 'zdist':
                stats_df = analyzer.zdist(**kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")

            for col in group_cols:
                stats_df[col] = group.iloc[0][col]
            return stats_df

        return df.groupby(group_cols).apply(apply_fn).reset_index(drop=True)