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