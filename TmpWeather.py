from unoUtil.readDataFile import *
from unoUtil.unoPostProcessing import *
from Func import *
from Func_stats import *
from Func_stats_wc import *
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
import seaborn as sns
import scipy.stats as stats
import itertools

# path 지정
path = r'D:\VehicleTest\Data\2025\0. 기반기술\1. Compd 온도별 평가\WetBraking\Data'
# path = r'C:\Users\HANTA\Desktop\작업\1. Compd 온도별 평가\WetBraking'

# pickle(Performance 데이터), txt(Weather&Road 데이터) 모두 선택
dfraw, fName = ReadData(path)

# 데이터 길이로 구분, 성능 데이터의 Columns이 가장 많은 것으로 보고 판단
# df_tuple = [(df, len(df.columns)) for df in dfraw]
# df_sort = sorted(df_tuple, key=lambda x: x[1], reverse=True)
# df_all, df_weather, df_temperature = df_sort[0][0], df_sort[1][0], df_sort[2][0]
df_weather = dfraw[0]

# 데이터 칼럼 이름 지정 및 Datetime 만들기
df_weather.columns = ['Time','WD','WD_WS','WD_WS_10min',
                      'WS','WS_10min','Temperature','Humidity',
                      'Atpressure','Dewpoint','Rain']
df_weather['Datetime'] = pd.to_datetime(df_weather['Time'], format='%Y%m%d%H%M%S')
df_weather['Year'] = df_weather['Datetime'].dt.year
df_weather['Month'] = df_weather['Datetime'].dt.month
df_weather['Daytime'] = df_weather['Datetime'].dt.strftime('%H')

def remove_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series >= lower) & (series <= upper)]

def remove_outliers_and_stats(group, column):
    clean_series = remove_outliers_iqr(group[column])
    return pd.Series({
        'mean': clean_series.mean(),
        'max': clean_series.max(),
        'min': clean_series.min(),
        'std': clean_series.std()})

def weather_cal(df,col):
    year_stats = (df.groupby(['Year']).
                  apply(remove_outliers_and_stats, column=col).reset_index())
    month_stats = (df.groupby(['Year','Month']).
                  apply(remove_outliers_and_stats, column=col).reset_index())
    # day_stats = (df.groupby(['Daytime']).
    #               apply(remove_outliers_and_stats, column='WS').reset_index())

    time_stats = (df.groupby(['Year', 'Month', 'Daytime']).
                  apply(remove_outliers_and_stats, column=col).reset_index())
    # time_stats_10min = (df.groupby(['Year', 'Month', 'Daytime']).
    #                     apply(remove_outliers_and_stats, column='WS_10min').reset_index())

    sns.set(style="whitegrid")

    ### 1. year_stats: Bar plot + table ###
    plt.figure(figsize=(8, 5))
    sns.barplot(data=year_stats, x='Year', y='mean', palette='viridis')
    plt.title('Yearly Mean Values')
    plt.ylabel('Mean')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.tight_layout()

    ### 2. month_stats: Line plot by year + table ###
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=month_stats, x='Month', y='mean', hue='Year', marker='o', palette='tab10')
    plt.title('Monthly Mean by Year')
    plt.ylabel('Mean')
    plt.xlabel('Month')
    plt.xticks(range(1, 13))
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    ### 3. time_stats: Subplots by Year, lineplot by Month ###
    unique_years = sorted(time_stats['Year'].unique())
    # unique_years = sorted(time_stats['Month'].unique())
    n_years = len(unique_years)
    cols = 3
    rows = (n_years + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    axes = axes.flatten()

    for i, year in enumerate(unique_years):
        ax = axes[i]
        # yearly_data = time_stats[time_stats['Year'] == year]
        yearly_data = time_stats[time_stats['Year'] == year].sort_values(by='Daytime')
        # yearly_data = time_stats[time_stats['Month'] == year].sort_values(by='Daytime')
        sns.lineplot(data=yearly_data, x='Daytime', y='mean', hue='Month', marker='o', ax=ax, palette='tab10')
        # ax.set_title(f'Year: {year}')
        ax.set_title(f'Month: {year}')
        ax.set_xlabel('Daytime')
        ax.set_ylabel('Mean')
        ax.legend(title='Month', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    return year_stats, month_stats, time_stats

WS = weather_cal(df_weather,'WS')
WS_10min = weather_cal(df_weather,'WS_10min')
df_select = df_weather[(df_weather['Month'] > 4) & (df_weather['Month'] < 9)]
WD = weather_cal(df_weather,'WD')
WD_select = weather_cal(df_select,'WD')
plt.figure()