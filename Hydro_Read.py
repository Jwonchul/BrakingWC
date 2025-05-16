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
path = r'D:\VehicleTest\Data\2025\0. 기반기술\1. Compd 온도별 평가\Hydroplaning\Data'

# pickle(Performance 데이터), txt(Weather&Road 데이터) 모두 선택
dfraw, fName = ReadData(path)

# 데이터 길이로 구분, 성능 데이터의 Columns이 가장 많은 것으로 보고 판단
df_tuple = [(df, len(df.columns)) for df in dfraw]
df_sort = sorted(df_tuple, key=lambda x: x[1], reverse=True)
df_all, df_weather, df_temperature = df_sort[0][0], df_sort[1][0], df_sort[2][0]

# 데이터 칼럼 이름 지정 및 Datetime 만들기
df_weather.columns = ['Time','WD','WD_WS','WD_WS_10min',
                      'WS','WS_10min','Temperature','Humidity',
                      'Atpressure','Dewpoint','Rain']
df_weather['Datetime'] = pd.to_datetime(df_weather['Time'])

df_temperature.columns = ['num','day','Time','Sensor','Road_Temperature']
df_temperature['Datetime'] = pd.to_datetime(df_temperature['day'] + ' ' + df_temperature['Time'],format='%Y.%m.%d %H.%M.%S')

# 노면 온도를 칼럼으로 변경하여 시간 독립적으로 처리
temp_pivot = df_temperature.pivot_table(index=['Datetime'],
                                        columns='Sensor',values='Road_Temperature',
                                        aggfunc='first').reset_index()
temp_pivot.columns.name = None
temp_pivot = temp_pivot.rename(columns=lambda x: f"{x}" if x not in ['Datetime'] else x)

# 기상과 노면 데이터 동기화 (환경 데이터 만들기)
df_condition = pd.merge_asof(df_weather.sort_values('Datetime'),temp_pivot,
                         on='Datetime',direction='nearest',tolerance=pd.Timedelta('20min'))
df_condition = df_condition.sort_values('Datetime').reset_index(drop=True)
df_condition = df_condition.drop(['Time'], axis=1)

## Datetime type확인해서 변경
for df in [df_all, df_condition]:
    if df['Datetime'].dtype != 'datetime64[ns]':
        df['Datetime'] = pd.to_datetime(df['Datetime'])

# 성능과 환경 데이터 동기화
df_total = pd.merge_asof(df_all.sort_values('Datetime'),df_condition,
                         on='Datetime',direction='nearest',tolerance=pd.Timedelta('20min'))
# 데이터 숫자화
df = df_total.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.dtype == 'object' else col)
for col in df_total.columns:
    if col in df.columns:
        if df[col].isna().sum() > 0 and df_total[col].dtype == 'object':
            df[col] = df_total[col]
df["day"] = df["Datetime"].dt.date

df_final = df

spec_list = sorted(df_final['RoadInfo'].unique())
n = len(spec_list)
for idx, spec_val in enumerate(spec_list):
    df_idx = df_final[df_final['RoadInfo'] == spec_val]
    plt_group(df_idx, ['GroupSpec','day','AMPM'], 'AMPM', 'WHC', 'Area')
    plt.suptitle(f'{spec_val}', fontsize=16)
    plt.tight_layout()
    plt_group(df_idx, ['GroupSpec','day','AMPM'], 'AMPM', 'WHC', 'MaxG')
    plt.suptitle(f'{spec_val}', fontsize=16)
    plt.tight_layout()
    plt_group(df_idx, ['GroupSpec','day','AMPM'], 'GroupSpec', 'WHC', 'Dist_mean',z='Dist_srtt')
    plt.suptitle(f'{spec_val}', fontsize=16)
    plt.tight_layout()

plt_group_multi(df_final,['GroupSpec','AMPM'],'WHC','Area')
plt_group_multi(df_final,['RoadInfo','AMPM'],'WHC','Area')
df_change_value, df_change_per = plt_change_v2(df_final,'GroupSpec','RoadInfo',
                                               numdist=4, x='WHC',y='Area',plot=True)
