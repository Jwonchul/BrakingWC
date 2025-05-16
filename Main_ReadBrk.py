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

# Performance Index
df_total = []
for name, grp in df.groupby(['TestItem','ReqNo','RoadInfo']):
    # 추후 Control 타이어의 그룹이 변경되는 경우에는 새로 코드 작성 필요
    std_row = grp[grp['GroupSpec'] == 'SRTT'].iloc[0]
    # grp_cols = grp.select_dtypes(include='number').columns
    grp_cols = ['Dist_mean', 'perDist', 'meanAx', 'maxAx', 'minAx', 'rmsAx']
    # grp_calculated = ((grp[grp_cols] / std_row[grp_cols].astype(float))*100).round(2)
    grp_calculated = ((std_row[grp_cols].astype(float) / grp[grp_cols]) * 100).round(2)
    grp_calculated.columns = ['Per_' + str(col) for col in grp_cols]
    grp_total = pd.concat([grp, grp_calculated], axis=1)
    grp_total['Dist_srtt'] = std_row[grp_cols]['Dist_mean'].astype(float)
    df_total.append(grp_total)
df_first = pd.concat(df_total, ignore_index=True)

# Ax 변화가 심한 경우 제외
df_first['CompareAx'] = abs(1-df_first['meanAx']/df_first['rmsAx'])*100
# df_final = df_first[abs(df_first['CompareAx'])<2.5] # 조건 대입
df_final = df_first # 모든 조건

# 변동계수 구하기
df_CV = df_final.copy()
df_CV['CV'] = (df_CV['Dist_std']/df_CV['Dist_mean'])*100
df_CV['CV_mean'] = df_CV['CV'].mean()*2

# Data excel save
file_path = r'D:\VehicleTest\Data\2025\0. 기반기술\1. Compd 온도별 평가\WetBraking\Data\dfall.xlsx'
df_final.to_excel(file_path, index=False)

# Road 구분
spec_list = sorted(df_final['RoadInfo'].unique())
n = len(spec_list)
for idx, spec_val in enumerate(spec_list):
    df_idx = df_final[df_final['RoadInfo'] == spec_val]
    plt_group(df_idx, ['GroupSpec','day','AMPM'], 'AMPM', 'WHC', 'Per_Dist_mean')
    plt.suptitle(f'{spec_val}', fontsize=16)
    plt.tight_layout()
    plt_group(df_idx, ['GroupSpec','day','AMPM'], 'AMPM', 'WHC', 'Dist_mean',z='Dist_srtt')
    plt.suptitle(f'{spec_val}', fontsize=16)
    plt.tight_layout()
    plt_group(df_idx, ['GroupSpec','day','AMPM'], 'GroupSpec', 'WHC', 'Dist_mean',z='Dist_srtt')
    plt.suptitle(f'{spec_val}', fontsize=16)
    plt.tight_layout()

plt_group(df_final, ['RoadInfo','GroupSpec','AMPM'], 'RoadInfo', 'WHC', 'Dist_mean')
plt_group(df_final, ['RoadInfo','GroupSpec','AMPM'], 'GroupSpec', 'WHC', 'Dist_mean')
plt_group(df_final, ['RoadInfo','AMPM'], 'AMPM', 'WHC', 'Dist_mean')
plt_group_multi(df_final,['RoadInfo', 'GroupSpec'],'WHC','Dist_mean',text='AMPM')

# SRTT 경향성
df_srtt = df_final[df_final['GroupSpec']=='SRTT']
plt_group_multi(df_srtt, ['RoadInfo','AMPM'],'WHC','Dist_mean',text='day')

group_cols = ['RoadInfo', 'day', 'TestSet']
fig, ax = plt.subplots(figsize=(10, 6))
for (road, spec, Day), grp in df_srtt.groupby(group_cols):
    # 라벨은 보기 좋게 포맷
    label = f"{road} - {spec} - {Day}"
    # 선 그래프
    ax.plot(grp['WHC'],grp['Dist_mean'],marker='o',linestyle='',
            label=label)  # matplotlib이 자동 색상 순환

    for x, y, ampm in zip(grp['WHC'], grp['Dist_mean'], grp['AMPM']):
        ax.text( x, y, label+'/'+ampm, va='bottom', ha='center', fontsize=8,clip_on=False)

ax.set_title("SRTT")
ax.set_xlabel('WHC')
ax.set_ylabel('Dist_mean')
ax.legend(loc="best")
ax.grid(True, linestyle="--", alpha=0.4)

df_spec = df_final[df_final['GroupSpec']!='SRTT']

## 제동거리 분포 계산
group_cols = ['GroupSpec','RoadInfo','Dist_mean']
colname = 'Dist_detail'
stats_t = StatsAnalyzer.compute_stats(df_spec, group_cols, colname, method='tdist', confidence=95)
df_t = df_spec.merge(stats_t, on=group_cols, how='left')

# stats_iqr = StatsAnalyzer.compute_stats(df_spec, group_cols, colname, method='iqr', percentage=25) iqr분포
# df_iqr = df_spec.merge(stats_iqr, on=group_cols, how='left')
#
# stats_z = StatsAnalyzer.compute_stats(df_spec, group_cols, colname, method='zdist', confidence=95) z분포
# df_z = df_spec.merge(stats_z, on=group_cols, how='left')

##
# df_org = df_final.copy()
# df_final = df_final.drop_duplicates(subset='Dist_mean', keep='first')
# df_change_value, df_change_per = plt_change_v2(df_final,'GroupSpec','RoadInfo',
#                                                numdist=4, x='Dist_srtt',y='Dist_mean',plot=True)
#
# # df_change_value.to_excel(r'D:\VehicleTest\Data\2025\0. 기반기술\1. Compd 온도별 평가\WetBraking\Data\dfchangevalue.xlsx',
# #                          index=False)
# # df_change_per.to_excel(r'D:\VehicleTest\Data\2025\0. 기반기술\1. Compd 온도별 평가\WetBraking\Data\dfchangeper.xlsx',
# #                        index=False)
#
# df_change_per['CV_mean'] = df_CV['CV'].mean()*2
# df_r = df_change_per[df_change_per['Dist_mean']>df_CV['CV'].mean()*2]
# df_f = df_change_per[df_change_per['Dist_mean']>df_CV['CV'].mean()]
#
# # df_tmp2 = df_change_per[abs(df_change_per['cohens_d'])>0.8]
# df_pvalue = df_change_value[abs(df_change_value['pvalue'])<0.05]
##

##

grouped = df_srtt.groupby('Dist_mean')
group_values = [group['Dist_detail'].values for _, group in grouped]
group_labels = list(grouped.groups.keys())
min_samples_per_group = min(len(g) for g in group_values)
n_groups = len(group_values)

# 정규성 & 등분산성 검정 함수
def check_assumptions(groups):
    # 정규성: Shapiro-Wilk
    normal = all(stats.shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)

    # 등분산성: Levene
    if len(groups) > 1:
        levene_p = stats.levene(*groups).pvalue
        equal_var = levene_p > 0.05
    else:
        equal_var = True  # 의미 없음

    return normal, equal_var

# 검정 선택
if n_groups == 2:
    group1, group2 = group_values
    normal, equal_var = check_assumptions([group1, group2])

    if min(len(group1), len(group2)) < 5 or not (normal and equal_var):
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        method = "Mann-Whitney U test"
    else:
        stat, p = stats.ttest_ind(group1, group2, equal_var=equal_var)
        method = "Independent t-test"

elif n_groups >= 3:
    normal, equal_var = check_assumptions(group_values)

    if min_samples_per_group < 5 or not (normal and equal_var):
        stat, p = stats.kruskal(*group_values)
        method = "Kruskal-Wallis test"
    else:
        stat, p = stats.f_oneway(*group_values)
        method = "One-way ANOVA"

else:
    raise ValueError("그룹 수가 2개 이상이어야 분석이 가능합니다.")

# 아래 수정 필요
first_key = group_labels[0]
first_group_index = grouped.get_group(first_key).index[0]

df_srtt['p_value'] = None
df_srtt.loc[:, 'p_value'] = df_srtt['Dist_srtt'].map(lambda x: p)
##

df_org = df_final.copy()
df_final = df_final.drop_duplicates(subset='Dist_mean', keep='first')
plt_group_multi(df_final, ['RoadInfo','day','AMPM'],'WHC','Dist_mean',text='day')
plt_group_multi(df_final,['RoadInfo', 'GroupSpec', 'day'],'WHC','Dist_mean',text='AMPM')
plt_group_multi(df_final,['RoadInfo', 'GroupSpec', 'day'],'WHC','Per_Dist_mean',text='AMPM')
plt_group_multi(df_final,['RoadInfo', 'GroupSpec'],'Front_Kurtosis','Dist_mean',text='AMPM')
##
# distselect =df_final[['Dist_0','Dist_1','Dist_2','Dist_3']]
# tdist = stats_tdist(distselect.loc[0], confidence=95, plot=True,chname='tdist')
# iqrdata = stats_iqr(distselect.loc[0], percentage=25,remove=True,chname='iqr')

## Spec 경향성
plt_stats_bar_v2(df_t,'GroupSpec','RoadInfo','Dist_mean',st='AMPM',num='WHC',numunit='°C')
plt_stats_bar_v2(df_t,'GroupSpec','RoadInfo','Dist_mean',st='AMPM',num='WHC',numunit='°C')
plt_stats_bar_v2(df_t,'GroupSpec','RoadInfo','Dist_mean',st='GroupSpec',num='Per_Dist_mean',numunit='%')
plt_group_multi(df_spec,['RoadInfo', 'GroupSpec', 'day'],'WHC','Dist_mean',text='AMPM')
# plt_stats_bar(df_final,'RoadInfo','WaterDepth',x=select_chname,st='GroupSpec',num='WHC',numunit='°C')

## SRTT 경향성
plt_stats_bar(df_srtt,'GroupSpec','RoadInfo',x=select_chname,st='AMPM')
plt_stats_bar(df_srtt,'day','RoadInfo',x=select_chname,st='TestSet')
plt_stats_bar(df_srtt,'TestSet','RoadInfo',x=select_chname,st='AMPM',num='WHC',numunit='°C')
# plt_stats_bar(df_srtt,'TestSet','RoadInfo',x=select_chname,st='day',num='WHC',numunit='°C')

# ## Slip 경향성
plt_group_multi(df_srtt,['RoadInfo', 'TestSet'],'Front_Kurtosis','Dist_mean',text='AMPM')
plt_group_multi(df_srtt,['RoadInfo', 'TestSet'],'Front_MaxSlip','Dist_mean',text='AMPM')
plt_group_multi(df_srtt,['RoadInfo', 'TestSet'],'Rear_Kurtosis','Dist_mean',text='AMPM')
plt_group_multi(df_srtt,['RoadInfo', 'TestSet'],'Rear_MaxSlip','Dist_mean',text='AMPM')

plt_group_multi(df_srtt,['RoadInfo', 'TestSet'],'Front_Kurtosis','Front_MaxSlip',text='AMPM')
plt_group_multi(df_srtt,['RoadInfo', 'TestSet'],'Rear_Kurtosis','Rear_MaxSlip',text='AMPM')
plt_group_multi(df_srtt,['RoadInfo', 'TestSet'],'Front_Kurtosis','Rear_Kurtosis',text='AMPM')
plt_group_multi(df_srtt,['RoadInfo', 'TestSet'],'Front_MaxSlip','Rear_MaxSlip',text='AMPM')
plt_group_multi(df_srtt,['RoadInfo', 'TestSet'],'Front_Kurtosis','Rear_Kurtosis',text='Dist_mean')


#
Fslip_chname = ['Front_MaxSlip','Front_Kurtosis','Front_Skew']
plt_group_slipstats(df_srtt,['RoadInfo', 'TestSet'],Fslip_chname)

Rslip_chname = ['Rear_MaxSlip','Rear_Kurtosis','Rear_Skew']
plt_group_slipstats(df_srtt,['RoadInfo', 'TestSet'],Rslip_chname)

# Graph Save
file_name = os.path.basename(fName[0])
folder_name = os.path.splitext(file_name)[0]
folder_path = os.path.join(os.path.dirname(fName[0]), folder_name)
os.makedirs(folder_path, exist_ok=True)
for fig_num in plt.get_fignums():  # 현재 생성된 모든 figure 번호 가져오기
    fig = plt.figure(fig_num)  # 해당 figure 불러오기
    axes = fig.get_axes()

    fig_title = f"figure_{fig_num}"

    # 파일 이름에 사용할 수 없는 문자 제거 (예: 파일명 오류 방지)
    fig_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in fig_title)

    # 저장 경로 설정
    save_path = os.path.join(folder_path, f"{fig_title}.png")
    fig.savefig(save_path)  # 그래프 저장
    print(f"Saved: {save_path}")
plt.close('all')

## 전체 편차 구하기
df_tdist = pd.concat([stats_tdist(row[['Dist_0', 'Dist_1', 'Dist_2', 'Dist_3']], chname='tdist')
                      for _, row in df_final.iterrows()], ignore_index=True)
means = df_tdist['tdist_mean'].values
ses = df_tdist['tdist_se'].values

between_var = np.var(means, ddof=1)
within_var = np.mean(ses ** 2)
total_var = between_var + within_var
total_std = np.sqrt(total_var)

# plt.figure()
# plt.plot(df_asp['perDist'],marker='o',linestyle='',color = 'red')
# plt.plot(df_con['perDist'],marker='o',linestyle='',color = 'blue')
# plt.title('Road & P.hydro')
# plt.ylabel('perDist')
#
# plt.figure()
# plt.plot(df_srtt['WHC'],df_srtt['Dist_mean'],marker='o',linestyle='',color = 'red')



# fig, ax = plt.subplots(figsize=(10, 6))
# for (road, spec, Day), grp in df_srtt.groupby(group_cols):
#     # 라벨은 보기 좋게 포맷
#     label = f"{road} - {spec} - {Day}"
#     # 선 그래프
#     ax.plot(grp['WS'],grp['Dist_mean'],marker='o',linestyle='',
#             label=label)  # matplotlib이 자동 색상 순환
#
#     for x, y, ampm in zip(grp['WS'], grp['Dist_mean'], grp['AMPM']):
#         ax.text( x, y, ampm, va='bottom', ha='center', fontsize=8,clip_on=True)
#
# ax.set_title("SRTT")
# ax.set_xlabel('WS')
# ax.set_ylabel('Dist_mean')
# ax.legend(loc="best")
# ax.grid(True, linestyle="--", alpha=0.4)

# group_cols = ["RoadInfo", "GroupSpec", "day"]
# spec = 'GroupSpec'
# xdata = 'TempSEP'
# ydata = 'Dist_mean'

# plt_group(df,group_cols,spec,xdata,ydata)

df_select = df_final[abs(df_final['Dist_range'])>1]

plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],'GroupSpec','BRK Dry Asp','Dist_mean')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],"RoadInfo",'BRK Dry Asp','Dist_mean')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],"RoadInfo",'BRK Dry Asp','Per_Dist_mean')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],"RoadInfo",'WHC','Per_Dist_mean')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day','AMPM'],"RoadInfo",'WHC','Per_Dist_mean')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],"RoadInfo",'WHC','Dist_mean')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],"RoadInfo",'WS','Dist_mean')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],"RoadInfo",'WS','Dist_std')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],"RoadInfo",'WHC','Dist_std')

df_std = df_final[abs(df_final['Dist_std'])<0.5]
plt_group(df_std,['RoadInfo', 'GroupSpec', 'day'],"RoadInfo",'WHC','Dist_mean')
plt_group(df_std,['RoadInfo', 'GroupSpec', 'day'],"RoadInfo",'WHC','Per_Dist_mean')

plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],'GroupSpec','Temperature','Dist_mean')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],'GroupSpec','WHC','Dist_mean')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],'GroupSpec','WHC','Per_Dist_mean')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],'GroupSpec','WHC','Dist_mean',z='Dist_srtt')
plt_group(df_final,['RoadInfo', 'GroupSpec', 'day'],'GroupSpec','BRK Dry Asp','Dist_mean',z='Dist_srtt')

df_change = plt_change(df_final,'GroupSpec','RoadInfo',x='WHC',y='Dist_mean',plot=True)
df_change = plt_change(df_final,'GroupSpec','RoadInfo',x='Dist_srtt',y='Dist_mean',plot=True)
df_change = plt_change(df_final,'GroupSpec','RoadInfo',x='WS',y='Dist_mean',plot=True)
# df_22 = plt_change(df_final,'GroupSpec','RoadInfo',x='WHC',y='Per_Dist_mean',plot=True)
df_select = df_change[abs(df_change['Dist_mean'])>1]
plt_group(df_select,['RoadInfo', 'GroupSpec'],"RoadInfo",'Per_Dist_mean','Dist_mean')
tmp = df[(df["RoadInfo"]=='A1C60') & (df['GroupSpec']=='H')]

df_tmp = df_change.select_dtypes(include='number')
df_sign = np.sign(df_tmp)



# plt_change(df,colorgroup,group,xcol,ycol):

## Con, Asp perDist & DistMean
df_asp = df_final[df_final['TestItem']=='wab']
df_con = df_final[df_final['TestItem']=='con']
plt.figure()
plt.subplot(221)
plt.plot(df_total['perDist'],df_total['Dist_mean'],marker='o',linestyle='',color = 'black')

plt.subplot(222)
plt.plot(df_asp['perDist'],df_asp['Dist_mean'],marker='o',linestyle='',color = 'red')
plt.plot(df_con['perDist'],df_con['Dist_mean'],marker='s',linestyle='',color = 'blue')

plt.subplot(223)
plt.plot(df_asp['perDist'],marker='o',linestyle='',color = 'red')
plt.plot(df_con['perDist'],marker='s',linestyle='',color = 'blue')

plt.subplot(224)
plt.plot(df_asp['Dist_mean'],marker='o',linestyle='',color = 'red')
plt.plot(df_con['Dist_mean'],marker='s',linestyle='',color = 'blue')

color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.figure(figsize=(10, 8))
# Subplot 221 : 전체
ax1 = plt.subplot(221)
ax1.plot(df_total['perDist'], df_total['Dist_mean'],
         marker='o', linestyle='', color='black', label='All')
ax1.set_title('All')
ax2 = plt.subplot(222)

for (gname, gdf), marker in zip(df_asp.groupby('GroupSpec'), itertools.cycle(['o', '^', 'v', 'D'])):
    ax2.plot(gdf['perDist'], gdf['Dist_mean'],
             marker=marker, linestyle='',
             color=next(color_cycle),
             label=f'asp-{gname}')
ax2.set_title('Dist_mean vs perDist (Asp)')
ax2.legend(fontsize=8)

ax3 = plt.subplot(223)

for (gname, gdf), marker in zip(df_con.groupby('GroupSpec'), itertools.cycle(['s', 'P', 'X', '*'])):
    ax3.plot(gdf['perDist'], gdf['Dist_mean'],
             marker=marker, linestyle='',
             color=next(color_cycle),
             label=f'con-{gname}')
ax3.set_title('Dist_mean vs perDist (Con)')
ax3.legend(fontsize=8)