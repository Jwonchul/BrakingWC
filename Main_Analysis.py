import matplotlib.pyplot as plt
from unoUtil.readDataFile.ReadDataFile import ReadDataFile
from unoUtil.vehicleTest._BRKTest_ import *
from unoUtil.unoPostProcessing import SP_MovingAverage
import os
from scipy.stats import skew, kurtosis
import re
import pickle
from Func_stats import *
from scipy import stats

path = r'D:\VehicleTest\Data\2025\0. 기반기술\1. Compd 온도별 평가\WetBraking'
# path = r'C:\Users\HANTA\Desktop\작업\1. Compd 온도별 평가\WetBraking'

# rawdf, fName = ReadDataFile(r'example_BRK_80-5_w4Slip.vbo', date=True)
rawdf, fName = ReadDataFile(path, filter='*.vbo', date=True, headendline=300)
all_results = []
detail_results = []
for k in range(len(rawdf)):
# 파일명에 따라 폴더 만들고 저장
    file_name = os.path.basename(fName[k])
    folder_name = os.path.splitext(file_name)[0]
    folder_path = os.path.join(os.path.dirname(fName[k]), folder_name)
    # os.makedirs(folder_path, exist_ok=True)

    # 파일명에서 데이터 끌어오기 (파일명에 따라서 변경 가능)
    parts = [part for part in folder_name.replace("-", "_").split("_")]
    # parts.append(parts[4].split('T')[1])
    parts.append(re.search(r'W(\d+)T(\d+)', parts[4]).group(1))
    parts.append(re.search(r'W(\d+)T(\d+)', parts[4]).group(2))
    columns = ['TestItem', 'ReqNo', 'TestSet', 'RoadInfo', 'CondInfo', 'AMPM', 'GroupSpec', 'WaterDepth', 'TempSEP']
    df_spec = pd.DataFrame([parts], columns=columns).reset_index(drop=True)

    # raw = rawdf[0][1]
    # df = AN_ACCBRK(rawdf[k][1], rawdf[k][3], slip=True, target= [80, 5], plot=True)
    df = AN_ACCBRK(rawdf[k][1], rawdf[k][3], slip=True, target= [80, 20], valid=4, plot=True)
    df["filterAcc"] = SP_MovingAverage(df[["AccForward"]], ws=30, plot=True).values
    df["dist"] = df["VelHorizontal"] / 3.6 / rawdf[0][3]

    # BrakingDist
    # dfbrk = df.loc[~df["testNum"].isna()] # 전체 제동거리 선택
    dfbrk = df.loc[~df["effNum"].isna()] # 선택한 제동거리 선택
    ## df_dist = (dfbrk.groupby("testNum")["dist"].sum().agg(["max", "min", "mean", "std"]).rename(index=lambda x: f"Dist_{x}").to_frame().T.reset_index(drop=True))
    # df_dist_sum = (dfbrk.groupby("testNum")["dist"].sum().reset_index(drop=True).
    #                rename(lambda x: f"Dist_{x}").to_frame().T.reset_index(drop=True))
    df_dist_detail = (dfbrk.groupby("testNum")["dist"].sum().reset_index(name="Dist_detail"))
    df_dist = (dfbrk.groupby("testNum")["dist"].sum().agg(["max", "min", "mean", "std"])
                     .rename(index=lambda x: f"Dist_{x}").to_frame().T.reset_index(drop=True))
    # df_dist = pd.concat([df_dist_sum, df_dist_stats], axis=1)

    ## GPS Position check
    df_dist['GPS_Sum']=dfbrk['GPS_Sats'].mean()
    df_dist['GPS_Max']=dfbrk['GPS_Sats'].max()
    df_dist['GPS_Min']=dfbrk['GPS_Sats'].min()

    # Longitudinal Acc
    dfbrk = dfbrk.copy()
    dfbrk['VelHorizontal_bin'] = pd.cut(dfbrk['VelHorizontal'], bins=range(0, 81, 10), right=False)
    df_acc = dfbrk.groupby('VelHorizontal_bin', observed=True)['filterAcc'].mean().to_frame().T.reset_index(drop=True)

    # Datetime
    mean_ts = dfbrk['Datetime'].mean()
    df_Datetime = pd.Series(mean_ts, name='Datetime')

    ## 추후 Wheel Speed 데이터가 없는 경우를 고려하여 if문 처리 필요한 부분
    # Partial Hydro
    perCol = ["VelHorizontal", "AccForward"]
    plotCol = ["waFL", "waFR"]
    pCol = [perCol + [x] for x in plotCol]
    pList = [dfbrk[col].rename(columns={col[2]: "wa"}) for col in pCol]
    prePartialdf = pd.concat(pList, axis=0).reset_index(drop=True)
    partialdf, cp, partialResult = AN_PartialHydro(prePartialdf[["AccForward", "wa", "VelHorizontal"]],
                                                      out=True, plot=True, supTitle='Partial')
    df_partial = partialResult.reset_index(drop=True)

    # SlipGraph
    dfslip = AN_ACCBRK(rawdf[k][1], rawdf[k][3], slip=True, target= [80, 10], valid=4, plot=True)
    dfslip["filterAcc"] = SP_MovingAverage(dfslip[["AccForward"]], ws=50, plot=True).values
    dfslip = dfslip.loc[~dfslip["effNum"].isna()] # 선택한 제동거리 선택

    front_dict = slip_stats(dfslip, "srFL", "srFR", "Front")
    rear_dict  = slip_stats(dfslip, "srRL", "srRR", "Rear")
    df_slip = pd.DataFrame([{**front_dict, **rear_dict}])

    # df_result = pd.concat([df_spec, df_dist,df_acc,df_Datetime], axis=1)
    ## 추후 Wheel Speed 데이터가 없는 경우를 고려하여 if문 처리 필요한 부분

    # Total Result
    df_result = pd.concat([df_spec, df_dist,df_acc,df_Datetime,df_partial,df_slip], axis=1)
    all_results.append(df_result)

    detail_results.append(df_dist_detail)
    # # Graph Save
    # for fig_num in plt.get_fignums():  # 현재 생성된 모든 figure 번호 가져오기
    #     fig = plt.figure(fig_num)  # 해당 figure 불러오기
    #     axes = fig.get_axes()
    #
    #     fig_title = f"figure_{fig_num}"  # 아무 제목도 없으면 figure 번호 사용
    #
    #     # 파일 이름에 사용할 수 없는 문자 제거 (예: 파일명 오류 방지)
    #     fig_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in fig_title)
    #
    #     # 저장 경로 설정
    #     save_path = os.path.join(folder_path, f"{fig_title}.png")
    #     fig.savefig(save_path)  # 그래프 저장
    #     print(f"Saved: {save_path}")

    # plt.close('all')
    print(f"Analysis End: {file_name}")

df_all = copy_and_sum_lists(all_results, detail_results)
# df_all = pd.concat(all_results, ignore_index=True)

# df Save
save_pickle = os.path.join(os.path.dirname(fName[0]), "Brk_data.pkl")
with open(save_pickle, "wb") as f:
    pickle.dump(df_all, f)

df_all['Dist_range']=df_all['Dist_max']-df_all['Dist_min']
df_all['Dist_range']=df_all['Dist_max']-df_all['Dist_min']
## ANOVA
df_stat = df_all[['Dist_0','Dist_1','Dist_2','Dist_3']]
f_stat, p_value = stats.f_oneway(*df_stat.values)
if p_value < 0.05:
    print("그룹 간 변동계수 차이가 유의미합니다.")
else:
    print("그룹 간 변동계수 차이가 유의미하지 않습니다.")
# t2 = tt.sort_values(by='GPS_Sum', ascending=False)
# a = tt[tt['GPS_Min']<6]
# cond1 = tt[tt['GPS_Sum']<8.2]
# cond2 = tt[tt['GPS_Sum']>8.5]