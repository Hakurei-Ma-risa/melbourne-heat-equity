"""
04_spatial.py — 空间自相关与空间回归（严谨性补强）
==================================================
1) Moran's I：热暴露是否空间聚集（预期是，证明 SUHI 结构）
2) OLS：LST ~ IRSAD + NDVI + NDBI（标准化系数），看 SES 独立效应
3) 残差 Moran's I：是否残留空间自相关 → 是否需要空间模型
4) 空间误差模型：控制空间依赖后 IRSAD 系数是否仍显著（核心稳健性检验）
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from libpysal.weights import Queen
from esda.moran import Moran
import statsmodels.api as sm

g = gpd.read_file('SA2_GreaterMelbourne.geojson'); g['SA2_CODE21'] = g.SA2_CODE21.astype(str)
d = pd.read_csv('heat_equity_results.csv'); d['SA2_CODE21'] = d.SA2_CODE21.astype(str)
g = g.merge(d, on='SA2_CODE21').dropna(subset=['LST', 'IRSAD_score', 'NDVI', 'NDBI']).reset_index(drop=True)
print(f'分析 SA2: {len(g)}')

w = Queen.from_dataframe(g, use_index=False); w.transform = 'r'

# 1) 热暴露空间聚集
mi = Moran(g['LST'].values, w)
print(f"\n[1] Moran's I 热暴露(LST): I={mi.I:.3f}, p={mi.p_sim:.3g}  (强聚集=SUHI空间结构)")

# 2) OLS 标准化系数
feats = ['IRSAD_score', 'NDVI', 'NDBI']
Xz = StandardScaler().fit_transform(g[feats].values)
y = g['LST'].values
ols = sm.OLS(y, sm.add_constant(Xz)).fit()
print('\n[2] OLS 标准化系数 (LST ~ IRSAD+NDVI+NDBI):')
for name, b, p in zip(['const'] + feats, ols.params, ols.pvalues):
    print(f'    {name:12} β={b:+.3f}  p={p:.2e}')
print(f'    R²={ols.rsquared:.3f}')

# 3) 残差空间自相关
mres = Moran(ols.resid, w)
print(f"\n[3] Moran's I OLS残差: I={mres.I:.3f}, p={mres.p_sim:.3g}  (>0且显著 → 需空间模型)")

# 4) 空间误差模型（控制空间依赖）
try:
    from spreg import GM_Error_Het
    m = GM_Error_Het(y.reshape(-1, 1), Xz, w=w, name_x=feats, name_y='LST')
    print('\n[4] 空间误差模型 (GM_Error_Het) 系数:')
    for name, b, p in zip(m.name_x, m.betas.flatten(), [v[1] for v in m.z_stat]):
        print(f'    {name:12} β={b:+.3f}  p={p:.2e}')
    irsad_p = dict(zip(m.name_x, [v[1] for v in m.z_stat])).get('IRSAD_score', float('nan'))
    if irsad_p < 0.05:
        print('    → 控制空间自相关后 IRSAD 仍显著 = 独立 SES 效应稳健')
    else:
        print(f'    → 控制空间自相关后 IRSAD 不再显著 (p={irsad_p:.2f}) = 独立 SES 效应不稳健；'
              '热不公平主要经由绿/灰基础设施与空间结构传导')
except Exception as e:
    print('\n[4] 空间模型跳过:', e)
