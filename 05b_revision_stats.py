"""05b — 审稿整改的稳健性分析：M2 地形控制 / M4 中介 / M5 空间交叉验证。"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, geopandas as gpd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, GroupKFold
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from libpysal.weights import Queen
from spreg import GM_Error_Het, OLS as spreg_OLS

g = gpd.read_file('SA2_GreaterMelbourne.geojson'); g['SA2_CODE21'] = g.SA2_CODE21.astype(str)
d = pd.read_csv('heat_equity_results.csv'); d['SA2_CODE21'] = d.SA2_CODE21.astype(str)
e = pd.read_csv('sa2_elevation.csv'); e['SA2_CODE21'] = e.SA2_CODE21.astype(str)
g = g.merge(d, on='SA2_CODE21').merge(e, on='SA2_CODE21')
g = g.dropna(subset=['LST', 'IRSAD_score', 'NDVI', 'NDBI', 'elevation']).reset_index(drop=True)
gm = g.to_crs(7855); g['x'] = gm.geometry.centroid.x; g['y'] = gm.geometry.centroid.y
print('n =', len(g))

def resid(y, X): return y - LinearRegression().fit(X, y).predict(X)

# ===== M2a 控制海拔后的偏相关 =====
Xe = g[['elevation']].values
r_raw, _ = stats.pearsonr(g.LST, g.IRSAD_score)
r_el, p_el = stats.pearsonr(resid(g.LST.values, Xe), resid(g.IRSAD_score.values, Xe))
print(f'\n[M2a] IRSAD–LST: raw r={r_raw:+.3f}; 控制海拔后 partial r={r_el:+.3f} (p={p_el:.1e})')
print(f'      海拔 vs IRSAD r={g.elevation.corr(g.IRSAD_score):+.2f}; 海拔 vs LST r={g.elevation.corr(g.LST):+.2f}')

# ===== M2b 建成区子集(剔除山区/森林) =====
urb = g[(g.elevation < 200) & (g.NDVI < g.NDVI.quantile(0.85))].copy()
ru, pu = stats.pearsonr(urb.LST, urb.IRSAD_score)
rp, pp = stats.pearsonr(resid(urb.LST.values, urb[['NDVI', 'NDBI']].values),
                        resid(urb.IRSAD_score.values, urb[['NDVI', 'NDBI']].values))
print(f'[M2b] 建成区子集 n={len(urb)}: IRSAD–LST r={ru:+.3f} (p={pu:.1e}); 偏相关(控NDVI/NDBI) r={rp:+.3f} (p={pp:.1e})')

# ===== M2c OLS + 空间误差模型(加入海拔) =====
feats = ['IRSAD_score', 'NDVI', 'NDBI', 'elevation']
Xz = StandardScaler().fit_transform(g[feats].values); y = g.LST.values
w = Queen.from_dataframe(g, use_index=False); w.transform = 'r'
m = GM_Error_Het(y.reshape(-1, 1), Xz, w=w, name_x=feats)
b = dict(zip(m.name_x, m.betas.flatten())); p = dict(zip(m.name_x, [v[1] for v in m.z_stat]))
print('[M2c] 空间误差模型(+海拔) 标准化β:')
for f in feats:
    print(f'        {f:12} β={b[f]:+.3f}  p={p[f]:.2e}')

# ===== M2d 多重共线性诊断 VIF（标准化预测变量，含常数项；VIF 与量纲无关）=====
Xc = sm.add_constant(Xz)
print('\n[M2d] VIF (多重共线性诊断, 经验阈值 <5 可接受, <10 通常无碍):')
for i, f in enumerate(feats):
    print(f'        {f:12} VIF={variance_inflation_factor(Xc, i + 1):.2f}')

# ===== M2e 空间依赖诊断 LM / Robust LM（论证选用空间误差模型优于空间滞后模型）=====
ols = spreg_OLS(y.reshape(-1, 1), Xz, w=w, spat_diag=True, moran=True, name_x=feats)
print('[M2e] OLS 残差空间诊断 (统计量 / p):')
print(f"        Moran's I (error) = {ols.moran_res[0]:+.3f}  (p={ols.moran_res[2]:.2e})")
print(f'        LM-Lag            = {ols.lm_lag[0]:8.2f}  (p={ols.lm_lag[1]:.2e})')
print(f'        LM-Error          = {ols.lm_error[0]:8.2f}  (p={ols.lm_error[1]:.2e})')
print(f'        Robust LM-Lag     = {ols.rlm_lag[0]:8.2f}  (p={ols.rlm_lag[1]:.2e})')
print(f'        Robust LM-Error   = {ols.rlm_error[0]:8.2f}  (p={ols.rlm_error[1]:.2e})')

# ===== M4 正式中介 IRSAD ->(NDVI,NDBI)-> LST =====
z = lambda s: (s - s.mean()) / s.std()
df4 = pd.DataFrame({'X': z(g.IRSAD_score).values, 'M1': z(g.NDVI).values,
                    'M2': z(g.NDBI).values, 'Y': z(g.LST).values})
c = sm.OLS(df4.Y, sm.add_constant(df4.X)).fit().params['X']
full = sm.OLS(df4.Y, sm.add_constant(df4[['X', 'M1', 'M2']])).fit()
cp, b1, b2 = full.params['X'], full.params['M1'], full.params['M2']
a1 = sm.OLS(df4.M1, sm.add_constant(df4.X)).fit().params['X']
a2 = sm.OLS(df4.M2, sm.add_constant(df4.X)).fit().params['X']
ind = c - cp
rng = np.random.default_rng(42); boots = []
for _ in range(2000):
    s = df4.sample(len(df4), replace=True, random_state=rng.integers(1_000_000_000))
    cc = sm.OLS(s.Y, sm.add_constant(s.X)).fit().params['X']
    cpp = sm.OLS(s.Y, sm.add_constant(s[['X', 'M1', 'M2']])).fit().params['X']
    boots.append(cc - cpp)
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f'\n[M4] 中介: total c={c:+.3f}, direct c\'={cp:+.3f}, indirect={ind:+.3f}')
print(f'      经NDVI {a1*b1:+.3f}, 经NDBI {a2*b2:+.3f}; indirect 95%CI [{lo:+.3f},{hi:+.3f}]; 中介比例 {ind/c*100:.0f}%')

# ===== M5 RF 空间块交叉验证 =====
fr = ['IRSAD_score', 'NDVI', 'NDBI', 'elevation', 'pop_density', 'elderly_pct', 'median_hhd_income']
gg = g.dropna(subset=fr).copy()
Xr, yr = gg[fr].values, gg.LST.values
blocks = KMeans(n_clusters=10, random_state=42, n_init=10).fit_predict(gg[['x', 'y']].values)
oob = RandomForestRegressor(500, random_state=42, oob_score=True).fit(Xr, yr).oob_score_
cv = cross_val_score(RandomForestRegressor(500, random_state=42), Xr, yr,
                     cv=GroupKFold(10), groups=blocks, scoring='r2')
print(f'\n[M5] RF: OOB R²={oob:.3f}; 空间块CV R²={cv.mean():.3f}±{cv.std():.3f} (10 blocks)')
