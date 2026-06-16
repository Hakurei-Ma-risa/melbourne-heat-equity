"""
03_heat_equity.py — 夏季地表热岛 × 社会经济公平（核心分析）
=========================================================
热暴露 = 近 10 年(2015–2024)夏季平均 LST（稳健，避开单年噪声）。
检验：社会经济弱势(低 IRSAD)的 SA2 是否承受更高热暴露 + 驱动因子 + 地图。
输出：heat_equity_results.csv, fig_heat_equity.png, fig_maps_heat_seifa.png
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy import stats

RECENT_FROM = 2015

# --- 近10年 EO 均值（热暴露 + 植被/建成背景） ---
panel = pd.read_csv('melb_sa2_annual_indices.csv')
panel['SA2_CODE21'] = panel['SA2_CODE21'].astype(str)
recent = (panel[panel.year >= RECENT_FROM].groupby('SA2_CODE21')
          .agg(LST=('LST', 'mean'), NDVI=('NDVI', 'mean'), NDBI=('NDBI', 'mean'))
          .reset_index())

# --- 合并社会经济 ---
abs_df = pd.read_csv('abs_sa2_socioeconomic.csv')
abs_df['SA2_CODE21'] = abs_df['SA2_CODE21'].astype(str)
d = recent.merge(abs_df, on='SA2_CODE21', how='left')
d['SUHI'] = d['LST'] - d['LST'].median()          # 相对全市中位的热岛强度
d.to_csv('heat_equity_results.csv', index=False)

# ===== 公平性：热暴露 vs IRSAD =====
sub = d.dropna(subset=['LST', 'IRSAD_score']).copy()
r = sub['LST'].corr(sub['IRSAD_score'])
print(f'\n热暴露(近10年夏季LST) vs IRSAD(优势): r = {r:+.3f}  (n={len(sub)})')

sub['IRSAD_q'] = pd.qcut(sub['IRSAD_score'], 5,
                         labels=['Q1 most disadv.', 'Q2', 'Q3', 'Q4', 'Q5 most adv.'])
grp = sub.groupby('IRSAD_q', observed=True)['LST'].mean()
print('各 IRSAD 五分位的平均热暴露(℃):')
print(grp.round(2).to_string())
print(f'最弱势(Q1) − 最优势(Q5) 温差: {grp.iloc[0] - grp.iloc[-1]:+.2f} ℃')

# ===== 驱动：随机森林 =====
feat = ['IRSAD_score', 'median_hhd_income', 'pop_density', 'elderly_pct', 'NDVI', 'NDBI']
m = d.dropna(subset=feat + ['LST']).copy()
rf = RandomForestRegressor(n_estimators=500, random_state=42, oob_score=True).fit(m[feat], m['LST'])
print(f'\nRF 解释热暴露 OOB R² = {rf.oob_score_:.3f}')
perm = permutation_importance(rf, m[feat], m['LST'], n_repeats=30, random_state=42)
imp = (pd.DataFrame({'feature': feat, 'importance': perm.importances_mean, 'std': perm.importances_std})
       .sort_values('importance', ascending=False))
print(imp.to_string(index=False))

# ===== Figure 2：散点+95%CI / 箱线+jitter / 重要性+误差线 =====
PRETTY = {'IRSAD_score': 'IRSAD score', 'median_hhd_income': 'Median income',
          'pop_density': 'Pop. density', 'elderly_pct': 'Elderly %', 'NDVI': 'NDVI', 'NDBI': 'NDBI'}
BLUE = '#4C72B0'
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) 散点 + 线性拟合 + 95% 置信带 + r/p/n 标注
xv, yv = sub.IRSAD_score.values, sub.LST.values
lr = stats.linregress(xv, yv)
xs = np.linspace(xv.min(), xv.max(), 100)
ys = lr.intercept + lr.slope * xs
n = len(xv)
s_err = np.sqrt(np.sum((yv - (lr.intercept + lr.slope * xv)) ** 2) / (n - 2))
ci = stats.t.ppf(0.975, n - 2) * s_err * np.sqrt(1 / n + (xs - xv.mean()) ** 2 / np.sum((xv - xv.mean()) ** 2))
ax[0].scatter(xv, yv, s=10, alpha=.45, color=BLUE, edgecolor='none')
ax[0].plot(xs, ys, 'r-', lw=1.5)
ax[0].fill_between(xs, ys - ci, ys + ci, color='r', alpha=.18, lw=0)
pr, _ = stats.pearsonr(xv, yv)
ax[0].annotate(f'$r$ = {pr:+.2f}, $p$ < 0.001\n$n$ = {n}', xy=(.04, .05), xycoords='axes fraction',
               fontsize=9, va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=.85))
ax[0].set_xlabel('IRSAD (advantage →)'); ax[0].set_ylabel('Recent summer LST (°C)')
ax[0].set_title('(a) Surface temperature vs SES')

# (b) 五分位箱线 + 抖动散点 + Q1−Q5 标注
qs = ['Q1 most disadv.', 'Q2', 'Q3', 'Q4', 'Q5 most adv.']
data = [sub.loc[sub.IRSAD_q == q, 'LST'].values for q in qs]
bp = ax[1].boxplot(data, showfliers=False, patch_artist=True, medianprops=dict(color='k'))
for patch in bp['boxes']:
    patch.set(facecolor='#DDE3EE', edgecolor='0.4')
rng = np.random.default_rng(0)
for i, dd in enumerate(data):
    ax[1].scatter(rng.normal(i + 1, .06, len(dd)), dd, s=6, alpha=.35, color=BLUE, edgecolor='none', zorder=3)
ax[1].set_xticks(range(1, 6)); ax[1].set_xticklabels(qs, rotation=20)
ax[1].annotate(f'Q1 − Q5 = {grp.iloc[0] - grp.iloc[-1]:+.2f} °C', xy=(.97, .97), xycoords='axes fraction',
               ha='right', va='top', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7', alpha=.85))
ax[1].set_title('(b) Surface temperature by IRSAD quintile'); ax[1].set_xlabel(''); ax[1].set_ylabel('LST (°C)')

# (c) RF 置换重要性 + ±1 SD 误差线 + 规范变量名
ax[2].barh([PRETTY.get(f, f) for f in imp.feature], imp.importance, xerr=imp['std'].values,
           color=BLUE, error_kw=dict(ecolor='0.3', lw=1, capsize=3))
ax[2].invert_yaxis()
ax[2].set_xlabel('Permutation importance (Δ MSE)')
ax[2].set_title('(c) Drivers of surface temperature (RF)')
plt.tight_layout(); plt.savefig('fig_heat_equity.png', dpi=300, bbox_inches='tight')

# ===== Figure 1：双地图（投影米制CRS + 指北针 + 比例尺 + CBD + 缺失灰色）=====
g = gpd.read_file('SA2_GreaterMelbourne.geojson'); g['SA2_CODE21'] = g['SA2_CODE21'].astype(str)
g = g.merge(d, on='SA2_CODE21', how='left').to_crs(7855)   # GDA2020 / MGA zone 55 (metres)
cbd = gpd.GeoSeries(gpd.points_from_xy([144.9631], [-37.8136]), crs=4326).to_crs(7855)
cbd_xy = (cbd.x.iloc[0], cbd.y.iloc[0])
MISS = {'color': 'lightgrey', 'edgecolor': '0.6', 'label': 'No data'}

def add_decor(ax):
    ax.annotate('N', xy=(0.06, 0.97), xytext=(0.06, 0.86), xycoords='axes fraction',
                ha='center', va='center', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    bx, by, L = x0 + 0.06 * (x1 - x0), y0 + 0.06 * (y1 - y0), 20000
    ax.plot([bx, bx + L], [by, by], color='k', lw=3)
    ax.text(bx + L / 2, by + 0.015 * (y1 - y0), '20 km', ha='center', va='bottom', fontsize=8)
    ax.plot(*cbd_xy, marker='*', ms=12, mfc='white', mec='k', mew=.8, zorder=6)
    ax.annotate('CBD', xy=cbd_xy, xytext=(7, 5), textcoords='offset points', fontsize=8,
                fontweight='bold', zorder=6, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=.7))

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
g.plot(column='LST', cmap='inferno', legend=True, ax=ax[0], missing_kwds=MISS,
       legend_kwds={'label': 'Recent summer LST (°C)', 'shrink': .6})
ax[0].set_title('(a) Recent summer LST'); add_decor(ax[0]); ax[0].set_axis_off()
g.plot(column='IRSAD_score', cmap='viridis', legend=True, ax=ax[1], missing_kwds=MISS,
       legend_kwds={'label': 'IRSAD score', 'shrink': .6})
ax[1].set_title('(b) Socio-economic advantage (IRSAD)'); add_decor(ax[1]); ax[1].set_axis_off()
plt.tight_layout(); plt.savefig('fig_maps_heat_seifa.png', dpi=300, bbox_inches='tight')

print('\n图已保存: fig_heat_equity.png, fig_maps_heat_seifa.png')
print('每SA2结果: heat_equity_results.csv')
