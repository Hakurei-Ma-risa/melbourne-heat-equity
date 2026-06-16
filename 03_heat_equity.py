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
imp = (pd.DataFrame({'feature': feat, 'importance': perm.importances_mean})
       .sort_values('importance', ascending=False))
print(imp.to_string(index=False))

# ===== 图1：散点 + 箱线 + 重要性 =====
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].scatter(sub.IRSAD_score, sub.LST, s=10, alpha=.5)
b = np.polyfit(sub.IRSAD_score, sub.LST, 1)
xs = np.array([sub.IRSAD_score.min(), sub.IRSAD_score.max()])
ax[0].plot(xs, np.polyval(b, xs), 'r-')
ax[0].set_xlabel('IRSAD (advantage →)'); ax[0].set_ylabel('Recent summer LST (°C)')
ax[0].set_title(f'Surface temperature vs SES   r={r:+.2f}')
sub.boxplot(column='LST', by='IRSAD_q', ax=ax[1], grid=False)
ax[1].set_title('Surface temperature by IRSAD quintile'); ax[1].set_xlabel(''); ax[1].set_ylabel('LST (°C)')
ax[1].tick_params(axis='x', rotation=20)
ax[2].barh(imp.feature, imp.importance); ax[2].invert_yaxis()
ax[2].set_title('Drivers of surface temperature (RF importance)')
plt.suptitle(''); plt.tight_layout(); plt.savefig('fig_heat_equity.png', dpi=200)

# ===== 图2：热暴露 vs IRSAD 双地图（米制CRS + 指北针 + 比例尺）=====
g = gpd.read_file('SA2_GreaterMelbourne.geojson'); g['SA2_CODE21'] = g['SA2_CODE21'].astype(str)
g = g.merge(d, on='SA2_CODE21', how='left').to_crs(7855)   # GDA2020 / MGA zone 55 (metres)

def add_decor(ax):
    ax.annotate('N', xy=(0.06, 0.97), xytext=(0.06, 0.86), xycoords='axes fraction',
                ha='center', va='center', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.5))
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    bx, by, L = x0 + 0.06 * (x1 - x0), y0 + 0.06 * (y1 - y0), 20000
    ax.plot([bx, bx + L], [by, by], color='k', lw=3)
    ax.text(bx + L / 2, by + 0.015 * (y1 - y0), '20 km', ha='center', va='bottom', fontsize=8)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
g.plot(column='LST', cmap='inferno', legend=True, ax=ax[0],
       legend_kwds={'label': 'Recent summer LST (°C)', 'shrink': .6})
ax[0].set_title('(a) Recent summer LST'); add_decor(ax[0]); ax[0].set_axis_off()
g.plot(column='IRSAD_score', cmap='viridis', legend=True, ax=ax[1],
       legend_kwds={'label': 'IRSAD score', 'shrink': .6})
ax[1].set_title('(b) Socio-economic advantage (IRSAD)'); add_decor(ax[1]); ax[1].set_axis_off()
plt.tight_layout(); plt.savefig('fig_maps_heat_seifa.png', dpi=200)

print('\n图已保存: fig_heat_equity.png, fig_maps_heat_seifa.png')
print('每SA2结果: heat_equity_results.csv')
