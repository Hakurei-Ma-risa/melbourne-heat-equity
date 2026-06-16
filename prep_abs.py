"""
prep_abs.py — 合并 ABS 社会经济数据 → abs_sa2_socioeconomic.csv
================================================================
输入（已在本目录）：
  - gcp_vic/ ... G01(人口/老年) + G02(中位收入)   ← GCP DataPack 解压
  - ~/Downloads/Statistical Area Level 2, Indexes, SEIFA 2021.xlsx (Table 1: IRSD/IRSAD)
  - sa2_shp/*SA2*.shp  (面积 AREASQKM21)
输出：abs_sa2_socioeconomic.csv（SA2_CODE21 + 热公平分析所需变量）
"""
import glob, re, os
import pandas as pd
import geopandas as gpd

HOME    = os.path.expanduser('~')
GCP_DIR = 'gcp_vic'
SEIFA   = f'{HOME}/Downloads/Statistical Area Level 2, Indexes, SEIFA 2021.xlsx'
SA2_SHP = glob.glob('sa2_shp/*SA2*.shp')[0]
OUT     = 'abs_sa2_socioeconomic.csv'


def find(pat):
    return glob.glob(f'{GCP_DIR}/**/{pat}', recursive=True)[0]


# --- G01: 总人口 + 65 岁以上(热脆弱人群) ---
g01 = pd.read_csv(find('*G01*SA2*.csv'))
eld = [c for c in g01.columns if c.endswith('_P') and re.search(r'(65_74|75_84|85ov|85_over)', c)]
print('老年(65+)列:', eld)
g01_out = pd.DataFrame({
    'SA2_CODE21': g01['SA2_CODE_2021'].astype(str),
    'pop':        g01['Tot_P_P'],
    'elderly':    g01[eld].sum(axis=1),
})

# --- G02: 中位家庭周收入 + 中位年龄 ---
g02 = pd.read_csv(find('*G02*SA2*.csv'))
g02_out = pd.DataFrame({
    'SA2_CODE21':        g02['SA2_CODE_2021'].astype(str),
    'median_hhd_income': g02['Median_tot_hhd_inc_weekly'],
    'median_age':        g02['Median_age_persons'],
})

# --- SEIFA Table 1: 第0列码, 第2列 IRSD分, 第4列 IRSAD分 ---
s = pd.read_excel(SEIFA, sheet_name='Table 1', header=None, skiprows=6).iloc[:, [0, 2, 4]]
s.columns = ['SA2_CODE21', 'IRSD_score', 'IRSAD_score']
s['SA2_CODE21'] = s['SA2_CODE21'].astype(str).str.replace(r'\.0$', '', regex=True)
s = s[s['SA2_CODE21'].str.match(r'^\d{9}$')]
for c in ['IRSD_score', 'IRSAD_score']:
    s[c] = pd.to_numeric(s[c], errors='coerce')

# --- SA2 面积 ---
shp = gpd.read_file(SA2_SHP)
acol = [c for c in shp.columns if c.upper().startswith('AREASQKM')][0]
area = pd.DataFrame({'SA2_CODE21': shp['SA2_CODE21'].astype(str), 'area_km2': shp[acol]})

# --- 合并 + 派生密度/老年比例 ---
df = (g01_out.merge(g02_out, on='SA2_CODE21', how='outer')
             .merge(s,       on='SA2_CODE21', how='left')
             .merge(area,    on='SA2_CODE21', how='left'))
df['pop_density'] = df['pop'] / df['area_km2']
df['elderly_pct'] = df['elderly'] / df['pop'] * 100
df.to_csv(OUT, index=False)
print(f'\n写出 {OUT}: {df.shape[0]} 行, 列={list(df.columns)}')

# --- 覆盖度检查：对照大墨尔本 359 个 SA2 ---
try:
    melb = pd.read_csv('melb_sa2_annual_indices.csv')['SA2_CODE21'].astype(str).unique()
    cov = df[df.SA2_CODE21.isin(melb)]
    print(f'大墨尔本 {len(melb)} 个 SA2 中, 有 IRSAD 的 {cov.IRSAD_score.notna().sum()}, '
          f'有收入的 {cov.median_hhd_income.notna().sum()}, 有面积的 {cov.area_km2.notna().sum()}')
except FileNotFoundError:
    pass
