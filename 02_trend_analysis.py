"""
项目 2 — 大墨尔本 EO 分析 ｜ Step 2：趋势 + 驱动分析
====================================================
输入：Step 1 导出的 melb_sa2_annual_indices.csv（SA2 × year × NDVI/NDBI/NDWI/LST）
产出：
  - 每个 SA2 的 LST/NDVI/NDBI 趋势（Mann-Kendall + Sen's slope）
  - join ABS 社会经济后，随机森林解析热岛(LST)驱动因子（重要性）
  - 基础图表（趋势分布、重要性、可选趋势地图）

依赖：pandas numpy pymannkendall scikit-learn geopandas matplotlib
"""

import numpy as np
import pandas as pd
import pymannkendall as mk
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ----------------------------- CONFIG -----------------------------
PANEL_CSV = "melb_sa2_annual_indices.csv"   # Step 1 导出的文件
ABS_CSV   = "abs_sa2_socioeconomic.csv"     # 你准备：列含 SA2_CODE21 + 驱动因子
SA2_SHP   = "SA2_GreaterMelbourne.shp"      # 可选：制图用 ABS SA2 矢量
ID_COL    = "SA2_CODE21"
# ABS 驱动因子列名（按你的 DataPack 改）
DRIVERS   = ["median_income", "pop_density", "dwelling_density"]
# ------------------------------------------------------------------


def sens_trend(group, value_col):
    """对一个 SA2 的时间序列做 Mann-Kendall + Sen's slope。"""
    s = group.sort_values("year")[value_col].dropna().values
    if len(s) < 8:                       # 点太少不做趋势
        return pd.Series({f"{value_col}_slope": np.nan, f"{value_col}_p": np.nan,
                          f"{value_col}_trend": "insufficient"})
    r = mk.original_test(s)
    return pd.Series({f"{value_col}_slope": r.slope, f"{value_col}_p": r.p,
                      f"{value_col}_trend": r.trend})


def compute_trends(panel):
    out = []
    for col in ["LST", "NDVI", "NDBI"]:
        t = panel.groupby(ID_COL).apply(lambda g: sens_trend(g, col))
        out.append(t)
    return pd.concat(out, axis=1).reset_index()


def driver_analysis(trends, abs_df):
    """用 ABS 驱动因子 + 植被/建成趋势 解释 LST 趋势（热岛增强）。"""
    df = trends.merge(abs_df, on=ID_COL, how="inner")
    feat = DRIVERS + ["NDVI_slope", "NDBI_slope"]
    df = df.dropna(subset=feat + ["LST_slope"])
    X, y = df[feat].values, df["LST_slope"].values

    rf = RandomForestRegressor(n_estimators=500, random_state=42, oob_score=True)
    rf.fit(X, y)
    print(f"RandomForest OOB R^2 = {rf.oob_score_:.3f}")

    perm = permutation_importance(rf, X, y, n_repeats=30, random_state=42)
    imp = (pd.DataFrame({"feature": feat, "importance": perm.importances_mean})
             .sort_values("importance", ascending=False))
    print("\n驱动因子重要性：\n", imp.to_string(index=False))
    return df, imp


def plots(trends, imp):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(trends["LST_slope"].dropna(), bins=30)
    ax[0].set_title("LST 年趋势 (Sen's slope) 分布"); ax[0].set_xlabel("℃/yr")
    ax[1].barh(imp["feature"], imp["importance"]); ax[1].invert_yaxis()
    ax[1].set_title("热岛(LST) 驱动因子重要性")
    plt.tight_layout(); plt.savefig("fig_trends_drivers.png", dpi=200)
    print("已保存 fig_trends_drivers.png")

    # 可选：趋势地图（需 SA2_SHP）
    # import geopandas as gpd
    # gdf = gpd.read_file(SA2_SHP).merge(trends, on=ID_COL)
    # gdf.plot(column="LST_slope", legend=True, cmap="RdBu_r")
    # plt.savefig("fig_lst_trend_map.png", dpi=200)


def main():
    panel = pd.read_csv(PANEL_CSV)
    trends = compute_trends(panel)
    trends.to_csv("sa2_trends.csv", index=False)
    print(f"趋势计算完成，{len(trends)} 个 SA2 -> sa2_trends.csv")

    try:
        abs_df = pd.read_csv(ABS_CSV)
        df, imp = driver_analysis(trends, abs_df)
        plots(trends, imp)
    except FileNotFoundError:
        print(f"\n[跳过驱动分析] 未找到 {ABS_CSV}。"
              f"准备好 SA2 级 ABS 社会经济 CSV（含 {ID_COL} 与 {DRIVERS}）后重跑。")


if __name__ == '__main__':
    main()
