"""
项目 2 — Step 0：准备 GEE 边界（Greater Melbourne GCCSA + SA2）
=============================================================
把 ABS ASGS 2021 的 SA2 shapefile 过滤成"大墨尔本"，输出 GEE 可上传的 GeoJSON。

ABS 下载（免费）：ASGS Edition 3 (2021) — Statistical Area Level 2 (SA2) 数字边界文件
  https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files
下载 "Statistical Area Level 2 (SA2) ... Shapefile"，解压到本目录任意位置即可
（脚本会自动递归查找含 SA2 的 .shp）。

依赖：pip install geopandas
"""

import glob
import sys
import geopandas as gpd

GCC_NAME     = "Greater Melbourne"
GCC_FIELD    = "GCC_NAME21"
KEEP         = ["SA2_CODE21", "SA2_NAME21", "GCC_NAME21"]
SIMPLIFY_TOL = 0.0005          # 度(~50m)，减小 asset 体积；设 None 关闭


def find_sa2_shp():
    cands = [p for p in glob.glob("**/*.shp", recursive=True) if "SA2" in p.upper()]
    if not cands:
        sys.exit("未找到 SA2 的 .shp —— 请先从 ABS 下载并解压 SA2 2021 Shapefile 到本目录。")
    return cands[0]


def main():
    shp = find_sa2_shp()
    print("读取:", shp)
    gdf = gpd.read_file(shp)

    missing = [c for c in KEEP if c not in gdf.columns]
    if missing:
        print("当前文件字段:", list(gdf.columns))
        sys.exit(f"缺字段 {missing} —— 按你的 ABS 版本调整 KEEP / GCC_FIELD。")

    melb = gdf[gdf[GCC_FIELD] == GCC_NAME].copy()
    if melb.empty:
        print(f"{GCC_FIELD} 取值示例:", gdf[GCC_FIELD].dropna().unique()[:10])
        sys.exit(f"未匹配到 '{GCC_NAME}' —— 检查取值后重试。")

    melb = melb.to_crs(4326)                      # GEE 要 WGS84
    melb = melb[KEEP + ["geometry"]]
    if SIMPLIFY_TOL:
        melb["geometry"] = melb.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    melb.to_file("SA2_GreaterMelbourne.geojson", driver="GeoJSON")
    print(f"✅ SA2_GreaterMelbourne.geojson（{len(melb)} 个 SA2）")

    gccsa = melb.dissolve().reset_index(drop=True)[["geometry"]]
    gccsa.to_file("GreaterMelbourne_GCCSA.geojson", driver="GeoJSON")
    print("✅ GreaterMelbourne_GCCSA.geojson（整体外边界，可替换 01 的 MELB_BBOX）")

    print("\n下一步：https://code.earthengine.google.com → Assets 标签 →")
    print("  NEW → Shape files / GeoJSON → 上传 SA2_GreaterMelbourne.geojson")
    print("  把生成的 asset 路径填进 01_gee_extract.py 的 SA2_ASSET。")
    print("（若 UI 不接受 GeoJSON，用 geopandas 另存为 .shp 后把 4 个文件打包 zip 上传。）")


if __name__ == "__main__":
    main()
