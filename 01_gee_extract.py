"""
项目 2 — 大墨尔本 EO 分析 ｜ Step 1：Google Earth Engine 提取
================================================================
逐年（2000–2024）austral-summer（12–2 月）合成 NDVI / NDBI / NDWI / LST，
并按 SA2 做分区统计，导出 CSV 到 Google Drive。

目标期刊：International Journal of Digital Earth (T&F, SCIE Q1)

前置：
  pip install earthengine-api geemap
  earthengine authenticate            # 一次性，浏览器登录
先在 https://code.earthengine.google.com 注册并建一个 Cloud 项目，填到 GEE_PROJECT。
然后按下方 CONFIG 自定义即可运行。
"""

# --- 强制 IPv4：本机 IPv6 不通会让 google 客户端卡在建连，这里让所有连接只走 IPv4 ---
import socket as _socket
_orig_gai = _socket.getaddrinfo
_socket.getaddrinfo = lambda *a, **k: [r for r in _orig_gai(*a, **k) if r[0] == _socket.AF_INET]
# --------------------------------------------------------------------------------

import ee

# ----------------------------- CONFIG -----------------------------
GEE_PROJECT   = "journal-34649077"          # <-- 改成你的 GEE Cloud 项目 id
START_YEAR    = 2000
END_YEAR      = 2024
# 上传 ABS ASGS 2021 SA2 边界为 GEE asset，填路径；核对字段名（见 SELECTORS）
SA2_ASSET     = "projects/journal-34649077/assets/SA2_GreaterMelbourne"  # <-- 改
EXPORT_FOLDER = "project2_melbourne_eo"
ZONAL_SCALE   = 30          # m；想更快可设 100
SELECTORS     = ['SA2_CODE21', 'SA2_NAME21', 'year', 'NDVI', 'NDBI', 'NDWI', 'LST']
# ------------------------------------------------------------------

ee.Initialize(project=GEE_PROJECT)

# bbox 用于过滤范围（须在 Initialize 之后才能调用 ee.Geometry）；正式版可换 GreaterMelbourne_GCCSA
MELB_BBOX = ee.Geometry.Rectangle([144.3, -38.5, 145.6, -37.5])

# Landsat Collection-2 Level-2 波段 -> 统一命名（关键：L5/7 与 L8/9 编号不同）
BANDS_L57 = {'SR_B1': 'blue', 'SR_B2': 'green', 'SR_B3': 'red', 'SR_B4': 'nir',
             'SR_B5': 'swir1', 'SR_B7': 'swir2', 'ST_B6': 'thermal'}
BANDS_L89 = {'SR_B2': 'blue', 'SR_B3': 'green', 'SR_B4': 'red', 'SR_B5': 'nir',
             'SR_B6': 'swir1', 'SR_B7': 'swir2', 'ST_B10': 'thermal'}


def mask_and_scale(img):
    """QA_PIXEL 云/影掩膜 + Collection-2 L2 比例因子（光学反射率；热红外 -> 摄氏度）。"""
    qa = img.select('QA_PIXEL')
    mask = (qa.bitwiseAnd(1 << 1).eq(0)        # dilated cloud
              .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
              .And(qa.bitwiseAnd(1 << 4).eq(0)))  # cloud shadow
    optical = img.select('SR_B.*').multiply(0.0000275).add(-0.2)
    thermal = img.select('ST_B.*').multiply(0.00341802).add(149.0).subtract(273.15)
    return img.addBands(optical, None, True).addBands(thermal, None, True).updateMask(mask)


def prep(collection_id, band_map):
    """掩膜+定标，再选取并重命名为统一波段。"""
    src, dst = list(band_map.keys()), list(band_map.values())

    def _f(img):
        img = mask_and_scale(img)
        return (img.select(src).rename(dst)
                   .copyProperties(img, ['system:time_start']))

    return ee.ImageCollection(collection_id).map(_f)


def add_indices(img):
    ndvi = img.normalizedDifference(['nir', 'red']).rename('NDVI')   # 植被
    ndbi = img.normalizedDifference(['swir1', 'nir']).rename('NDBI')  # 建成区
    ndwi = img.normalizedDifference(['green', 'nir']).rename('NDWI')  # 水体
    lst  = img.select('thermal').rename('LST')                        # 地表温度(℃)
    return img.addBands([ndvi, ndbi, ndwi, lst])


def summer_composite(year, coll):
    """austral 夏季：Dec(year-1) .. Feb(year) 的中值合成。"""
    start = ee.Date.fromYMD(year - 1, 12, 1)
    end   = ee.Date.fromYMD(year, 3, 1)
    return (coll.filterDate(start, end).median()
                .select(['NDVI', 'NDBI', 'NDWI', 'LST'])
                .set('year', year))


def zonal_for_year(year, coll, sa2):
    comp = summer_composite(year, coll)
    fc = comp.reduceRegions(collection=sa2, reducer=ee.Reducer.mean(), scale=ZONAL_SCALE)
    return fc.map(lambda f: f.set('year', year))


def main():
    melb = MELB_BBOX
    sa2 = ee.FeatureCollection(SA2_ASSET).filterBounds(melb)

    landsat = (prep('LANDSAT/LT05/C02/T1_L2', BANDS_L57)
               .merge(prep('LANDSAT/LE07/C02/T1_L2', BANDS_L57))
               .merge(prep('LANDSAT/LC08/C02/T1_L2', BANDS_L89))
               .merge(prep('LANDSAT/LC09/C02/T1_L2', BANDS_L89))
               .filterBounds(melb)
               .filterDate(f'{START_YEAR - 1}-12-01', f'{END_YEAR}-03-01')
               .map(add_indices))

    years = list(range(START_YEAR, END_YEAR + 1))
    all_fc = ee.FeatureCollection(
        [zonal_for_year(y, landsat, sa2) for y in years]
    ).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=all_fc,
        description='melb_sa2_annual_indices',
        folder=EXPORT_FOLDER,
        fileNamePrefix='melb_sa2_annual_indices',
        fileFormat='CSV',
        selectors=SELECTORS,
    )
    task.start()
    print('已提交导出任务：melb_sa2_annual_indices')
    print('在 https://code.earthengine.google.com 的 Tasks 标签查看进度，')
    print(f'完成后 CSV 会出现在 Google Drive 的 "{EXPORT_FOLDER}" 文件夹。')

    # 可选：同时导出某年的 LST/NDVI/NDBI 栅格用于制图（取消注释）
    # for y in [START_YEAR, END_YEAR]:
    #     img = summer_composite(y, landsat).select(['LST', 'NDVI', 'NDBI']).clip(melb)
    #     ee.batch.Export.image.toDrive(
    #         image=img, description=f'melb_indices_{y}', folder=EXPORT_FOLDER,
    #         region=melb, scale=30, maxPixels=1e13).start()


if __name__ == '__main__':
    main()
