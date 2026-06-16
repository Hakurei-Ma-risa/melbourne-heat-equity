"""
项目 2 — 冒烟测试
================
不需要 SA2/边界上传、不用导出到 Drive，直接打印大墨尔本几个年份的
夏季平均 NDVI / NDBI / LST，确认整条 GEE 提取链路（波段统一 + 云掩膜 +
指数 + 夏季合成）跑得通。粗分辨率(1km)纯为出结果快。
"""

# 强制 IPv4（本机 IPv6 不通，否则 google 客户端会卡死建连）
import socket as _socket
_o = _socket.getaddrinfo
_socket.getaddrinfo = lambda *a, **k: [r for r in _o(*a, **k) if r[0] == _socket.AF_INET]

import ee
ee.Initialize(project='journal-34649077')

MELB = ee.Geometry.Rectangle([144.3, -38.5, 145.6, -37.5])
B57 = {'SR_B1': 'blue', 'SR_B2': 'green', 'SR_B3': 'red', 'SR_B4': 'nir',
       'SR_B5': 'swir1', 'SR_B7': 'swir2', 'ST_B6': 'thermal'}
B89 = {'SR_B2': 'blue', 'SR_B3': 'green', 'SR_B4': 'red', 'SR_B5': 'nir',
       'SR_B6': 'swir1', 'SR_B7': 'swir2', 'ST_B10': 'thermal'}


def mask_scale(img):
    qa = img.select('QA_PIXEL')
    m = (qa.bitwiseAnd(1 << 1).eq(0)
           .And(qa.bitwiseAnd(1 << 3).eq(0))
           .And(qa.bitwiseAnd(1 << 4).eq(0)))
    opt = img.select('SR_B.*').multiply(0.0000275).add(-0.2)
    th = img.select('ST_B.*').multiply(0.00341802).add(149.0).subtract(273.15)
    return img.addBands(opt, None, True).addBands(th, None, True).updateMask(m)


def prep(cid, bm):
    s, d = list(bm.keys()), list(bm.values())
    return ee.ImageCollection(cid).map(
        lambda i: mask_scale(i).select(s).rename(d).copyProperties(i, ['system:time_start']))


def indices(img):
    return img.addBands([
        img.normalizedDifference(['nir', 'red']).rename('NDVI'),
        img.normalizedDifference(['swir1', 'nir']).rename('NDBI'),
        img.select('thermal').rename('LST')])


land = (prep('LANDSAT/LT05/C02/T1_L2', B57)
        .merge(prep('LANDSAT/LE07/C02/T1_L2', B57))
        .merge(prep('LANDSAT/LC08/C02/T1_L2', B89))
        .merge(prep('LANDSAT/LC09/C02/T1_L2', B89))
        .filterBounds(MELB).map(indices))

print(f"{'year':>6} {'NDVI':>9} {'NDBI':>9} {'LST(C)':>9}")
for y in [2001, 2010, 2020, 2024]:
    comp = land.filterDate(f'{y-1}-12-01', f'{y}-03-01').median().select(['NDVI', 'NDBI', 'LST'])
    s = comp.reduceRegion(ee.Reducer.mean(), MELB, scale=1000, maxPixels=int(1e10)).getInfo()
    fmt = lambda v: f"{v:9.3f}" if v is not None else f"{'NA':>9}"
    print(f"{y:>6} {fmt(s.get('NDVI'))} {fmt(s.get('NDBI'))} {fmt(s.get('LST'))}")

print("\n打印出真实数值 = 整条提取链路已通。下一步做 SA2 边界即可跑完整版。")
