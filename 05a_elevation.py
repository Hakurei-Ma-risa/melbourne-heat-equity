"""05a — 取每个 SA2 的平均海拔(SRTM),用于控制 M2 地形混淆。"""
import socket as _s
_o = _s.getaddrinfo
_s.getaddrinfo = lambda *a, **k: [r for r in _o(*a, **k) if r[0] == _s.AF_INET]
import ee, pandas as pd
ee.Initialize(project='journal-34649077')

sa2 = ee.FeatureCollection('projects/journal-34649077/assets/SA2_GreaterMelbourne')
dem = ee.Image('USGS/SRTMGL1_003').select('elevation')
fc = dem.reduceRegions(collection=sa2, reducer=ee.Reducer.mean(), scale=90)
feats = fc.getInfo()['features']
rows = [{'SA2_CODE21': str(f['properties']['SA2_CODE21']),
         'elevation': f['properties'].get('mean')} for f in feats]
df = pd.DataFrame(rows)
df.to_csv('sa2_elevation.csv', index=False)
print(f'海拔已存 {len(df)} 个 SA2，范围 {df.elevation.min():.0f}–{df.elevation.max():.0f} m，均值 {df.elevation.mean():.0f}')
