# Test code to extract soil properties from SoilGrids in Switzerland using Google Earth Engine
# In the end didn't use it as no relationship was significant

#%%
import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd

ee.Initialize(project='ee-sebbiass')

import seaborn as sns

#%%

# Load soilgrid
soc = ee.Image("projects/soilgrids-isric/soc_mean")
ph = ee.Image("projects/soilgrids-isric/phh2o_mean")
soiltype = ee.Image("OpenLandMap/SOL/SOL_GRTGROUP_USDA-SOILTAX_C/v01")
lulc = ee.ImageCollection('ESA/WorldCover/v100').first()
srtm = ee.Image("USGS/SRTMGL1_003")

gdf = gpd.read_file('/Users/seb/Documents/WORK/Data/GIS/Vector/NaturalEarth/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
gdf = gdf[gdf['ADMIN'] == 'Switzerland']

bounds = gdf.iloc[0].geometry.bounds

minx, miny, maxx, maxy = bounds
num_points = 5000

xs = np.random.uniform(minx, maxx, num_points)
ys = np.random.uniform(miny, maxy, num_points)

points = gpd.GeoSeries(gpd.points_from_xy(xs, ys))
points = gpd.GeoDataFrame(geometry=points, crs=gdf.crs)

intersected_points = gpd.sjoin(points, gdf, predicate='intersects')

# Prepare list of coordinates
coords = intersected_points.geometry.apply(lambda geom: [geom.x, geom.y]).tolist()

# Create EE FeatureCollection from points
ee_points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(coord)) for coord in coords])

#%%
def getData(image, scale):

    # Extract SOC values at each point
    data = image.sampleRegions(collection=ee_points, scale=scale, geometries=True)

    # Get results as a list of dictionaries
    dict_data = data.getInfo()

    # Optionally, convert to pandas DataFrame
    data_df = pd.DataFrame([f['properties'] for f in dict_data['features']])
    
    return data_df

soc_df = getData(soc, 250)
ph_df = getData(ph, 250)
soiltype_df = getData(soiltype, 250)
lulc_df = getData(lulc, 100)
srtm_df = getData(srtm, 30)
#%%


lulc_dict = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}

soiltype_dict = {
    0: "NODATA",
    1: "Albaqualfs",
    2: "Cryaqualfs",
    4: "Durixeralfs",
    6: "Endoaqualfs",
    7: "Epiaqualfs",
    9: "Fragiaqualfs",
    10: "Fragiudalfs",
    11: "Fragixeralfs",
    12: "Fraglossudalfs",
    13: "Glossaqualfs",
    14: "Glossocryalfs",
    15: "Glossudalfs",
    16: "Haplocryalfs",
    17: "Haploxeralfs",
    18: "Hapludalfs",
    19: "Haplustalfs",
    25: "Natraqualfs",
    26: "Natrixeralfs",
    27: "Natrudalfs",
    28: "Natrustalfs",
    29: "Palecryalfs",
    30: "Paleudalfs",
    31: "Paleustalfs",
    32: "Palexeralfs",
    38: "Rhodustalfs",
    39: "Vermaqualfs",
    41: "Eutroboralfs",
    42: "Ochraqualfs",
    43: "Glossoboralfs",
    44: "Cryoboralfs",
    45: "Natriboralfs",
    46: "Paleboralfs",
    50: "Cryaquands",
    58: "Fulvicryands",
    59: "Fulvudands",
    61: "Haplocryands",
    63: "Haploxerands",
    64: "Hapludands",
    74: "Udivitrands",
    75: "Ustivitrands",
    76: "Vitraquands",
    77: "Vitricryands",
    80: "Vitrixerands",
    82: "Aquicambids",
    83: "Aquisalids",
    85: "Argidurids",
    86: "Argigypsids",
    87: "Calciargids",
    89: "Calcigypsids",
    90: "Gypsiargids",
    92: "Haplargids",
    93: "Haplocalcids",
    94: "Haplocambids",
    96: "Haplodurids",
    97: "Haplogypsids",
    98: "Haplosalids",
    99: "Natrargids",
    100: "Natridurids",
    101: "Natrigypsids",
    102: "Paleargids",
    103: "Petroargids",
    104: "Petrocalcids",
    105: "Petrocambids",
    107: "Petrogypsids",
    110: "Calciorthids",
    111: "Camborthids",
    112: "Paleorthids",
    113: "Durorthids",
    114: "Durargids",
    115: "Gypsiorthids",
    116: "Nadurargids",
    118: "Cryaquents",
    119: "Cryofluvents",
    120: "Cryopsamments",
    121: "Cryorthents",
    122: "Endoaquents",
    123: "Epiaquents",
    124: "Fluvaquents",
    126: "Frasiwassents",
    131: "Hydraquents",
    133: "Psammaquents",
    134: "Psammowassents",
    135: "Quartzipsamments",
    136: "Sulfaquents",
    137: "Sulfiwassents",
    138: "Torrifluvents",
    139: "Torriorthents",
    140: "Torripsamments",
    141: "Udifluvents",
    142: "Udipsamments",
    143: "Udorthents",
    144: "Ustifluvents",
    145: "Ustipsamments",
    146: "Ustorthents",
    147: "Xerofluvents",
    148: "Xeropsamments",
    149: "Xerorthents",
    153: "Udarents",
    154: "Torriarents",
    155: "Xerarents",
    179: "Cryofibrists",
    180: "Cryofolists",
    181: "Cryohemists",
    182: "Cryosaprists",
    183: "Frasiwassists",
    184: "Haplofibrists",
    185: "Haplohemists",
    186: "Haplosaprists",
    189: "Sphagnofibrists",
    190: "Sulfihemists",
    191: "Sulfisaprists",
    196: "Udifolists",
    201: "Borosaprists",
    202: "Medisaprists",
    203: "Borohemists",
    206: "Calcicryepts",
    207: "Calciustepts",
    208: "Calcixerepts",
    209: "Cryaquepts",
    210: "Durixerepts",
    212: "Durustepts",
    213: "Dystrocryepts",
    215: "Dystroxerepts",
    216: "Dystrudepts",
    217: "Dystrustepts",
    218: "Endoaquepts",
    219: "Epiaquepts",
    220: "Eutrudepts",
    221: "Fragiaquepts",
    222: "Fragiudepts",
    225: "Halaquepts",
    226: "Haplocryepts",
    228: "Haploxerepts",
    229: "Haplustepts",
    230: "Humaquepts",
    231: "Humicryepts",
    233: "Humixerepts",
    234: "Humudepts",
    235: "Humustepts",
    245: "Ustochrepts",
    246: "Eutrochrepts",
    247: "Dystrochrepts",
    248: "Eutrocryepts",
    249: "Haplaquepts",
    250: "Xerochrepts",
    251: "Cryochrepts",
    252: "Fragiochrepts",
    253: "Haplumbrepts",
    254: "Cryumbrepts",
    255: "Dystropepts",
    256: "Vitrandepts",
    268: "Argialbolls",
    269: "Argiaquolls",
    270: "Argicryolls",
    271: "Argiudolls",
    272: "Argiustolls",
    273: "Argixerolls",
    274: "Calciaquolls",
    275: "Calcicryolls",
    276: "Calciudolls",
    277: "Calciustolls",
    278: "Calcixerolls",
    279: "Cryaquolls",
    280: "Cryrendolls",
    283: "Durixerolls",
    284: "Durustolls",
    285: "Endoaquolls",
    286: "Epiaquolls",
    287: "Haplocryolls",
    289: "Haploxerolls",
    290: "Hapludolls",
    291: "Haplustolls",
    292: "Haprendolls",
    294: "Natraquolls",
    296: "Natrixerolls",
    297: "Natrudolls",
    298: "Natrustolls",
    299: "Palecryolls",
    300: "Paleudolls",
    301: "Paleustolls",
    302: "Palexerolls",
    303: "Vermudolls",
    306: "Haploborolls",
    307: "Argiborolls",
    308: "Haplaquolls",
    309: "Cryoborolls",
    310: "Natriborolls",
    311: "Calciborolls",
    312: "Paleborolls",
    342: "Alaquods",
    343: "Alorthods",
    345: "Duraquods",
    348: "Durorthods",
    349: "Endoaquods",
    350: "Epiaquods",
    351: "Fragiaquods",
    353: "Fragiorthods",
    354: "Haplocryods",
    356: "Haplohumods",
    357: "Haplorthods",
    358: "Humicryods",
    367: "Haplaquods",
    368: "Cryorthods",
    370: "Albaquults",
    371: "Endoaquults",
    372: "Epiaquults",
    373: "Fragiaquults",
    374: "Fragiudults",
    375: "Haplohumults",
    376: "Haploxerults",
    377: "Hapludults",
    378: "Haplustults",
    381: "Kandiudults",
    385: "Kanhapludults",
    387: "Paleaquults",
    388: "Palehumults",
    389: "Paleudults",
    390: "Paleustults",
    391: "Palexerults",
    396: "Rhodudults",
    399: "Umbraquults",
    401: "Ochraquults",
    403: "Calciaquerts",
    405: "Calciusterts",
    406: "Calcixererts",
    409: "Dystraquerts",
    410: "Dystruderts",
    412: "Endoaquerts",
    413: "Epiaquerts",
    414: "Gypsitorrerts",
    415: "Gypsiusterts",
    417: "Haplotorrerts",
    418: "Haploxererts",
    419: "Hapluderts",
    420: "Haplusterts",
    422: "Natraquerts",
    424: "Salitorrerts",
    429: "Chromusterts",
    430: "Pellusterts",
    431: "Chromoxererts",
    432: "Pelluderts",
    433: "Torrerts"
}

#%%

soiltype_df['soiltype'] = soiltype_df['grtgroup'].map(soiltype_dict)
soiltype_df = soiltype_df.drop(columns=['grtgroup'])

lulc_df['landcover'] = lulc_df['Map'].map(lulc_dict)
lulc_df = lulc_df.drop(columns=['Map'])

# Merge all DataFrames on their coordinates
merged_df = soc_df.join(ph_df).join(soiltype_df).join(lulc_df).join(srtm_df)


merged_df.to_csv('/Users/seb/Documents/WORK/Teaching/UNIGE/Master/Data Science/Data/soil_switzerland.csv', index=False)

#%%

merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('100-200cm_mean')]
merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('15-30cm_mean')]
merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('30-60cm_mean')]
merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('60-100cm_mean')]

sns.pairplot(merged_df.drop(columns='landcover'), hue="soiltype")
# sns.pairplot(merged_df.drop(columns='soiltype'), hue="landcover")