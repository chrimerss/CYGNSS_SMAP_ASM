import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from osgeo import gdal
from functools import partial
import pygrib
import h5py
import affine
from multiprocessing import Pool
import time

def preprocess(data, threshold1=0, threshold2=0.01):
        # this function drops nan and values between thresholds
        data= data.astype('float32')
        cols= data.columns
        for col in cols:
            # data= data.apply(np.log)
            data=data[data[col]>=threshold2]
            data.clip(lower=threshold2, inplace=True)
            # data.dropna(inplace=True)
        # print(data)
        data= data.apply(np.log)
        return data
    
def mtc(X):
        #check X has shape (time sequence, 3)
        N_boot= 100
        rmse= np.zeros((N_boot,3))
        cc= np.zeros((N_boot, 3))
        for i in range(N_boot):
            sigma= np.zeros(3)
            r= np.zeros(3)
            sample= bootstrap_resample(X, n=N_boot)
            # print(X.columns)
            cov= sample.cov().to_numpy()
            # print(cov)
            # compute RMSE
            if (cov==0).any().any():
                rmse[i,:]=np.nan
                cc[i,:]= np.nan
            else:
                sigma[0]= cov[0,0] - (cov[0,1]*cov[0,2])/(cov[1,2])
                sigma[1]= cov[1,1] - (cov[0,1]*cov[1,2])/(cov[0,2])
                sigma[2]= cov[2,2] - (cov[0,2]*cov[1,2])/(cov[0,1])
                # print(cov[0,0], cov[1,1], cov[2,2])

                sigma[sigma<0]= np.nan
                sigma= sigma**.5

                #compute correlation coefficient
                r[0] = (cov[0,1]*cov[0,2])/(cov[0,0]*cov[1,2])
                r[1] = (cov[0,1]*cov[1,2])/(cov[1,1]*cov[0,2]);
                r[2] = (cov[0,2]*cov[1,2])/(cov[2,2]*cov[0,1]);

                #sign function here?
                r[r<0] = 0.0001
                r[r>1] = 1
                r= r**.5
                r[r<1e-3] = 0

                rmse[i,:]= sigma
                cc[i, :]= r

        return np.nanmean(rmse, axis=0), np.nanmean(cc,axis=0)
    
def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : pandas data frame
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)

    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X.iloc[resample_i, :]
    return X_resample


def getIndex(lons,lats, lon, lat):
    col= np.where(abs(lons[0,:] - lon)== np.nanmin(abs(lons[0,:] - lon)))[0]
    row= np.where(abs(lats[:,0] - lat) == np.nanmin(abs(lats[:, 0] - lat)))[0]
                  
    return row, col

def thread(lon, lat, time):
    try:
        fname= '/media/scratch/ZhiLi/SMAP/SMAP_L3_SM_P_E_%s_R16510_001.h5'%time.strftime('%Y%m%d')
        with h5py.File(fname, 'r') as h5:
            soil= h5['Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm'][:]
            lons= h5['Soil_Moisture_Retrieval_Data_PM/longitude_pm'][:]
            lats= h5['Soil_Moisture_Retrieval_Data_PM/latitude_pm'][:]

        row, col= getIndex(lons, lats, lon, lat)
        val= soil[row, col][0]
    except OSError:
        val= np.nan

    return time, val

def SMAPretrieval(lon, lat, periods):

        
    df= pd.DataFrame(index= periods, columns=['smap'])
    func= partial(thread, lon, lat)
    with Pool(24) as pool:
        results= pool.map(func, periods)
    for result in results:
        df.loc[result[0], 'smap']= result[1]
        
    return df
            
def NOAHretrieval(lon, lat, periods):
    df= pd.DataFrame(index=periods, columns=['noah'])
    for time in periods:
        fname= 'data/NLDAS_NOAH0125_H.A%s.%s.002.grb.SUB.grb'%(time.strftime('%Y%m%d'),
                                                              time.strftime('%H%M'))
        try:
            gribs= pygrib.open(fname)
            grib= gribs.message(4)
            lats, lons= grib.latlons()
            arr= grib.values
            soilDF= pd.DataFrame()
            gribs.close()
            datetime= fname.split('.')[1][1:]+fname.split('.')[2]

            row, col= getIndex(lons, lats, lon, lat)

            df.loc[time, 'noah']= arr[row, col][0].astype(np.float32)
        except OSError:
            pass

    return df

def MESOretrieval(lon, lat, periods):

    df= pd.DataFrame(index= periods, columns= ['meso'])
    for time in periods:
        try:
#             ==============OK data====================
            fname= '/media/scratch/ZhiLi/OK_mesonet/%s.tif'%time.strftime('%Y%m%d%H%M%S')
            fname= 'regridedMESO/%s.npy'%time.strftime('%Y%m%d%H')
            arr= np.load(fname)
            lons= arr[2]
            lats= arr[1]
            row, col= getIndex(lons, lats, lon, lat)
            df.loc[time, 'meso']= arr[0,row, col]
#             =============OSU gridded data=============
#             fname= '/media/scratch/ZhiLi/Meso_gridded/%s.tif'%time.strftime('%Y%m%d%H')
#             raster= gdal.Open(fname)
#             geo= raster.GetGeoTransform()
#             xsize= raster.RasterXSize
#             ysize= raster.RasterYSize
#             lons= np.arange(geo[0], geo[0]+xsize*geo[1], geo[1])
#             lats= np.arange(geo[3], geo[3]+ysize*geo[-1], geo[-1])
#             lons, lats= np.meshgrid(lons, lats)
#             row, col= getIndex(lons, lats, lon, lat)
#             df.loc[time, 'meso']= raster.ReadAsArray()[row, col][0].astype(np.float32)
        except AttributeError:
            pass
    
    return df   
    
def pixelTSretrieval(lon,lat):
    periods= pd.date_range('20150901180000', '20190702180000', freq='D')
    df= pd.DataFrame(index= periods, columns=['smap','noah','meso'])
#     ===============Seasonal TC================
    seasons= [periods[periods.month//4==m] for m in range(4)]
    for i, periods in enumerate(seasons):
        
        smap= SMAPretrieval(lon, lat, periods).astype(float)
        noah= NOAHretrieval(lon, lat, periods).astype(float)/100.
        meso= MESOretrieval(lon, lat, periods).astype(float)
        df['smap']= smap
        df['noah']= noah
        df['meso']= meso
        df[df<0]= np.nan
#     return df
        RMSE, CC= mtc(df.dropna())
#         print('RMSE shape: ', RMSE)
    
        yield i, RMSE, CC
    
    
def main():
#     ==========OSU data============
#     sample= gdal.Open('/media/scratch/ZhiLi/Meso_gridded/2015090106.tif')
#     nlon, nlat= sample.RasterXSize, sample.RasterYSize
#     geo= sample.GetGeoTransform()
#     lons= geo[0]+ np.arange(nlon)*geo[1]
#     lats= geo[3]+ np.arange(nlat)*geo[-1]
#     ==========OK data=============
    sample= np.load('regridedMESO/2015040118.npy')
    lons= sample[2][0,:]
    lats= sample[1][:,0]
    nlon= len(lons)
    nlat= len(lats)
    RMSE_field= np.zeros((3,4,nlat, nlon)) * -9999.
    CC_field= np.zeros((3,4,nlat, nlon))* -9999.
    for n, lon in enumerate(lons):
        for m, lat in enumerate(lats):
            print('%d/%d'%((m+1)*(n+1), len(lons)*len(lats)))
#             if sample.ReadAsArray()[m,n]<0: pass
            if sample[0,m,n]<0: pass
            else:
                 try:
                     for s, RMSE, CC in pixelTSretrieval(lon, lat):
#                 RMSE, CC= pixelTSretrieval(lon, lat) 
                         RMSE_field[:, s, m, n]= RMSE
                         CC_field[:, s, m, n]= CC
                 except:
                     pass
    
#     arr2raster('RMSE_MESO_2020_05_21.tif', RMSE_field[2], lons, lats)
#     arr2raster('RMSE_NOAH_2020_05_21.tif', RMSE_field[1], lons, lats)
#     arr2raster('RMSE_SMAP_2020_05_21.tif', RMSE_field[0], lons, lats)
#     arr2raster('CC_SMAP_2020_05_21.tif', CC_field[0], lons, lats)
#     arr2raster('CC_NOAH_2020_05_21.tif', CC_field[1], lons, lats)
#     arr2raster('CC_MESO_2020_05_21.tif', CC_field[2], lons, lats)
            
    return RMSE_field, CC_field

def arr2raster(dst, arr, lons, lats):
    cols= arr.shape[1]
    rows= arr.shape[0]
    originX= lons[0]
    originY= lats[0]
    lon_diff= lons[1] - lons[0]
    lat_diff= lats[1]- lats[0]
    driver= gdal.GetDriverByName('GTiff')
    outdata= driver.Create(dst, cols, rows, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform((originX, lon_diff, 0, originY, 0, lat_diff))
    outdata.SetProjection('EPSG:4326')
    outdata.GetRasterBand(1).WriteArray(arr)
    outdata.GetRasterBand(1).SetNoDataValue(-9999.)

if __name__=='__main__':
    
    RMSE, CC= main()
# 	np.save('RMSE_raster_20.npy', RMSE)
# 	np.save('CC_raster_PM.npy',CC)
    np.save('RMSE_raster_PM_seasons.npy', RMSE)
    np.save('CC_raster_PM_seasons.npy', CC)
