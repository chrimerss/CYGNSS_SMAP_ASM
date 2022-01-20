# SMAP
import datetime as dte
from functools import partial
import multiprocessing
import h5py
import numpy as np
import os

def read_SML3P_AM(filepath):
    ''' This function extracts lat, lon and soil moisture from SMAP L3 P HDF5 file.
    
    Parameters
    ----------
    filepath : str
        File path of a SMAP L3 HDF5 file
    Returns
    -------
    soil_moisture_am: numpy.array
    '''    
    with h5py.File(filepath, 'r') as f:
        # Extract data info
        group_id_am = 'Soil_Moisture_Retrieval_Data_AM'
        var_id_am = 'soil_moisture'
        flag_id_am = 'retrieval_qual_flag'
        soil_moisture_am = f[group_id_am][var_id_am][:,:]
        flag_am = f[group_id_am][flag_id_am][:,:]
        soil_moisture_am[soil_moisture_am==-9999.0]=np.nan;
#         soil_moisture_am[(flag_am>>0)&1==1]=np.nan
        filename = os.path.basename(filepath)
        yyyymmdd= filename.split('_')[5]
        yyyy = int(yyyymmdd[0:4]);        mm = int(yyyymmdd[4:6]);        dd = int(yyyymmdd[6:8])
        date=dte.datetime(yyyy,mm,dd,6)
        lon=f[group_id_am]['longitude'][:]
        lat= f[group_id_am]['latitude'][:]
    return soil_moisture_am,date,lon,lat

def read_SML3P_PM(filepath):
    ''' This function extracts lat, lon and soil moisture from SMAP L3 P HDF5 file.
    
    Parameters
    ----------
    filepath : str
        File path of a SMAP L3 HDF5 file
    Returns
    -------
    soil_moisture_am: numpy.array
    '''    
    with h5py.File(filepath, 'r') as f:
        # Extract data info
        group_id_am = 'Soil_Moisture_Retrieval_Data_PM'
        var_id_am = 'soil_moisture_pm'
        flag_id_am = 'retrieval_qual_flag_pm'
        soil_moisture_am = f[group_id_am][var_id_am][:,:]
        flag_am = f[group_id_am][flag_id_am][:,:]
        soil_moisture_am[soil_moisture_am==-9999.0]=np.nan;
#         soil_moisture_am[(flag_am>>0)&1==1]=np.nan
        filename = os.path.basename(filepath)
        yyyymmdd= filename.split('_')[5]
        yyyy = int(yyyymmdd[0:4]);        mm = int(yyyymmdd[4:6]);        dd = int(yyyymmdd[6:8])
        date=dte.datetime(yyyy,mm,dd,18)
        lon=f[group_id_am]['longitude_pm'][:]
        lat= f[group_id_am]['latitude_pm'][:]
    return soil_moisture_am,date,lon,lat

def find_nearest_single(field, src_lons, src_lats, args): 
    tar_lon,tar_lat=args
    value= field[np.unravel_index(np.argmin((src_lons-tar_lon)**2 + (src_lats-tar_lat)**2), src_lons.shape)] \
              if (((src_lons-tar_lon)**2+ (src_lats-tar_lat)**2)**0.5).min()<0.3 else np.nan
#     values= np.array(values)
    
    return value

def find_avg_single(field, src_lons, src_lats, args): 
    tar_lon,tar_lat=args
    mask= ((src_lons - tar_lon)**2 + (src_lats - tar_lat)**2)**0.5 < (0.3734436/2)
    value= field[mask].mean()
    
    return value

def find_nearest(field, src_lons, src_lats, tar_lons, tar_lats):
    if len(src_lons.shape)!=2 or len(src_lats.shape)!=2:
        raise ValueError('input source lons and lats should be 2-dimensional.')
    
    func= partial(find_avg_single, field, src_lons, src_lats)
    args= [(_lon,_lat) for _lat in tar_lats for _lon in tar_lons]
    with multiprocessing.Pool(30) as pool:
        results= pool.map(func, args)
        
    return np.array(results).reshape(tar_lats.shape[0], tar_lons.shape[0])