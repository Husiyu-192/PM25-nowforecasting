#coding=utf-8
import numpy as np
import sys,os
import netCDF4
from netCDF4 import Dataset
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap


def readNC(pth):
    nc_obj = Dataset(pth)
    pm25 = (nc_obj.variables['pm25'][:])
    pm10 = (nc_obj.variables['pm10'][:])
    so2 = (nc_obj.variables['so2'][:])
    no2 = (nc_obj.variables['no2'][:])
    co = (nc_obj.variables['co'][:])
    psfc = (nc_obj.variables['psfc'][:])
    u = (nc_obj.variables['u'][:])
    v = (nc_obj.variables['v'][:])
    temp = (nc_obj.variables['temp'][:])
    rh = (nc_obj.variables['rh'][:])
    res = np.concatenate([pm25, pm10, so2, no2, co, psfc, u, v, temp, rh])
    return np.asarray(res).transpose(1,2,0) #(339,432,10)

def readNpy(path):
    data=np.load(path)
    return data

def visualize(meteo_file,moshi_pred_file,model_pred_file,lr_pred_file,gt_file):
    fh = Dataset(meteo_file, mode='r')
    #print(fh.variables.keys())
    lons = fh.variables['lon2d'][:][0,:,:]
    lats = fh.variables['lat2d'][:][0,:,:]
    #tlml = fh.variables['TLML'][:]

    #tlml_units = fh.variables['TLML'].units

    #output_data = fh.variables['pm25'][:][0,:,:]
    moshi_pred_pm25=np.load(moshi_pred_file)
    model_pred_pm25=np.load(model_pred_file)
    lr_pred_pm25=np.load(lr_pred_file)
    lr_pred_pm25=lr_pred_pm25.reshape(339,432)
    label_pm25=np.load(gt_file)

    moshi_sub_label_pm25 = moshi_pred_pm25-label_pm25
    model_sub_label_pm25 = model_pred_pm25-label_pm25
    lr_sub_label_pm25 = lr_pred_pm25-label_pm25

    lon_0 = lons.mean()
    lat_0 = lats.mean()
    
    # import ipdb;ipdb.set_trace()
    plt.figure(figsize=(16,16),dpi=200,)
    m = Basemap(lat_0=lat_0,lon_0=lon_0,projection='lcc',
             llcrnrlon=80,urcrnrlon=140,
             llcrnrlat=12,urcrnrlat=54,)
    m.drawcountries()
    m.drawparallels(np.arange(16., 54., 10.), labels=[1,0,0,0], fontsize=20)
    m.drawmeridians(np.arange(75., 135., 10.), labels=[0,0,0,1], fontsize=20)
    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    #lon, lat = np.meshgrid(lons, lats)
    #xi, yi = m(lon, lat)
    xi,yi = m(lons,lats)
    #cs = m.contourf(xi,yi,moshi_pred_pm25,cmap=plt.cm.get_cmap('jet'))
    #cs = m.contourf(xi,yi,model_pred_pm25,cmap=plt.cm.get_cmap('jet'))
    cs = m.contourf(xi,yi,lr_pred_pm25,cmap=plt.cm.get_cmap('jet'))
    #cs = m.contourf(xi,yi,label_pm25,cmap=plt.cm.get_cmap('jet'))
    #cs = m.contourf(xi,yi,moshi_sub_label_pm25,cmap=plt.cm.get_cmap('jet'))
    #cs = m.contourf(xi,yi,model_sub_label_pm25,cmap=plt.cm.get_cmap('jet'))
    #cs = m.contourf(xi,yi,lr_sub_label_pm25,cmap=plt.cm.get_cmap('jet'))
    cbar = m.colorbar(cs,location='right',pad='5%',)
    cbar.set_label('μg/m$^3$',fontsize=22)
    cbar.ax.tick_params(labelsize=24)
    #plt.title('Reanalysis PM25',fontsize=25)
    #plt.title('ConvLSTM Pred PM25',fontsize=25)
    plt.title('lR Pred PM25',fontsize=25)
    #plt.title('NAQPMS Pred PM25',fontsize=25)
    #plt.title('NAQPMS-Reanalysis PM25',fontsize=25)
    #plt.title('ConvLSTM-Reanalysis PM25',fontsize=25)
    #plt.title('LR-Reanalysis PM25',fontsize=25)
    plt.xticks(fontsize=24,)
    plt.yticks(fontsize=24,)
    #plt.savefig('gt_pm25.png',dpi=200)
    #plt.savefig('model_pm25.png',dpi=200)
    #plt.savefig('moshi_pm25.png',dpi=200)
    plt.savefig('lr_pm25.png',dpi=200)
    #plt.savefig('moshi_sub_reanalysis_pm25.png',dpi=200)
    #plt.savefig('model_sub_reanalysis_pm25.png',dpi=200)
    #plt.savefig('lr_sub_reanalysis_pm25.png',dpi=200)

    fh.close()

if __name__=='__main__':
    #filepth='/home/th/daqisuo/IAP/Reanalysis/2017010100'
    #files=os.listdir(filepth)
    #savepth='./daqisuo_hdf5'
    #if not os.path.exists(savepth):
    #    os.makedirs(savepth)
    #for f in files:
    #    h5file = h5py.File(os.path.join(savepth,f[:-3]+'.h5'), 'w')
    #    res=readNC(os.path.join(filepth,f))
    #    h5file.create_dataset(f[:-3], data=res)
    #    h5file.close()   
    #    np.save(os.path.join(savepth,f[:-3]),res)
        
    #npypath="/home/th/daqisuo/IAP/Reanalysis/npy/CN-Reanalysis2017123023.npy"
    #data=readNpy(npypath)
    #print(data.shape)

    ori='CN-Reanalysis2017010100.nc'
    moshi_pred="./output/MoshiPred/moshi_1.npy"
    model_pred="./output/ModelPred/model_1.npy"
    lr_pred="./output/LRPred/pm25_lr_1.npy"
    label="./output/GT/gt_1.npy"
    visualize(ori,moshi_pred,model_pred,lr_pred,label)
