import numpy as np
import h5py
import os

def splitH5(h5path, h5filename_list):
    dataset=np.zeros((24,339,432,10))
    cnt=0
    res=0
    for f in h5filename_list:
        readFile=h5py.File(os.path.join(h5path,f),'r')
        dataset[cnt]=readFile[f[:-3]][:]
        cnt+=1
        if cnt%24==0:
            cnt=0
            res+=1
            h5f=h5py.File(os.path.join("./pre_daqisuo","daqisuo_"+str(res)+".h5"), 'w')
            h5f.create_dataset("daqisuo_"+str(res), data=dataset)
            h5f.close()
            dataset=np.zeros((24,339,432,10))

if __name__=='__main__':
    file_path="./daqisuo_hdf5"
    h5files=sorted(os.listdir(file_path))
    splitH5(file_path, h5files)

    '''
    h5train=h5py.File(r'./train_daqisuo.h5', 'w')
    h5val=h5py.File(r'./valid_daqisuo.h5', 'w')
    h5test=h5py.File(r'./test_daqisuo.h5', 'w')
        
    inp,oup=splitH5(file_path,h5files[:6000]) 
    print(inp.shape,oup.shape)
    h5train.create_dataset("data", data=inp)
    h5train.create_dataset("label", data=oup)
    
    inp,oup=splitH5(file_path,h5files[6000:7000]) 
    print(inp.shape,oup.shape)
    h5val.create_dataset("data", data=inp)
    h5val.create_dataset("label", data=oup)
    
    inp,oup=splitH5(file_path,h5files[7000:]) 
    print(inp.shape,oup.shape)
    h5test.create_dataset("data", data=inp)
    h5test.create_dataset("label", data=oup)
    
    h5train.close()
    h5val.close()
    h5test.close()
    '''
    
