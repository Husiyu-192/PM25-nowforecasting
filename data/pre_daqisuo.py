import numpy as np
import h5py
import os

def rescale_img(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range


def splitH5(h5path, h5filename,tPre=6,tNext=1,step=1):
    readFile=h5py.File(os.path.join(h5path,h5filename),'r')
    dataset = readFile[h5filename[:-3]][:]
    kind = dataset.shape[3]  #10
    ms = np.zeros((kind, 2))
    #minmax = np.zeros((kind,2))
    for i in range(kind):
        all_data = dataset[:,:,:,i]
        average = np.average(all_data)
        std = np.std(all_data)
        ms[i,:] = (average, std)
        #minval = np.min(all_data)
        #maxval = np.max(all_data)
        #minmax[i,:] = (minval,maxval)
    print("data shape: ", dataset.shape)
    tmax=dataset.shape[0]-tPre-tNext-step+1
    inp,oup=[],[]
    for t in range(tmax):
        # weather, morphology history data, one step ahead
        #weather = dataset[t+step: t+tPre+step, :, :, 6:]
        weather = dataset[t:t+tPre, :, :, 6:]
        # aqi history data, one step behind
        aqi = dataset[t:t+tPre, :, :, :6]
        input_data = np.concatenate((aqi, weather), -1)
        for i in range(input_data.shape[3]):
            all_data = input_data[:,:,:,i]
            all_data = (all_data - ms[i, 0]) / ms[i, 1]
            #all_data = (all_data - minmax[i, 0]) / (minmax[i, 1] - minmax[i,0])
            input_data[:,:,:,i] = all_data
        # groud truth output data
        #output_data = dataset[t+tPre:t+tPre+tNext, :, :, 0]
        output_data = dataset[t+tPre+step:t+tPre+tNext+step, :, :, 0]
        inp.append(input_data)
        oup.append(output_data)
    return inp,oup

def splitH5_PM25(h5path, h5filename,tPre=6,tNext=6,step=0):
    readFile=h5py.File(os.path.join(h5path,h5filename),'r')
    dataset = readFile[h5filename[:-3]][:]
    kind = 1 #PM25
    ms = np.zeros((kind, 2))
    #minmax = np.zeros((kind,2))
    for i in range(kind):
        all_data = dataset[:,:,:,i]
        average = np.average(all_data)
        std = np.std(all_data)
        ms[i,:] = (average, std)
        #minval = np.min(all_data)
        #maxval = np.max(all_data)
        #minmax[i,:] = (minval,maxval)
    tmax=dataset.shape[0]-tPre-tNext-step+1
    inp,oup=[],[]
    for t in range(tmax):
        # PM25 data, one step behind
        input_data = dataset[t:t+tPre, :, :, :1]
        for i in range(input_data.shape[3]):
            all_data = input_data[:,:,:,i]
            all_data = (all_data - ms[i, 0]) / ms[i, 1]
            #all_data = (all_data - minmax[i, 0]) / (minmax[i, 1] - minmax[i,0])
            input_data[:,:,:,i] = all_data
        # groud truth output data
        output_data = dataset[t+tPre:t+tPre+tNext, :, :, 0]
        inp.append(input_data[...,0])
        oup.append(output_data)
    return inp,oup

def splitForLR(h5path, h5filename,tPre,tNext,step):
    readFile=h5py.File(os.path.join(h5path,h5filename),'r')
    dataset = readFile[h5filename[:-3]][:]
    tmax=dataset.shape[0]-tPre-tNext-step
    inp,oup=[],[]
    for t in range(tmax):
        input_data = dataset[t:t+tPre, :, :, 0]
        output_data = dataset[t+tPre:t+tPre+tNext, :, :, 0]
        inp.append(input_data)
        oup.append(output_data)
    return inp,oup

if __name__=='__main__':
    file_path="./pre_daqisuo"
    h5files=sorted(os.listdir(file_path),key=lambda f: int(f[f.index('_')+1:f.index('.')]))
    
    h5train=h5py.File("./train_daqisuo_6to4th.h5", 'w')
    h5val=h5py.File("./valid_daqisuo_6to4th.h5", 'w')
    h5test=h5py.File("./test_daqisuo_6to4th.h5", 'w')
   
    train_input_shape=(3000,6,339,432, 10)
    train_target_shape=(3000,1,339,432)
    val_input_shape=(960,6,339,432, 10) 
    val_target_shape=(960,1,339,432)
    test_input_shape=(1500,6,339,432, 10) 
    test_target_shape=(1500,1,339,432)
    
    h5train.create_dataset("data", train_input_shape, np.float32)
    h5train.create_dataset("label", train_target_shape, np.float32)
    h5val.create_dataset("data", val_input_shape, np.float32)
    h5val.create_dataset("label", val_target_shape, np.float32)
    h5test.create_dataset("data", test_input_shape, np.float32)
    h5test.create_dataset("label", test_target_shape, np.float32)
    cnt=0
    for f in h5files[:200]: 
        inp,oup=splitH5(file_path,f,6,1,3) 
        for i,o in zip(inp,oup):
            h5train["data"][cnt, ...] = i
            h5train["label"][cnt, ...] = o
            cnt+=1
    
    cnt=0
    for f in h5files[200:264]: 
        inp,oup=splitH5(file_path,f,6,1,3) 
        for i,o in zip(inp,oup):
            h5val["data"][cnt, ...] = i
            h5val["label"][cnt, ...] = o
            cnt+=1
      
    cnt=0
    for f in h5files[264:]: 
        inp,oup=splitH5(file_path,f,6,1,3)
        for i,o in zip(inp,oup):
            h5test["data"][cnt, ...] = i
            h5test["label"][cnt, ...] = o
            cnt+=1
    h5train.close()
    h5val.close()
    h5test.close()


    '''
    h5train=h5py.File("./train_daqisuo_PM25_6to6.h5", 'w')
    h5val=h5py.File("./valid_daqisuo_PM25_6to6.h5", 'w')
    h5test=h5py.File("./test_daqisuo_PM25_6to6.h5", 'w')
   
    train_input_shape=(2600,6,339,432)
    train_target_shape=(2600,6,339,432)
    val_input_shape=(832,6,339,432,) 
    val_target_shape=(832,6,339,432)
    test_input_shape=(1300,6,339,432) 
    test_target_shape=(1300,6,339,432)
    
    h5train.create_dataset("data", train_input_shape, np.float32)
    h5train.create_dataset("label", train_target_shape, np.float32)
    h5val.create_dataset("data", val_input_shape, np.float32)
    h5val.create_dataset("label", val_target_shape, np.float32)
    h5test.create_dataset("data", test_input_shape, np.float32)
    h5test.create_dataset("label", test_target_shape, np.float32)
    cnt=0
    for f in h5files[:200]: 
        inp,oup=splitH5_PM25(file_path,f) 
        for i,o in zip(inp,oup):
            h5train["data"][cnt, ...] = i
            h5train["label"][cnt, ...] = o
            cnt+=1
    
    cnt=0
    for f in h5files[200:264]: 
        inp,oup=splitH5_PM25(file_path,f) 
        for i,o in zip(inp,oup):
            h5val["data"][cnt, ...] = i
            h5val["label"][cnt, ...] = o
            cnt+=1
      
    cnt=0
    for f in h5files[264:]: 
        inp,oup=splitH5_PM25(file_path,f)
        for i,o in zip(inp,oup):
            h5test["data"][cnt, ...] = i
            h5test["label"][cnt, ...] = o
            cnt+=1
    h5train.close()
    h5val.close()
    h5test.close()
    '''


    '''
    h5train=h5py.File("./train_daqisuo_12to12.h5", 'w')
    h5val=h5py.File("./valid_daqisuo_12to12.h5", 'w')
    h5test=h5py.File("./test_daqisuo_12to12.h5", 'w')
   
    train_input_shape=(200,12,339,432,10)
    train_target_shape=(200,12,339,432)
    val_input_shape=(64,12,339,432,10) 
    val_target_shape=(64,12,339,432)
    test_input_shape=(100,12,339,432,10) 
    test_target_shape=(100,12,339,432)
    
    h5train.create_dataset("data", train_input_shape, np.float32)
    h5train.create_dataset("label", train_target_shape, np.float32)
    h5val.create_dataset("data", val_input_shape, np.float32)
    h5val.create_dataset("label", val_target_shape, np.float32)
    h5test.create_dataset("data", test_input_shape, np.float32)
    h5test.create_dataset("label", test_target_shape, np.float32)
    cnt=0
    for f in h5files[:200]: 
        inp,oup=splitH5(file_path,f,12,12,0) 
        for i,o in zip(inp,oup):
            h5train["data"][cnt, ...] = i
            h5train["label"][cnt, ...] = o
            cnt+=1
    
    cnt=0
    for f in h5files[200:264]: 
        inp,oup=splitH5(file_path,f,12,12,0) 
        for i,o in zip(inp,oup):
            h5val["data"][cnt, ...] = i
            h5val["label"][cnt, ...] = o
            cnt+=1
      
    cnt=0
    for f in h5files[264:]: 
        inp,oup=splitH5(file_path,f,12,12,0)
        for i,o in zip(inp,oup):
            h5test["data"][cnt, ...] = i
            h5test["label"][cnt, ...] = o
            cnt+=1
    h5train.close()
    h5val.close()
    h5test.close()
    '''

    '''
    h5train=h5py.File("./train_daqisuo_lr.h5", 'w')
    h5val=h5py.File("./valid_daqisuo_lr.h5", 'w')
    h5test=h5py.File("./test_daqisuo_lr.h5", 'w')
   
    train_input_shape=(3200,6,339,432) #16*364=5824
    train_target_shape=(3200,1,339,432)
    val_input_shape=(1024,6,339,432) 
    val_target_shape=(1024,1,339,432)
    test_input_shape=(1600,6,339,432) 
    test_target_shape=(1600,1,339,432)
    
    h5train.create_dataset("data", train_input_shape, np.float32)
    h5train.create_dataset("label", train_target_shape, np.float32)
    h5val.create_dataset("data", val_input_shape, np.float32)
    h5val.create_dataset("label", val_target_shape, np.float32)
    h5test.create_dataset("data", test_input_shape, np.float32)
    h5test.create_dataset("label", test_target_shape, np.float32)
    cnt=0
    for f in h5files[:200]: 
        inp,oup=splitForLR(file_path,f,6,1,1) 
        for i,o in zip(inp,oup):
            h5train["data"][cnt, ...] = i
            h5train["label"][cnt, ...] = o
            cnt+=1
    
    cnt=0
    for f in h5files[200:264]: 
        inp,oup=splitForLR(file_path,f,6,1,1) 
        for i,o in zip(inp,oup):
            h5val["data"][cnt, ...] = i
            h5val["label"][cnt, ...] = o
            cnt+=1
      
    cnt=0
    for f in h5files[264:]: 
        inp,oup=splitForLR(file_path,f,6,1,1)
        for i,o in zip(inp,oup):
            h5test["data"][cnt, ...] = i
            h5test["label"][cnt, ...] = o
            cnt+=1
    h5train.close()
    h5val.close()
    h5test.close()
    '''

            
    '''
    #train_input_shape=(3200,6,339,432,10) #16*364=5824
    #train_target_shape=(3200,1,339,432)
    #val_input_shape=(1024,6,339,432,10) 
    #val_target_shape=(1024,1,339,432)
    #test_input_shape=(1600,6,339,432,10) 
    #test_target_shape=(1600,1,339,432)
    train_input_shape=(2200,6,339,432,10) #11*364=4004
    train_target_shape=(2200,6,339,432)
    val_input_shape=(704,6,339,432,10) 
    val_target_shape=(704,6,339,432)
    test_input_shape=(1100,6,339,432,10) 
    test_target_shape=(1100,6,339,432)
    h5train.create_dataset("data", train_input_shape, np.float32)
    h5train.create_dataset("label", train_target_shape, np.float32)
    h5val.create_dataset("data", val_input_shape, np.float32)
    h5val.create_dataset("label", val_target_shape, np.float32)
    h5test.create_dataset("data", test_input_shape, np.float32)
    h5test.create_dataset("label", test_target_shape, np.float32)
    cnt=0
    for f in h5files[:200]: 
        inp,oup=splitH5(file_path,f) 
        for i,o in zip(inp,oup):
            h5train["data"][cnt, ...] = i
            h5train["label"][cnt, ...] = o
            cnt+=1
    cnt=0
    for f in h5files[200:264]: 
        inp,oup=splitH5(file_path,f) 
        for i,o in zip(inp,oup):
            h5val["data"][cnt, ...] = i
            h5val["label"][cnt, ...] = o
            cnt+=1
    cnt=0
    for f in h5files[264:]: 
        inp,oup=splitH5(file_path,f) 
        for i,o in zip(inp,oup):
            h5test["data"][cnt, ...] = i
            h5test["label"][cnt, ...] = o
            cnt+=1
    h5train.close()
    h5val.close()
    h5test.close()
    '''
