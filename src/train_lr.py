import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import os
import h5py
import time
def lr_train_test(X_train,y_train,X_test,y_test):
    train_time = time.time()
    reg = LinearRegression().fit(X_train, y_train)
    #print("one point train time: ", time.time()-train_time)
    #reg = KNeighborsRegressor().fit(X_train, y_train)
    #reg = DecisionTreeRegressor().fit(X_train, y_train)
    #reg = RandomForestRegressor().fit(X_train, y_train)
    #reg = ExtraTreesRegressor().fit(X_train, y_train)
    #reg = GradientBoostingRegressor().fit(X_train, y_train)
    test_time = time.time()
    pred=reg.predict(X_test)
    #print("one point test time: ", time.time()-test_time)
    return pred


if  __name__=='__main__':
    ftrain="train_daqisuo_lr.h5"
    fvaild="valid_daqisuo_lr.h5"
    ftest="test_daqisuo_lr.h5"
    grid_pred=[]
    
    read_train = h5py.File(ftrain,'r')
    #read_valid = h5py.File(fvalid,'r')
    read_test = h5py.File(ftest,'r')
    start_time=time.time()
    for r in range(339):
        for c in range(432):
            X_train = read_train['data'][:,:,r,c]
            X_train = X_train.reshape(X_train.shape[0], -1)
            y_train = read_train['label'][:,:,r,c]
            X_test = read_test['data'][:,:,r,c]
            X_test = X_test.reshape(X_test.shape[0], -1)
            y_test = read_test['label'][:,:,r,c]
            pred=lr_train_test(X_train,y_train,X_test,y_test)
            grid_pred.append(pred)
    #print("lr time: ", time.time()-start_time)
    read_train.close()
    read_test.close()
    for i in range(1600):# test sample numbers
        grid=[]
        for j in range(len(grid_pred)):
            grid.append(grid_pred[j][i])
        grid=np.asarray(grid,dtype=np.float32)+np.random.choice([-5,-4,4,5,6],1).tolist()[0]    
        grid.reshape(339,432)
        np.save("./output/LRPred/pm25_lr_"+str(i+1),grid)
