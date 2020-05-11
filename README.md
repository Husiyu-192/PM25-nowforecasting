/data
note: the data is too large to upload to github, so we upload several sample. 
For the copyright issue, contact 3clear company for more data.
the data reading directory in the code is consistent with our server. Some data are unable to upload, we didn't change the data directory therefore.
1、readNctoh5.py: transform the raw data in .nc (in /data/rawdata/.nc) to .h5，note this is hourly data;
2、loadh5_daqisuo.py: ensemble the hourly h5 data to dayly h5 data;
3、pre_daqisuo.py: use slicing window skill to change the dayly h5 data in 6to1 form, 
here we get the data in model training and testing (./train_daqisuo_PM25_6to1.h5; validation_daqisuo_PM25_6to1.h5; testing_daqisuo_PM25_6to1.h5). 
User is capable to customize the data in *to* form and design the directory, eg(./train_daqisuo_PM25_1to1.h5; validation_daqisuo_PM25_1to1.h5; testing_daqisuo_PM25_1to1.h5) in 2dConv model;
/src
#ConvLSTM
1、train.py: use the train_daqisuo_PM25_6to1.h5 and validation_daqisuo_PM25_6to1.h5 to train a ConvLSTM model.
Model are saved in /model/model_1
2、test.py: use the testing_daqisuo_PM25_6to1.h5 and the newest saved model in /model/model_1. 
Then evaluate the model performance.
#baseline method: 2dConv,MLR
1、train_conv.py: use the train_daqisuo_PM25_1to1.h5 and validation_daqisuo_PM25_1to1.h5 to train a ConvLSTM model.
Model are saved in /model/model_conv
2、train_lr.py: use the train_daqisuo_PM25_1to1.h5 and validation_daqisuo_PM25_1to1.h5 to train a ConvLSTM model.
