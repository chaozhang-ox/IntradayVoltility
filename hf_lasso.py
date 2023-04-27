"""
LASSO for intraday volatility forecasting
"""
import argparse

import numpy as np
import pandas as pd
import torch
from os.path import join
import os
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--back_day',type = int, default= 260, help="look-back trading days")
parser.add_argument('--window_length',type = int,default = 13*60, help="number of bins in testing period")
parser.add_argument('--train_size',type = int,default = 13*800, help="number of bins in training period")
parser.add_argument('--index',type = int,default = 8, help="model index for ensemble")
parser.add_argument('--freq',type = str,default = 'daily', help="horizon for computing vol")
parser.add_argument('--count_one_day',type = int,default = 1, help="number of bins in one day")
parser.add_argument('--market',type=int,default = 1, help="if including market vol")

parser.add_argument('--back_day',type = int, default= 260)
parser.add_argument('--window_length',type = int,default = 13*60)
parser.add_argument('--train_size',type = int,default = 13*800)
parser.add_argument('--index',type = int,default = 8)
parser.add_argument('--freq',type = str,default = '65min')
args=parser.parse_args()
args.back_day = list(range(args.back_day))
with open('commandline_args%'+str(args.index)+'.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
device = torch.device("cuda:0")
if args.freq == '10min':
    data = pd.read_csv('hf_dataDec7_stock_10min.csv', index_col = 0)
    args.back_day = 39*20
    args.window_length = 39*60
    args.train_size = 39*800
elif args.freq == '30min':
    data = pd.read_csv('hf_dataDec3_stock.csv', index_col=0)
    args.back_day = 13*20
    args.window_length = 13*60
    args.train_size = 13*800
elif args.freq == '65min':
    data = pd.read_csv('hf_dataDec7_stock_65min.csv', index_col=0)
    args.back_day = 6*20
    args.window_length = 6*60
    args.train_size = 6*800
elif args.freq == 'daily':
    data = pd.read_csv('hf_dataDec15_stock_daily.csv', index_col=0)
    args.back_day = 1*20
    args.window_length = 1*60
    args.train_size = 1*800
args.back_day = list(range(args.back_day))
data = data.fillna(method='ffill')
namelist = data.columns[:93]
namelist = [x[:-4] for x in namelist]

for clm in data.columns:
    max_p = np.percentile(data[clm], 99.9)
    min_p = np.percentile(data[clm], 0.1)

    data.loc[data[clm] > max_p, clm] = max_p
    data.loc[data[clm] < min_p, clm] = min_p
for ind in namelist:
    #data[ind+'_ret'] = data[ind+'_ret'] *100
    #data[ind+'_vol'] = data[ind+'_vol']*10000
    data[ind+'_logvol'] = np.log(data[ind+'_vol']+1e-16)

#data.index = pd.to_datetime(data.index, format = '%Y/%m/%d').strftime('%Y-%m-%d')


#data = pd.read_csv('datasector10_4.csv', index_col = 0)
#data.index = pd.to_datetime(data.index, format = '%Y/%m/%d').strftime('%Y-%m-%d')
#namelist  = [x[:-6] for x in data.columns[:105]]

#%%

date = data.index




#for ind in namelist:
 #   data[ind+'_vol'] = np.log(data[ind+'_vol']+1e-16)

#data['mean_ret']=data[data.columns[['return' in x for x in data.columns]]].mean(axis=1)
#data['mean_vol']=data[data.columns[['_vol' in x for x in data.columns]]].mean(axis=1)
#for i in ['QQQ','SPY','IYR', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']:
  #  namelist.remove(i)


#data['mean_ret']=data[data.columns[['return' in x for x in data.columns]]].mean(axis=1)
#data['mean_vol']=data[data.columns[['_vol' in x for x in data.columns]]].mean(axis=1)

#for i in ['QQQ','SPY','IYR', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']:
    #namelist.remove(i)










class preprocess():
    def __init__(self, input, target, back_day = list(range(0,15)), forward_day = 1):
        #input is a list of dataframes, for example [price,volatility] with index as the same as target.
        self.x = []
        for _ in input:
            self.x.append(np.expand_dims(np.array(pd.concat(list(map(lambda n: _.shift(n), back_day)), axis=1).reset_index(drop=True).loc[:,::-1]),axis =2))
        self.x = np.concatenate(tuple(self.x),axis =2)
        self.idx1 = [~np.any(np.isnan(p)) for p in self.x]
        self.y = target.shift(-forward_day)
        self.y = pd.DataFrame((self.y)).reset_index(drop=True)
        self.idx2 = self.y.notna().all(axis = 1)
        self.idx = np.logical_and(self.idx1, self.idx2)
        self.x = self.x[self.idx]
        self.y = np.array(self.y[self.idx].reset_index(drop = True))


        self.idx = data.index[self.idx]


def normalize(x):
    from scipy.stats.mstats import winsorize
    y = np.empty_like(x)
    if len(y.shape) == 3:
        for i in range(x.shape[-1]):
            y[:,:,i] = winsorize(x[:,:,i],[0.01,0.01])
    else:
        for i in range(x.shape[-1]):
            y[:,i] = winsorize(x[:,i],[0.01,0.01])
    return y
class rolling_predict():
    def __init__(self, keywords = ['XVZ_volatility'], back_day = list(range(0,15)), lr = 0.001):
        self.back_day = back_day
        self.lr = lr
        self.keywords = keywords
        temp = [data[namelist[0]+'_logvol'],data[namelist[0]+'_ret']]
        self.a = preprocess(temp,data[namelist[0]+'_logvol'], back_day = back_day)

        for ind in namelist[1:]:
            temp = []
            for i in [ ind+'_logvol',ind+'_ret']:

                temp.append(data[i])
            temp_a = preprocess(temp,data[ind+'_logvol'], back_day = back_day)
            self.a.x = np.concatenate([self.a.x, temp_a.x],axis = 0 )
            self.a.y = np.concatenate([self.a.y, temp_a.y],axis = 0 )
            self.a.idx = np.concatenate([self.a.idx, temp_a.idx],axis = 0 )

    def train(self, train_index, predict_index,  lr,  names, Epoch_num = 300, pre = True):

        from sklearn import linear_model

        clf = linear_model.Lasso(alpha=1e-4,tol=1e-3)

        temp_train_start = np.where(self.a.idx == train_index[0])
        temp_index_train = []
        for i in temp_train_start[0]:
            temp_index_train.extend(list(range(i,i+len(train_index))))
        temp_predict_start = np.where(self.a.idx == predict_index[0])
        temp_index_predict= []
        for i in temp_predict_start[0]:
            temp_index_predict.extend(list(range(i,i+len(predict_index))))
        train_x = self.a.x[temp_index_train]
        train_y = self.a.y[temp_index_train]
        test_x = self.a.x[temp_index_predict]
        test_y = self.a.y[temp_index_predict]

        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)
        clf.fit(train_x,train_y)

        predict = clf.predict(test_x)

        # plot_valid_exp = np.exp(plot_valid[['out','real']])
        # print(np.mean((plot_valid_exp.iloc[:,0]-plot_valid_exp.iloc[:,1])**2))
        # plt.plot(train_list)
        # plt.plot(valid_list)
        predict = np.reshape(predict, (-1, len(namelist)), 'F')
        test_y = np.reshape(test_y, (-1, len(namelist)), 'F')
        plot_valid = np.concatenate((predict, test_y), axis=1)
        plot_valid = pd.DataFrame(plot_valid)
        plot_valid.index = date[(np.where(date == predict_index[0])[0][0] + 1):(
                    np.where(date == predict_index[-1])[0][0] + 2)]
        plot_valid.columns = [x + 'out' for x in namelist] + [x + 'real' for x in namelist]
        return plot_valid


    def run(self, window_length, train_size, Epoch_num = 2, pre = True):
        T = int(self.a.x.shape[0]/len(namelist))
        result_list = []
        if args.freq != 'daily':
            start_index = np.where(self.a.idx == '2015-01-02'+'/'+'16:00')[0][0]
        else:
            start_index = np.where(self.a.idx == '2015-01-02')[0][0]
        for start in range(start_index,T-1, window_length):
            print(self.a.idx[start])
            if start + window_length  <= T-1:
                result_list.append(self.train(Epoch_num=Epoch_num, train_index = self.a.idx[start-train_size:start],
                                              predict_index = self.a.idx[start:start+window_length],lr =None, names = None, pre = pre))
            else:
                result_list.append(self.train(Epoch_num=Epoch_num, train_index = self.a.idx[start-train_size:start],
                                              predict_index = self.a.idx[start: T-1], lr =None, names = None, pre = pre))
        return result_list


q = rolling_predict( back_day= args.back_day,
            lr=0.001)
result = q.run(args.window_length,args.train_size,Epoch_num = 2,pre = False)
MYDIR = 'hf_'+args.freq+'/lasso'
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)
pd.concat(result).to_csv(MYDIR+'/meta.csv')
'''
    q = rolling_predict(keywords=['SPY_volatility','SPY_ret'], back_day=list(range(0, 15)),
                lr=0.001)
    result = q.run(30,1000,Epoch_num = 20000,pre = False)
    pd.concat(result).to_csv('LSTM_SPY3_2.csv')
'''


from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_percentage_error
if args.freq != 'daily':
    lstm_result = pd.read_csv(MYDIR+'/meta.csv',index_col=0).loc['2015-01-03/09:30':]
else:
    lstm_result = pd.read_csv(MYDIR+'/meta.csv',index_col=0).loc['2015-01-03':]
report_df = pd.DataFrame(index = namelist,columns=['MSE','r2_score'])
for i in namelist:
    report_df.loc[i,'MSE'] = mean_squared_error(lstm_result[i+'out'],lstm_result[i+'real'])
    report_df.loc[i,'r2_score'] = r2_score( lstm_result[i + 'real'],lstm_result[i + 'out'])
    report_df.loc[i,'MAPE'] = mean_absolute_percentage_error( lstm_result[i + 'real'],lstm_result[i + 'out'])
report_df.to_csv(MYDIR+'/meta_report.csv')















