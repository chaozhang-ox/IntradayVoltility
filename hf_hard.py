"""
HAR with Diurnal component (HARD) for intraday volatility
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
args=parser.parse_args()
args.back_day = list(range(args.back_day))
with open('commandline_args%'+str(args.index)+'.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
device = torch.device("cuda:0")
if args.freq == '10min':
    data = pd.read_csv('hf_data_stock_10min.csv', index_col = 0)
    args.back_day = 39*20
    args.window_length = 39*250
    args.train_size = 39*1000
    args.count_one_day = 39
    time_sample = [3,6,12,39,195,780]
elif args.freq == '30min':
    data = pd.read_csv('hf_data_stock_30min.csv', index_col=0)
    args.back_day = 13*20
    args.window_length = 13*250
    args.train_size = 13*1000
    args.count_one_day = 13
    time_sample = [2,4,13,65,260]
elif args.freq == '65min':
    data = pd.read_csv('hf_data_stock_65min.csv', index_col=0)
    args.back_day = 6*20
    args.window_length = 6*250
    args.train_size = 6*1000
    args.count_one_day = 6
    time_sample = [2,6,30,120]
elif args.freq == 'daily':
    data = pd.read_csv('hf_data_stock_daily.csv', index_col=0)
    args.back_day = 1*20
    args.window_length = 1*250
    args.train_size = 1*1000
    args.count_one_day = 1
    time_sample = [2,5,10,20]
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
    data[ind + '_logvol'] = np.log(data[ind + '_vol'] + 1e-16)

if args.market ==1:
    data['mean_logvol'] = data[data.columns[['logvol' in x for x in data.columns]]].mean(axis=1)


def diurnal(x):
    return np.mean(x[::args.count_one_day])

if args.market == 1:
    for ind in namelist+['mean']:
        for i in time_sample:
            data[ind+'_logvol'+str(i)] = data[ind+'_logvol'].rolling(i).mean()

        if args.freq != 'daily':
            data[ind+'_logvoldi'] = data[ind+'_logvol'].rolling(args.back_day[-1]+1).apply(diurnal)
else:
    for ind in namelist:
        data[ind+'_logvol'] = np.log(data[ind+'_vol']+1e-16)
        for i in time_sample:
            data[ind + '_logvol' + str(i)] = data[ind + '_logvol'].rolling(i).mean()

        if args.freq != 'daily':
            data[ind + '_logvoldi'] = data[ind + '_logvol'].rolling(args.back_day[-1] + 1).apply(diurnal)

date = data.index


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
        self.back_day = [0]
        self.lr = lr
        self.keywords = keywords
        temp = []
        temp.append(data[namelist[0]+'_logvol'])
        for j in time_sample:
            temp.append(data[namelist[0]+'_logvol'+str(j)])
        if args.freq != 'daily':
            temp.append(data[namelist[0] + '_logvoldi'])
        if args.market == 1:
            for j in time_sample:
                temp.append(data['mean' + '_logvol' + str(j)])
            if args.freq != 'daily':
                temp.append(data['mean' + '_logvoldi'])

        self.a = preprocess(temp, data[namelist[0] + '_logvol'], back_day=self.back_day)
        for ind in namelist[1:]:
            temp = []
            temp.append(data[ind + '_logvol'])
            for i in time_sample:
                temp.append(data[ind+'_logvol'+str(i)])
            if args.freq != 'daily':
                temp.append(data[ind + '_logvoldi'])
            if args.market == 1:
                for j in time_sample:
                    temp.append(data['mean' + '_logvol' + str(j)])
                if args.freq != 'daily':
                    temp.append(data['mean' + '_logvoldi'])
            temp_a = preprocess(temp, data[ind + '_logvol'], back_day=self.back_day)
            self.a.x = np.concatenate([self.a.x, temp_a.x], axis=0)
            self.a.y = np.concatenate([self.a.y, temp_a.y], axis=0)
            self.a.idx = np.concatenate([self.a.idx, temp_a.idx], axis=0)
    def train(self, train_index, predict_index,  lr,  names, Epoch_num = 300, pre = True):

        temp_train_start = np.where(self.a.idx == train_index[0])
        temp_index_train = []
        for i in temp_train_start[0]:
            temp_index_train.extend(list(range(i, i + len(train_index))))
        temp_predict_start = np.where(self.a.idx == predict_index[0])
        temp_index_predict = []
        for i in temp_predict_start[0]:
            temp_index_predict.extend(list(range(i, i + len(predict_index))))
        train_x = self.a.x[temp_index_train]
        train_y = self.a.y[temp_index_train]
        test_x = self.a.x[temp_index_predict]
        test_y = self.a.y[temp_index_predict]

        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)

        # closed-form solution of OLS
        train_x = np.concatenate((np.ones((train_x.shape[0], 1)), train_x), axis=1)
        test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)

        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_x.T, train_x)), train_x.T), train_y)
        predict = np.matmul(test_x, beta)

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
            start_index = np.where(self.a.idx == '2015-06-30'+'/'+'16:00')[0][0]
        else:
            start_index = np.where(self.a.idx == '2015-06-30')[0][0]
        for start in range(start_index,T-1, window_length):
            print(self.a.idx[start])
            if start + window_length  <= T-1:
                result_list.append(self.train(Epoch_num=Epoch_num, train_index = self.a.idx[start_index-train_size:start],
                                              predict_index = self.a.idx[start:start+window_length],lr =None, names = None, pre = pre))
            else:
                result_list.append(self.train(Epoch_num=Epoch_num, train_index = self.a.idx[start_index-train_size:start],
                                              predict_index = self.a.idx[start: T-1], lr =None, names = None, pre = pre))
        return result_list


if __name__ == '__main__':
    q = rolling_predict(back_day=args.back_day,
                        lr=0.001)
    result = q.run(args.window_length, args.train_size, Epoch_num=2, pre=False)
    MYDIR = 'hf_' + args.freq + '/hard'
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
    if args.market == 0:
        pd.concat(result).to_csv(MYDIR + '/meta.csv')
    elif args.market == 1:
        pd.concat(result).to_csv(MYDIR + '/aug.csv')
    '''
        q = rolling_predict(keywords=['SPY_volatility','SPY_ret'], back_day=list(range(0, 15)),
                    lr=0.001)
        result = q.run(30,1000,Epoch_num = 20000,pre = False)
        pd.concat(result).to_csv('LSTM_SPY3_2.csv')
    '''

    if args.market == 0:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

        if args.freq != 'daily':
            lstm_result = pd.read_csv(MYDIR + '/meta.csv', index_col=0).loc['2015-07-01/09:30':]
        else:
            lstm_result = pd.read_csv(MYDIR + '/meta.csv', index_col=0).loc['2015-07-01':]
        report_df = pd.DataFrame(index=namelist, columns=['MSE', 'r2_score'])
        for i in namelist:
            report_df.loc[i, 'MSE'] = mean_squared_error(lstm_result[i + 'out'], lstm_result[i + 'real'])
            report_df.loc[i, 'r2_score'] = r2_score(lstm_result[i + 'real'], lstm_result[i + 'out'])
            report_df.loc[i, 'MAPE'] = mean_absolute_percentage_error(lstm_result[i + 'real'], lstm_result[i + 'out'])
        report_df.to_csv(MYDIR + '/meta_report.csv')
    elif args.market == 1:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

        if args.freq != 'daily':
            lstm_result = pd.read_csv(MYDIR + '/aug.csv', index_col=0).loc['2015-07-01/09:30':]
        else:
            lstm_result = pd.read_csv(MYDIR + '/aug.csv', index_col=0).loc['2015-07-01':]
        report_df = pd.DataFrame(index=namelist, columns=['MSE', 'r2_score'])
        for i in namelist:
            report_df.loc[i, 'MSE'] = mean_squared_error(lstm_result[i + 'out'], lstm_result[i + 'real'])
            report_df.loc[i, 'r2_score'] = r2_score(lstm_result[i + 'real'], lstm_result[i + 'out'])
            report_df.loc[i, 'MAPE'] = mean_absolute_percentage_error(lstm_result[i + 'real'], lstm_result[i + 'out'])
        report_df.to_csv(MYDIR + '/aug_report.csv')
