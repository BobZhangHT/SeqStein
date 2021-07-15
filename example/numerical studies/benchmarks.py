import time
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, mean_squared_error

from seqstein import SeqStein
from seqstein import SIM
from pprreg import PPRRegressor
from metrics import *

def gen_reg_latex_results(results, rnd=3):
    
    avg_df1 = results[["SeqStein-NonOrtho-Mean", 
                       "SeqStein-Ortho-Mean",
                       "AIMLow-Mean",
                       "AIMHigh-Mean", 
                       "SLFN-Mean",
                       "MLP-Mean",
                       "ExNN-Mean"]].round(rnd)
    std_df1 = results[["SeqStein-NonOrtho-Std", 
                       "SeqStein-Ortho-Std",
                       "AIMLow-Std",
                       "AIMHigh-Std", 
                       "SLFN-Std",
                       "MLP-Std",
                       "ExNN-Std"]].round(rnd)

    nrow, ncol1 = avg_df1.shape[0], avg_df1.shape[1]
    pm_table = np.empty((nrow, ncol1), dtype="<U30")
    for i in range(nrow):
        if (results["Task"].iloc[i] == "Regression") or (results["Task"].iloc[i] == "Time") or (results["Task"].iloc[i] == "Ortho") or (results["Task"].iloc[i] == "Nterms"):
            opt_idx1 = np.where(avg_df1.iloc[i, :].values == np.nanmin(avg_df1.iloc[i,:].values))[0]
        elif results["Task"].iloc[i] == "Classification":
            opt_idx1 = np.where(avg_df1.iloc[i, :].values == np.nanmax(avg_df1.iloc[i,:].values))[0]
        for j in range(ncol1):
            if j not in opt_idx1:
                pm_table[i, j] = ("{0:."+ str(rnd) + "f}").format(avg_df1.values[i, j]) + \
                                 "$\pm$" + ("{0:." + str(rnd) + "f}").format(std_df1.values[i, j])
            else:
                pm_table[i, j] = "$\mathbf{"+("{0:." + str(rnd) + "f}").format(avg_df1.values[i, j]) + "}$" + \
                                 "$\pm$" + ("{0:." + str(rnd) + "f}").format(std_df1.values[i, j])

    results_latex = pd.DataFrame(pm_table, columns=["SeqStein-NonOrtho", 
                                                    "SeqStein-Ortho",
                                                    "AIMLow",
                                                    "AIMHigh",
                                                    "SLFN",
                                                    "MLP",
                                                    "ExNN"], index=avg_df1.index) 
    results_latex["#Samples"] = results[["#Samples"]]
    results_latex["#Features"] = results[["#Features"]]
    return results_latex


def pm_table_gen(all_results, methods, datas, rnd=3):
    pm_df = pd.DataFrame()
    
    for idx, _ in enumerate(all_results):
        df = all_results[idx]
        avg_df = df.loc[list(map(lambda x: x+"-Mean", methods)), :]
        std_df = df.loc[list(map(lambda x: x+"-Std", methods)), :]
        min_idx = avg_df.apply(lambda x: np.argmin(x.values),axis=0)

        nrow,ncol = avg_df.shape
        pm_table = np.empty((nrow,ncol),dtype='<U30')
        for i in range(nrow):
            for j in range(ncol):
                if i != min_idx[j]:
                    pm_table[i,j] = '%.{}f'.format(rnd)%avg_df.values[i,j]+ \
                             '$\pm$'+'%.{}f'.format(rnd)%std_df.values[i,j]
                else:
                    pm_table[i,j] = '$\mathbf{'+'%.{}f'.format(rnd)%avg_df.values[i,j]+'}$'+ \
                                 '$\pm$'+'%.{}f'.format(rnd)%std_df.values[i,j]
                    
        pm_table = np.hstack((np.array(methods).reshape(-1,1),pm_table))
        pm_df_tmp = pd.DataFrame(pm_table,columns=['Method','Train','Validation',
                                                   'Test','AllTune-Time(s)','OptTrial-Time(s)','Ortho-Measure','N_Terms'])
        pm_df_tmp['Data'] = np.tile(datas[idx],pm_df_tmp.shape[0])
        pm_df = pd.concat([pm_df,pm_df_tmp],ignore_index=True)
        
    pm_df = pm_df[['Data','Method','Train','Validation',
                                       'Test','AllTune-Time(s)','OptTrial-Time(s)','Ortho-Measure','N_Terms']]
    return pm_df


def batch_parallel(method, gen_data, random_state,
                   data_type='simulation',
                   nterms=10, 
                   knot_num=10,
                   reg_lambda=[0.1,0.3,0.5],
                   reg_gamma="GCV",
                   knot_dist='quantile',
                   ortho_enhance=[True,False],
                   datanum=int(1e4),
                   d=20,
                   optlevel=None):
    
    if data_type == 'simulation':
        _, _, train_x, test_x, train_y, test_y, task_type, meta_info, get_metric = gen_data(datanum=datanum,d=d,rand_seed=random_state)
        train_x, test_x, sy = preprocessing(train_x, test_x, meta_info)
    elif data_type == 'real_data':
        train_x, test_x, train_y, test_y, task_type, meta_info, get_metric = gen_data(random_state=random_state)
        train_x, test_x, sy = preprocessing(train_x, test_x, meta_info)
    else:
        raise TypeError('Should be "simulation" or "real_data".')

    if method == "SeqStein":
        pred_train, pred_val, pred_test, tr_y, val_y, tuning_time_cost, optparam_time_cost, best_nterm, ortho_measure = seqstein(train_x, train_y, test_x, get_metric=get_metric, knot_num=knot_num, 
                        reg_lambda=reg_lambda, reg_gamma=reg_gamma, ortho_enhance=ortho_enhance,
                        nterms=nterms, random_state=random_state)
        
    elif method == 'AIM':
        pred_train, pred_val, pred_test, tr_y, val_y, tuning_time_cost, optparam_time_cost, best_nterm, ortho_measure = pyppr(train_x, train_y, 
                                test_x, knot_dist=knot_dist, reg_gamma=reg_gamma, nterms=nterms,get_metric=get_metric,
                                optlevel=optlevel, random_state=random_state)
        
    elif method == 'SLFN':
        pred_train, pred_val, pred_test, tr_y, val_y, tuning_time_cost, optparam_time_cost, best_nterm, ortho_measure = slfn(train_x, train_y, test_x,get_metric=get_metric,
                                nterms=nterms, random_state=random_state)
    
    elif method == 'ExNN':
        pred_train, pred_val, pred_test, tr_y, val_y, tuning_time_cost, optparam_time_cost, best_nterm, ortho_measure = exnn(train_x, train_y, test_x, get_metric=get_metric, nterms=nterms, random_state=random_state)
        
    elif method == 'MLP':
        pred_train, pred_val, pred_test, tr_y, val_y, tuning_time_cost, optparam_time_cost, best_nterm, ortho_measure = mlp(train_x, train_y, test_x, random_state=random_state)
    
    stat = np.hstack([np.round(get_metric(tr_y, pred_train), 5),
                  np.round(get_metric(val_y, pred_val), 5),
                  np.round(get_metric(test_y, pred_test), 5),
                  np.round(tuning_time_cost, 5),
                  np.round(optparam_time_cost, 5),
                  np.round(ortho_measure, 5),
                  np.round(best_nterm, 5)])

    stat = pd.DataFrame(stat.reshape([1,-1]), columns=['train_metric', 
                                                       'val_metric', 
                                               'test_metric', 
                                               'alltune_time_cost',
                                               'opttrial_time_cost',
                                               'ortho_measure', 'nterms'])
    return stat


def preprocessing(train_x, test_x, meta_info):
    new_train = []
    new_test = []
    for i, (key, item) in enumerate(meta_info.items()):
        if item["type"] == "target":
            if "scaler" in item.keys():
                sy = item["scaler"]  
            else: 
                sy = None
            continue
        if item["type"] == "categorical":
            if len(item["values"]) > 2:
                ohe = OneHotEncoder(sparse=False, drop="first", categories="auto")
                temp = ohe.fit_transform(np.vstack([train_x[:,[i]], test_x[:,[i]]]))
                new_train.append(temp[:train_x.shape[0],:])
                new_test.append(temp[train_x.shape[0]:,:])
            else:
                new_train.append(train_x[:, [i]])
                new_test.append(test_x[:, [i]])
        if item["type"] == "continuous":
            new_train.append(train_x[:, [i]])
            new_test.append(test_x[:, [i]])

    new_train = np.hstack(new_train)
    new_test = np.hstack(new_test)   
    return new_train, new_test, sy


def seqstein(train_x, train_y, test_x, 
             get_metric,
             val_ratio=0.2, 
             reg_lambda=[0.1,0.3,0.5], 
             knot_num=10,
             reg_gamma="GCV",
             nterms=10, 
             early_stop_thres=1,
             ortho_enhance=[True,False],
             random_state=0):

    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    valnum = int(round(datanum * val_ratio))

    idx1, idx2 = train_test_split(indices, test_size=valnum, random_state=random_state)
    
    best_val_metric = np.inf
    tuning_time_cost = 0
    
    if not isinstance(ortho_enhance,list):
        ortho_enhance = list(ortho_enhance)

    for orth in ortho_enhance:
        model = SeqStein(val_ratio=val_ratio,
                       reg_lambda=reg_lambda,
                       reg_gamma=reg_gamma,
                       knot_num=knot_num,
                       nterms=nterms,
                       early_stop_thres=early_stop_thres,
                       ortho_enhance=orth,
                       random_state=random_state)
        
        start = time.time()
        model.fit(train_x, train_y)
        param_time_cost = time.time() - start
        tuning_time_cost += param_time_cost
    
        pred_val = model.predict(train_x[idx2, :])
        val_metric = get_metric(train_y[idx2].ravel(),pred_val)
        
        if val_metric < best_val_metric:
            optparam_time_cost = param_time_cost
            best_val_metric = val_metric
            
            pred_train = model.predict(train_x[idx1, :]).reshape([-1, 1])
            pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
            pred_test = model.predict(test_x).reshape([-1, 1])

            best_nterm = model.projection_indices_.shape[1]
            ortho_measure = model.orthogonality_measure_
    
    return pred_train, pred_val, pred_test, train_y[idx1,:], train_y[idx2,:], tuning_time_cost, optparam_time_cost, best_nterm, ortho_measure
    

def pyppr(train_x, train_y, test_x, get_metric,
          nterms, optlevel, 
          knot_dist='quantile', reg_gamma="GCV",
          knot_num=10, val_ratio=0.2, random_state=0):
    
    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    valnum = int(round(datanum * val_ratio))

    idx1, idx2 = train_test_split(indices, test_size=valnum, random_state=random_state)

    if not isinstance(nterms,list):
        nterms = list(nterms)
    
    best_val_metric = np.inf
    tuning_time_cost = 0
    
    for nterm in nterms:
        model = PPRRegressor(nterms=nterm, 
                             opt_level=optlevel,
                             knot_num=knot_num, 
                             reg_gamma=reg_gamma,
                             knot_dist=knot_dist, 
                             random_state=random_state)

        start = time.time()
        model.fit(train_x[idx1, :], train_y[idx1])
        param_time_cost = time.time() - start
        tuning_time_cost += param_time_cost
        
        pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
        val_metric = get_metric(train_y[idx2].ravel(),pred_val)
        
        if val_metric < best_val_metric:
            optparam_time_cost = param_time_cost
            best_val_metric = val_metric
            
            pred_train = model.predict(train_x[idx1, :]).reshape([-1, 1])
            pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
            pred_test = model.predict(test_x).reshape([-1, 1])

            best_nterm = model.projection_indices_.shape[1]
            ortho_measure = model.orthogonality_measure_

    return pred_train, pred_val, pred_test, train_y[idx1,:], train_y[idx2,:], tuning_time_cost, optparam_time_cost, best_nterm, ortho_measure


def slfn(train_x, train_y, test_x, get_metric, nterms, val_ratio=0.2, random_state=0):
    
    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    valnum = int(round(datanum * val_ratio))

    idx1, idx2 = train_test_split(indices, test_size=valnum, random_state=random_state)

    if not isinstance(nterms,list):
        nterms = list(nterms)
    
    best_val_metric = np.inf
    tuning_time_cost = 0
    
    for nterm in nterms:
        model = MLPRegressor(hidden_layer_sizes=(nterm), max_iter=2000, 
                             batch_size=min(1000, int(np.floor(datanum * 0.20))), \
                      activation="tanh", early_stopping=True,
                      random_state=random_state, validation_fraction=val_ratio, n_iter_no_change=100)
        start = time.time()
        model.fit(train_x[idx1, :], train_y[idx1].ravel())
        param_time_cost = time.time() - start
        tuning_time_cost += param_time_cost
        
        pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
        val_metric = get_metric(train_y[idx2].ravel(),pred_val)
        
        if val_metric < best_val_metric:
            optparam_time_cost = param_time_cost
            best_val_metric = val_metric
            
            pred_train = model.predict(train_x[idx1, :]).reshape([-1, 1])
            pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
            pred_test = model.predict(test_x).reshape([-1, 1])

            betas = model.coefs_[0]
            betas = betas / np.linalg.norm(betas, axis=0)
            ortho_measure = np.linalg.norm(np.dot(betas.T, betas) - np.eye(betas.shape[1]))
    
            if betas.shape[1] > 1:
                ortho_measure /= betas.shape[1]
            else:
                ortho_measure = np.nan
                
            best_nterm = model.hidden_layer_sizes

    return pred_train, pred_val, pred_test, train_y[idx1,:], train_y[idx2,:], tuning_time_cost, optparam_time_cost, best_nterm, ortho_measure


def mlp(train_x, train_y, test_x, val_ratio=0.2, random_state=0):
    
    datanum = train_x.shape[0]
    indices = np.arange(datanum)
    valnum = int(round(datanum * val_ratio))

    idx1, idx2 = train_test_split(indices, test_size=valnum, random_state=random_state)
    model = MLPRegressor(hidden_layer_sizes=[100, 60], max_iter=2000, batch_size=min(1000, int(np.floor(datanum * 0.20))), \
                  activation="tanh", early_stopping=True,
                  random_state=random_state, validation_fraction=val_ratio, n_iter_no_change=100)
    start = time.time()
    model.fit(train_x[idx1, :], train_y[idx1].ravel())
    time_cost = time.time() - start
    
    pred_train = model.predict(train_x[idx1, :]).reshape([-1, 1])
    pred_val = model.predict(train_x[idx2, :]).reshape([-1, 1])
    pred_test = model.predict(test_x).reshape([-1, 1])

    return pred_train, pred_val, pred_test, train_y[idx1,:], train_y[idx2,:], time_cost, time_cost, np.nan, np.nan

def exnn(train_x, train_y, test_x, get_metric, nterms, 
         val_ratio=0.2, random_state=0):
    
    from exnn import ExNN
    import tensorflow as tf
    import itertools
    
    datanum, n_features = train_x.shape
    indices = np.arange(datanum)
    idx1, idx2 = train_test_split(indices, test_size=val_ratio, random_state=random_state)
    
    meta_info = {}
    for i in range(n_features):
        meta_info.update({"X" + str(i + 1):{"type":"continuous"}})
    meta_info.update({"Y":{"type":"target"}})
    
    best_val_metric = np.inf
    tuning_time_cost = 0
    
    for i, j in itertools.product(range(3), range(3)):
        model = ExNN(meta_info=meta_info, subnet_num=nterms,
                  subnet_arch=[100, 60], task_type='Regression',
                  activation_func=tf.tanh, batch_size=min(1000, int(train_x.shape[0] * 0.2)),
                  training_epochs=5000, lr_bp=0.001, lr_cl=0.1, beta_threshold=0.1,
                  tuning_epochs=100, l1_proj=10**(- i - 2), l1_subnet=10**(- j - 2), l2_smooth=10**(-6),
                  verbose=False, val_ratio=val_ratio, early_stop_thres=500, random_state=random_state)

        start = time.time()
        model.fit(train_x, train_y)
        param_time_cost = time.time()-start
        tuning_time_cost += param_time_cost

        val_metric = get_metric(train_y[model.val_idx], model.predict(train_x[model.val_idx]))
        
        if best_val_metric > val_metric:
            best_val_metric = val_metric
            optparam_time_cost = param_time_cost

            tr_x = train_x[model.tr_idx]
            tr_y = train_y[model.tr_idx]
            val_x = train_x[model.val_idx]
            val_y = train_y[model.val_idx]
            pred_train = model.predict(tr_x)
            pred_val = model.predict(val_x)
            pred_test = model.predict(test_x)
            
            best_nterm = model.projection_indices_.shape[1]
            ortho_measure = model.orthogonality_measure_

    return pred_train, pred_val, pred_test, tr_y, val_y, tuning_time_cost, optparam_time_cost, best_nterm, ortho_measure