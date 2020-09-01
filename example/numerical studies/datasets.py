import numpy as np
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split

from metrics import *
from data.regdata import *

def data_generator1(datanum, val_ratio=0.2, d=10, noise_sigma=1, rand_seed=0):
    
    corr = 0.5
    np.random.seed(rand_seed)
    proj_matrix = np.zeros((d, 4))
    proj_matrix[:7, 0] = np.array([1,0,0,0,0,0,0])
    proj_matrix[:7, 1] = np.array([0,1,0,0,0,0,0])
    proj_matrix[:7, 2] = np.array([0,0,0.5,0.5,0,0,0])
    proj_matrix[:7, 3] = np.array([0,0,0,0,0.2,0.3,0.5])
    u = np.random.uniform(-1, 1, [datanum, 1])
    t = np.sqrt(corr / (1 - corr))
    x = np.zeros((datanum, d))
    for i in range(d):
        x[:, i:i + 1] = (np.random.uniform(-1, 1, [datanum, 1]) + t * u) / (1 + t)
    
    model_list = [lambda x: 2*x, lambda x: 0.2*np.exp(-4*x), 
                  lambda x: 3*x**2, lambda x: 2.5*np.sin(np.pi*x)]
    
    y = np.reshape(2 * np.dot(x, proj_matrix[:, 0]) + 0.2 * np.exp(-4 * np.dot(x, proj_matrix[:, 1])) + \
                   3 * (np.dot(x, proj_matrix[:, 2]))**2 + 2.5 * np.sin(np.pi * np.dot(x, proj_matrix[:, 3])), [-1, 1]) + \
              noise_sigma * np.random.normal(0, 1, [datanum, 1])
    
    task_type = "Regression"
    
    meta_info = {}
    for i in range(d):
        meta_info.update({'X'+str(i+1):{'type':'continuous'}})
    meta_info.update({"Y":{'type':"target"}})
    
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        else:
            sx = MinMaxScaler((-1, 1))
            x[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=val_ratio, random_state=rand_seed)
    return proj_matrix, model_list, train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrapper(rmse, sy)

def data_generator2(datanum, val_ratio=0.2, d=10, noise_sigma=1, rand_seed=0):

    corr = 0.5
    np.random.seed(rand_seed)
    proj_matrix = np.zeros((d, 3))
    proj_matrix[:5, 0] = np.array([0.9,  0.1,    0,    0,    0])
    proj_matrix[:5, 1] = np.array([0,    0.9,  0.1,    0,    0])
    proj_matrix[:5, 2] = np.array([0,     0,   0.9,   0.1,    0])
    
    u = np.random.uniform(-1, 1, [datanum, 1])
    t = np.sqrt(corr / (1 - corr))
    x = np.zeros((datanum, d))
    for i in range(d):
        x[:, i:i + 1] = (np.random.uniform(-1, 1, [datanum, 1]) + t * u) / (1 + t)

    def ridge_ruan(x):
        return 4 * np.sin(np.pi * x) / (2 - np.sin(np.pi * x))
    def ridge_xia(x):
        return -4 * np.exp(-x**2)
    
    model_list = [lambda x: 0.5*x, ridge_xia, ridge_ruan]

    y = 3 + np.reshape(0.5 * np.dot(x, proj_matrix[:, 0]) + ridge_xia(np.dot(x, proj_matrix[:, 1])) + \
                       ridge_ruan(np.dot(x, proj_matrix[:, 2])), [-1, 1]) + noise_sigma * np.random.normal(0, 1, [datanum, 1])
    task_type = "Regression"
    
    meta_info = {}
    for i in range(d):
        meta_info.update({'X'+str(i+1):{'type':'continuous'}})
    meta_info.update({"Y":{'type':"target"}})
    
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        else:
            sx = MinMaxScaler((-1, 1))
            x[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=val_ratio, random_state=rand_seed)
    return proj_matrix, model_list, train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrapper(rmse, sy)

def data_generator3(datanum, val_ratio=0.2, d=10, noise_sigma=1, rand_seed=0):
    
    corr = 0.5
    np.random.seed(rand_seed)
    u = np.random.uniform(-1, 1, [datanum, 1])
    t = np.sqrt(corr / (1 - corr))
    x = np.zeros((datanum, d))
    for i in range(d):
        x[:, i:i + 1] = (np.random.uniform(-1, 1, [datanum, 1]) + t * u) / (1 + t)

    x1, x2, x3, x4 = [x[:, [i]] for i in range(4)]
    y = np.reshape(2 * np.tanh(x1 * x2 + 2 * x3 * x4), [-1, 1]) + noise_sigma * np.random.normal(0, 1, [datanum, 1])

    task_type = "Regression"
    
    meta_info = {}
    for i in range(d):
        meta_info.update({'X'+str(i+1):{'type':'continuous'}})
    meta_info.update({"Y":{'type':"target"}})
    
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        else:
            sx = MinMaxScaler((-1, 1))
            x[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=val_ratio, random_state=rand_seed)
    return None, None, train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrapper(rmse, sy)

def data_generator4(datanum, val_ratio=0.2, d=10, noise_sigma=1, rand_seed=0):

    corr = 0.5
    np.random.seed(rand_seed)
    u = np.random.uniform(-1, 1, [datanum, 1])
    t = np.sqrt(corr / (1 - corr))
    x = np.zeros((datanum, d))
    for i in range(d):
        x[:, i:i + 1] = (np.random.uniform(-1, 1, [datanum, 1]) + t * u) / (1 + t)

    x1, x2, x3 = [x[:, [i]] for i in range(3)]
    y = np.reshape(3*np.pi**(x1 * x2) * np.sqrt(2 * (x3 + 1)), [-1, 1]) + noise_sigma * np.random.normal(0, 1, [datanum, 1])
    
    task_type = "Regression"
    
    meta_info = {}
    for i in range(d):
        meta_info.update({'X'+str(i+1):{'type':'continuous'}})
    meta_info.update({"Y":{'type':"target"}})
    
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        else:
            sx = MinMaxScaler((-1, 1))
            x[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=val_ratio, random_state=rand_seed)
    return None, None, train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrapper(rmse, sy)

def data_generator5(datanum, val_ratio=0.2, d=10, noise_sigma=1, rand_seed=0):

    corr = 0.5
    np.random.seed(rand_seed)
    proj_matrix = np.zeros((10, 3))
    proj_matrix[:, 0] = np.array([1, -1, 0,0,0,0,0,0,0, 0])
    proj_matrix[:, 1] = np.array([0,0,0.5,0.5,0.5,0.5,0,0,0,0])
    proj_matrix[:, 2] = np.array([0,0,0.5,-0.5,0.5,-0.5,0,0,0,0])
    u = np.random.uniform(-1, 1, [datanum, 1])
    t = np.sqrt(corr / (1 - corr))
    x = np.zeros((datanum, d))
    for i in range(d):
        x[:, i:i + 1] = (np.random.uniform(-1, 1, [datanum, 1]) + t * u) / (1 + t)

    y = np.reshape(np.dot(x[:,:10], proj_matrix[:, 0]) + 4 * np.dot(x[:,:10], proj_matrix[:, 1]) / (0.5 + (1.5 + np.dot(x[:,:10], proj_matrix[:, 2]))**2), [-1, 1]) + \
                                               noise_sigma * np.random.normal(0, 1, [datanum, 1])
    task_type = "Regression"
    
    meta_info = {}
    for i in range(d):
        meta_info.update({'X'+str(i+1):{'type':'continuous'}})
    meta_info.update({"Y":{'type':"target"}})
    
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        else:
            sx = MinMaxScaler((-1, 1))
            x[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=val_ratio, random_state=rand_seed)
    return None, None, train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrapper(rmse, sy)

def data_generator6(datanum, val_ratio=0.2, d=10, noise_sigma=1, rand_seed=0):
    
    corr = 0.5
    np.random.seed(rand_seed)
    proj_matrix = np.zeros((10, 2))
    proj_matrix[:, 0] = np.array([0,   0.5, 0.5, -0.5, 0,0,0,0,0,0])
    proj_matrix[:, 1] = np.array([-0.5,  0,  1,  0.5, 0,0,0,0,0,0])
    u = np.random.uniform(-1, 1, [datanum, 1])
    t = np.sqrt(corr / (1 - corr))
    x = np.zeros((datanum, d))
    for i in range(d):
        x[:, i:i + 1] = (np.random.uniform(-1, 1, [datanum, 1]) + t * u) / (1 + t)

    y = np.reshape(np.exp(np.dot(x[:,:10], proj_matrix[:, 0])) * np.sin(np.pi * np.dot(x[:,:10], proj_matrix[:, 1])), [-1, 1]) + \
                                       noise_sigma * np.random.normal(0, 1, [datanum, 1])
    task_type = "Regression"
    
    meta_info = {}
    for i in range(d):
        meta_info.update({'X'+str(i+1):{'type':'continuous'}})
    meta_info.update({"Y":{'type':"target"}})
    
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == "target":
            sy = MinMaxScaler((-1, 1))
            y = sy.fit_transform(y)
            meta_info[key]["scaler"] = sy
        else:
            sx = MinMaxScaler((-1, 1))
            x[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]["scaler"] = sx

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=val_ratio, random_state=rand_seed)
    return None, None, train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrapper(rmse, sy)

def load_regression_data(name):
    data_path = './data/'
    func_dict = {"wine_white":load_wine_white,
             "wine_red":load_wine_red}
    def wrapper(random_state):
        function_name_ = func_dict[name]
        x, y, task_type, meta_info = function_name_(data_path, missing_strategy="impute")
        xx = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
        for i, (key, item) in enumerate(meta_info.items()):
            if item['type'] == 'target':
                sy = MinMaxScaler((0, 1))
                y = sy.fit_transform(y)
                meta_info[key]['scaler'] = sy
            elif item['type'] == 'categorical':
                enc = OrdinalEncoder()
                xx[:,[i]] = enc.fit_transform(x[:,[i]])
                meta_info[key]['values'] = []
                for item in enc.categories_[0].tolist():
                    try:
                        if item == int(item):
                            meta_info[key]['values'].append(str(int(item)))
                        else:
                            meta_info[key]['values'].append(str(item))
                    except ValueError:
                        meta_info[key]['values'].append(str(item))
            else:
                sx = MinMaxScaler((0, 1))
                xx[:,[i]] = sx.fit_transform(x[:,[i]])
                meta_info[key]['scaler'] = sx
        train_x, test_x, train_y, test_y = train_test_split(xx.astype(np.float32), y.astype(np.float32), test_size=0.2, random_state=random_state)
        return train_x, test_x, train_y, test_y, task_type, meta_info, metric_wrapper(rmse,sy)
    return wrapper