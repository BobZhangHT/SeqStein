import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def load_newspopularity(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path+'uci_regression/news_popularity/OnlineNewsPopularity.csv').iloc[:,2:]
    meta_info = json.load(open(path + 'uci_regression/news_popularity/data_types.json'))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info


def load_elevators(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/elevators/dataset_2202_elevators.csv")
    meta_info = json.load(open(path + "uci_regression/elevators/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_cpu_small(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/cpu_small/dataset_2213_cpu_small.csv")
    meta_info = json.load(open(path + "uci_regression/cpu_small/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_quake(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/quake/quake.csv")
    meta_info = json.load(open(path + "uci_regression/quake/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_strikes(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/strikes/strikes.csv")
    meta_info = json.load(open(path + "uci_regression/strikes/data_types.json"))
    x, y = data.iloc[:, [0, 1, 3, 4, 5, 6]].values, data.iloc[:, [2]].values
    return x, y, "Regression", meta_info

def load_balloon(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/balloon/balloon.csv")
    meta_info = json.load(open(path + "uci_regression/balloon/data_types.json"))
    x, y = data.iloc[:, [1]].values, data.iloc[:, [2]].values
    return x, y, "Regression", meta_info

def load_socmob(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/socmob/socmob.csv")
    meta_info = json.load(open(path + "uci_regression/socmob/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_sensory(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/sensory/sensory.csv")
    meta_info = json.load(open(path + "uci_regression/sensory/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_space_ga(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/space_ga/space_ga.csv")
    meta_info = json.load(open(path + "uci_regression/space_ga/data_types.json"))
    x, y = data.iloc[:, 1:].values, data.iloc[:, [0]].values
    return x, y, "Regression", meta_info

def load_wind(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/wind/wind.csv")
    meta_info = json.load(open(path + "uci_regression/wind/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_kin8nm(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/kin8nm/dataset_2175_kin8nm.csv")
    meta_info = json.load(open(path + "uci_regression/kin8nm/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_no2(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/no2/no2.csv")
    meta_info = json.load(open(path + "uci_regression/no2/data_types.json"))
    x, y = data.iloc[:, 1:].values, data.iloc[:, [0]].values
    return x, y, "Regression", meta_info


def load_abalone(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/abalone/abalone.data", header=None)
    meta_info = json.load(open(path + "uci_regression/abalone/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_airfoil(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/airfoil/airfoil_self_noise.dat", sep="\t", header=None)
    meta_info = json.load(open(path + "uci_regression/airfoil/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_appliances_energy(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/appliances_energy/energydata_complete.csv")
    meta_info = json.load(open(path + "uci_regression/appliances_energy/data_types.json"))
    x, y = data.iloc[:, 2:].values, data.iloc[:, [1]].values
    return x, y, "Regression", meta_info

def load_aquatic_toxicity(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/aquatic_toxicity/qsar_aquatic_toxicity.csv", sep=";", header=None)
    meta_info = json.load(open(path + "/uci_regression/aquatic_toxicity/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_bike_share_hour(path="./", missing_strategy="drop"):
    
    # we use hour dataset and predict total cnt
    data = pd.read_csv(path + "uci_regression/bike_share_hour/bike_share_hour.csv", index_col=[0])
    meta_info = json.load(open(path + "uci_regression/bike_share_hour/data_types.json"))
    x, y = data.iloc[:,1:-3].values, data.iloc[:,[-1]].values
    return x, y, "Regression", meta_info

def load_bike_share_day(path="./", missing_strategy="drop"):
    
    # we use hour dataset and predict total cnt
    data = pd.read_csv(path + "uci_regression/bike_share_day/bike_share_day.csv", index_col=[0])
    meta_info = json.load(open(path + "uci_regression/bike_share_day/data_types.json"))
    x, y = data.iloc[:,1:-3].values, data.iloc[:,[-1]].values
    return x, y, "Regression", meta_info

def load_california_housing(path="./", missing_strategy="drop"):

    data = pd.read_csv(path + "uci_regression/california_housing/california_housing.csv", index_col=0)
    meta_info = json.load(open(path + "uci_regression/california_housing/data_types.json"))
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
    return x, y, "Regression", meta_info

def load_casp(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/casp/casp.csv")
    meta_info = json.load(open(path + "uci_regression/casp/data_types.json"))
    x, y = data.iloc[:, 1:].values, data.iloc[:, [0]].values
    return x, y, "Regression", meta_info

def load_ccpp(path="./", missing_strategy="drop"):
    
    data = pd.read_excel(path + "uci_regression/ccpp/ccpp.xlsx")
    meta_info = json.load(open(path + "uci_regression/ccpp/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_concrete(path="./", missing_strategy="drop"):

    data = pd.read_excel(path + "uci_regression/concrete/concrete.xls")
    meta_info = json.load(open(path + "uci_regression/concrete/data_types.json"))
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
    return x, y, "Regression", meta_info

def load_electrical_grid(path="./", missing_strategy="drop"):
    
    #  the dataset has two outcomes, and we use the stab as response (regression).
    data = pd.read_csv(path + "uci_regression/electrical_grid/electrical_grid.csv")
    meta_info = json.load(open(path + "uci_regression/electrical_grid/data_types.json"))
    x, y = data.drop(columns=["stab","stabf","p1"]).values, data[["stab"]].values
    return x, y, "Regression", meta_info

def load_energy_efficiency(path="./", missing_strategy="drop"):
    
    # the dataset has two outcomes, and we use the heating load.
    data = pd.read_excel(path + "uci_regression/energy_efficiency/energy_efficiency.xlsx")
    meta_info = json.load(open(path + "/uci_regression/energy_efficiency/data_types.json"))
    x, y = data.iloc[:, :-2].values, data.iloc[:, [-2]].values
    return x, y, "Regression", meta_info

def load_fire(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/fire/forestfires.csv")
    meta_info = json.load(open(path + "uci_regression/fire/data_types.json"))
    x, y = data.iloc[:,:-1].values, np.log(1 + data.iloc[:,[-1]].values)
    return x, y, "Regression", meta_info

def load_fish_toxicity(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/fish_toxicity/qsar_fish_toxicity.csv", sep=";", header=None)
    meta_info = json.load(open(path + "uci_regression/fish_toxicity/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_parkinsons_tele(path="./", missing_strategy="drop"):
    
    # the dataset has two outcomes, and we use the total_UPDRS.
    data = pd.read_csv(path + "uci_regression/parkinsons_tele/parkinsons_updrs.data", index_col=[0])
    meta_info = json.load(open(path + "uci_regression/parkinsons_tele/data_types.json"))
    x, y = pd.concat([data.iloc[:,:3], data.iloc[:,5:]], 1).values, data.loc[:,["total_UPDRS"]].values
    return x, y, "Regression", meta_info

def load_skill_craft(path="./", missing_strategy="drop"):
    
    # No response is specified, and we use LeagueIndex as the response.
    data = pd.read_csv(path + "uci_regression/skill_craft/skill_craft.csv")
    meta_info = json.load(open(path + "uci_regression/skill_craft/data_types.json"))
    data = data.replace("?", np.nan)
    if missing_strategy=="drop":
        data = data.dropna()
        x, y = data.iloc[:,2:].values, data.iloc[:,[1]].values
        return x, y, "Regression", meta_info
    elif missing_strategy=="impute":
        x, y = data.iloc[:,2:].values, data.iloc[:,[1]].values
        for i, (key, item) in enumerate(meta_info.items()):
            if item['type'] == 'target':
                continue
            elif item['type'] == 'categorical':
                imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                x[:, [i]] = imp_mean.fit_transform(x[:, [i]])
            else:
                imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
                x[:, [i]] = imp_mean.fit_transform(x[:, [i]])
        return x, y, "Regression", meta_info

def load_superconductivty(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/superconductivty/superconductivty.csv")
    meta_info = json.load(open(path + "uci_regression/superconductivty/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info

def load_wine_white(path="./", missing_strategy="drop"):
    
    # white wine dataset
    data = pd.read_csv(path + "uci_regression/wine_white/wine_white.csv", sep=";")
    meta_info = json.load(open(path + "uci_regression/wine_white/data_types.json"))
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
    return x, y, "Regression", meta_info

def load_wine_red(path="./", missing_strategy="drop"):
    
    # red wine dataset
    data = pd.read_csv(path + "uci_regression/wine_red/wine_red.csv", sep=";")
    meta_info = json.load(open(path + "uci_regression/wine_red/data_types.json"))
    x, y = data.iloc[:,:-1].values, data.iloc[:,[-1]].values
    return x, y, "Regression", meta_info

def load_yacht_hydrodynamics(path="./", missing_strategy="drop"):
    
    data = pd.read_csv(path + "uci_regression/yacht_hydrodynamics/yacht_hydrodynamics.csv", header=None, sep=",")
    meta_info = json.load(open(path + "uci_regression/yacht_hydrodynamics/data_types.json"))
    x, y = data.iloc[:, :-1].values, data.iloc[:, [-1]].values
    return x, y, "Regression", meta_info
