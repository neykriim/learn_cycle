import pandas as pd
import numpy as np

def parameterCounts(dataPath='statistic new.csv', classes = ["SA", "FA"]):
    '''
    Сколько каждого типа льда всего
    '''
    ice_types = pd.read_csv('ice_types.csv', index_col=0)[classes]
    dataset = pd.read_csv(dataPath, index_col=0)
    dataset.columns = pd.MultiIndex.from_tuples(zip(dataset.columns, [""] * len(dataset.columns)))

    class_values = []
    for c in classes:
        #class_values += [f"{c}={i}" for i in ice_types[c].unique()]
        class_values += [(c, i) for i in ice_types[c].unique()]
    for un in class_values:
        dataset[un] = 0

    for idx, row in dataset.iterrows():
        for i, un in enumerate(row[('uniques', '')].split(', ')):
            count = int(row[('counts', '')].split(', ')[i])
            un = int(un)

            ice_type = ice_types.iloc[un]
            #for column in [f"{ice_type.index[i]}={ice_type.values[i]}" for i in range(len(ice_type))]:
            for column in [(ice_type.index[i], ice_type.values[i]) for i in range(len(ice_type))]:
                dataset.loc[idx, column] += count

    dataset = dataset[['image']+classes]
    dataset["count"] = 1
    return dataset

def ignoreValues(dataset, classes = ["SA", "FA"], values=[-9, 99]):
    '''
    Убрать пропуски
    '''
    ignoreVariants = [(c, v) for c in classes for v in values]
    ignoreVariants = [i for i in ignoreVariants if i in dataset.columns]

    dataset = dataset[pd.concat([dataset[cond] == 0 for cond in ignoreVariants], axis=1).all(axis=1)]
    dataset = dataset.drop(columns=ignoreVariants)
    return dataset

def skipOneColor(dataset, classes = ["SA", "FA"], all=True):
    '''
    Убрать одноцветные изображения
    '''
    if all:
        # где все классы одноцветные
        dataset = dataset[(dataset[classes] == 1000000).sum(axis=1) < len(classes)]
    else:
        # где хотябы один класс одноцветный
        dataset = dataset[(dataset[classes] == 1000000).sum(axis=1) == 0]
    return dataset

def skipRare(dataset, classes = ["SA", "FA"], treshold=0.05):
    '''
    Убрать редкие значения
    '''
    d = dataset.loc[:, (classes, slice(None))].mul(dataset[('count', '')], axis=0).sum()
    ignoreVariants = d[d < treshold * dataset['count'].sum() * 1000000].index

    dataset = dataset[pd.concat([dataset[cond] == 0 for cond in ignoreVariants], axis=1).all(axis=1)]
    dataset = dataset.drop(columns=ignoreVariants)

    return dataset

def skipUnbalanced(dataset, classes = ["SA", "FA"], all=True, treshold=0.25):
    '''
    Убрать снимки с неравномерным распределением параметров
    '''
    if all:
        d = (dataset[classes] / 1000000).replace(0,np.nan)
        dataset = dataset[(d.max(axis=1) - d.min(axis=1)) <= treshold]
        #dataset = dataset[(dataset[classes] / 1000000).replace(0,np.nan).std(axis=1, skipna=True) <= treshold]
    else:
        for c in classes:
            d = (dataset[c] / 1000000).replace(0,np.nan)
            dataset = dataset[(d.max(axis=1) - d.min(axis=1)) <= treshold]
            #dataset = dataset[(dataset[classes] / 1000000)[c].replace(0,np.nan).std(axis=1, skipna=True) <= treshold]
    return dataset

def checkNaN(dataset, classes = ["SA", "FA"]):
    '''
    Проверить наличие не размеченных пикселей
    '''
    return dataset[dataset[classes].sum(axis=1) == len(classes) * 1000000]