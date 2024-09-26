import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso


def set_indexes(dfs):
    for df in dfs:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

def get_best_model(X_train, X_test, y_train, y_test):
    exp_10 = np.logspace(-4,4,30)
    best_model = None
    best_acc = -np.inf
    for i in exp_10:
        lasso_reg = Lasso(alpha=i, max_iter=20000)
        lasso_reg.fit(X_train, y_train)
        test_acc = lasso_reg.score(X_test, y_test)
        nfeat = np.count_nonzero(lasso_reg.coef_)
        if test_acc > best_acc and nfeat <= 15: # limit to top 15 features
            best_model = lasso_reg
            best_acc = test_acc
    return best_model

def get_features_by_region(dfs_map, scaled=False):
    """
    Returns a map of region to features using lasso
    """
    features_by_region = {}
    coefs_by_region = {}
    for region, df in dfs_map.items():
        X = df.drop(columns=["critical_staffing_shortage_today_yes"])
        y = df[["critical_staffing_shortage_today_yes"]]
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
        if scaled:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        best_model = get_best_model(X_train, X_test, y_train, y_test)

        # filter out indices that Lasso has deemed 0
        indices = []
        coefs = {}
        for i, coef in enumerate(best_model.coef_):
            if abs(coef) > 0:
                indices.append(i)
                coefs[i] = coef
                
        features = []
        coefs_map = {}
        for index in indices:
            column_name = df.drop(columns=["critical_staffing_shortage_today_yes"]).columns[index]
            features.append(column_name)
            coefs_map[column_name] = coefs[index]
            
        # Only return statistically significant columns
        columns_exclude = list(set(X.columns) - set(features))
        X = df.drop(columns=["critical_staffing_shortage_today_yes", *columns_exclude])
        y = df["critical_staffing_shortage_today_yes"]

        statistically_significant_columns = []
        for column in X.columns:
            result = stats.pearsonr(X[column], y)
            if result.pvalue < 0.05:
                statistically_significant_columns.append(column)
            else:
                del coefs_map[column]

        features_by_region[region] = statistically_significant_columns
        coefs_by_region[region] = coefs_map
    return features_by_region, coefs_by_region

def create_rolling_df(df, shift, ignore_columns=[]):
    """
    Creates lagged columns by the number of shifts
    specified. For shift = 2, each feature column will
    have -2 appended to the column name and will be
    shifted by 2.
    """
    data = {}
    numerics = df.select_dtypes('number')
    for col in numerics.columns:
        if col in ignore_columns:
            continue
        new_df = pd.DataFrame(index=df.index)
        new_df[f"{col}-{shift}"] = df[col].shift(shift)
        data[f"{col}-{shift}"] = new_df
    new_df = pd.concat(data.values(), axis=1, ignore_index=True)
    new_df.columns = data.keys()
    new_df = pd.concat([df[ignore_columns], new_df], axis=1)
    return new_df.dropna()