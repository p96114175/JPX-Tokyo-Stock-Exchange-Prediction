import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
import jpx_tokyo_market_prediction
import warnings; warnings.filterwarnings("ignore")

# function
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

def get_features(df):
    df_feat = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat

def get_Xy_and_model(df_train):
    df_proc = get_features(df_train)
    df_proc['y'] = df_train['Target']
    df_proc = df_proc.dropna(how = "any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    try:
        model = lgb.LGDMRegressor(device_type = 'gpu')
        model.fit(X, y)
    except:
        model = lgb.LGBMRegressor()
        model.fit(X, y)
    return X, y, model

# main thread
stock_prices = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
X, y, model = get_Xy_and_model(stock_prices)
Xs, ys, model = X, y, model


env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (df_test, options, financials, trades, secondary_prices, df_pred) in iter_test:
    df_pred['row_id'] = (df_pred['Date'].astype(str) + '_' + df_pred['SecuritiesCode'].astype(str))
    df_test['row_id'] = (df_test['Date'].astype(str) + '_' + df_pred['SecuritiesCode'].astype(str))
    
    x_test = get_features(df_test)
    y_pred = model.predict(x_test)
    
    df_pred['Target'] = y_pred
    df_pred = df_pred.sort_values(by = "Target", ascending = False)
    df_pred['Rank'] = np.arange(0, 2000)
    df_pred = df_pred.sort_values(by = "SecuritiesCode", ascending = True)
    df_pred.drop(["Target"], axis = 1)
    submission = df_pred[["Date", "SecuritiesCode", "Rank"]]
    env.predict(submission)