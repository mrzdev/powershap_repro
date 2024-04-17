import pandas as pd
from xgboost import XGBRegressor
from powershap  import PowerShap

params = {
    'eval_metric': 'rmse',
    'max_delta_step': 10,
    'distribution': 'normal',
    'use_best_model': True,
    'seed': 2020,
    'n_estimators': 10000,
    'early_stopping_rounds': 5,
    'booster': 'gbtree',
    'verbosity': 0,
    'verbose': False,
    'max_depth': 9,
    'learning_rate': 0.004511090441650392,
    'alpha': 100.0,
    'gamma': 1.0,
    'lambda': 5.0,
    'colsample_bytree': 0.6722702706392691,
    'subsample': 0.8073968512821978
}

df = pd.read_parquet("df.parquet")
X = df.loc[:, df.columns != 'y']
y = df['y']
model = XGBRegressor(**params)
fsel = PowerShap(model = model, show_progress = False)
fsel.fit(X, y, shuffle=False)

# ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (110,) + inhomogeneous part.