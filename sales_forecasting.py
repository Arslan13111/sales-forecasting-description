# walmart_sales_forecasting_full_fixed.py
"""
Fixed Walmart Sales Forecasting pipeline (XGBoost + LightGBM) compatible with newer library versions.

Usage (example):
python walmart_sales_forecasting_full_fixed.py --train train.csv --test test.csv --stores stores.csv --features features.csv --store 1 --dept 1

Outputs saved into --output (default 'results/').
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
import lightgbm as lgb
import joblib
import time

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_and_merge(train_path, test_path=None, stores_path=None, features_path=None):
    train = pd.read_csv(train_path, parse_dates=['Date'])
    test = pd.read_csv(test_path, parse_dates=['Date']) if test_path and Path(test_path).exists() else None
    stores = pd.read_csv(stores_path) if stores_path and Path(stores_path).exists() else None
    features = pd.read_csv(features_path, parse_dates=['Date']) if features_path and Path(features_path).exists() else None

    if stores is not None:
        train = train.merge(stores, on='Store', how='left')
        if test is not None:
            test = test.merge(stores, on='Store', how='left')

    if features is not None:
        train = train.merge(features, on=['Store','Date'], how='left')
        if test is not None:
            test = test.merge(features, on=['Store','Date'], how='left')

    return train, test, stores, features

def create_time_features(df):
    df = df.sort_values('Date').reset_index(drop=True)
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

def create_lags_rolls_group(df, target_col='Weekly_Sales', lags=(1,2,3,4), rolls=(3,7)):
    group_cols = [c for c in ('Store','Dept') if c in df.columns]
    if group_cols:
        df = df.sort_values(group_cols + ['Date']).reset_index(drop=True)
        def _create(g):
            for lag in lags:
                g[f'lag_{lag}'] = g[target_col].shift(lag)
            for w in rolls:
                g[f'roll_mean_{w}'] = g[target_col].shift(1).rolling(window=w).mean()
            g['pct_change_1'] = g[target_col].pct_change(periods=1)
            return g
        df = df.groupby(group_cols, group_keys=False).apply(_create).reset_index(drop=True)
    else:
        df = df.sort_values('Date').reset_index(drop=True)
        for lag in lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        for w in rolls:
            df[f'roll_mean_{w}'] = df[target_col].shift(1).rolling(window=w).mean()
        df['pct_change_1'] = df[target_col].pct_change(periods=1)
    # conservative fill
    df.fillna(0, inplace=True)
    return df

def seasonal_decompose_plot(series, period, out_file):
    try:
        res = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
        fig = res.plot()
        fig.set_size_inches(10,8)
        plt.tight_layout()
        fig.savefig(out_file, dpi=200)
        plt.close(fig)
        return True
    except Exception:
        return False

def time_series_cv_train(X, y, n_splits=5, xgb_params=None, lgb_params=None):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    xgb_oof = np.zeros(len(y))
    lgb_oof = np.zeros(len(y))
    fold_metrics = []
    xgb_models = []
    lgb_models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # XGBoost: fit normally (avoid passing eval_metric/early stopping in fit to keep compatibility)
        xgb = XGBRegressor(**(xgb_params or {}))
        try:
            xgb.fit(X_tr, y_tr)  # simple fit
        except TypeError:
            # fallback: some versions expect different args — try without extras
            xgb = XGBRegressor()
            xgb.fit(X_tr, y_tr)
        p_xgb = xgb.predict(X_val)
        xgb_oof[val_idx] = p_xgb
        xgb_models.append(xgb)

        # LightGBM: attempt early stopping; if fails, fallback to normal fit
        lgbm = lgb.LGBMRegressor(**(lgb_params or {}))
        try:
            lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', early_stopping_rounds=50, verbose=False)
        except Exception:
            lgbm = lgb.LGBMRegressor(**(lgb_params or {}))
            lgbm.fit(X_tr, y_tr)
        p_lgb = lgbm.predict(X_val)
        lgb_oof[val_idx] = p_lgb
        lgb_models.append(lgbm)

        fold_rmse_x = rmse(y_val, p_xgb)
        fold_rmse_l = rmse(y_val, p_lgb)
        fold_mae_x = mean_absolute_error(y_val, p_xgb)
        fold_mae_l = mean_absolute_error(y_val, p_lgb)
        fold_metrics.append({
            'fold': fold,
            'xgb_rmse': fold_rmse_x,
            'lgb_rmse': fold_rmse_l,
            'xgb_mae': fold_mae_x,
            'lgb_mae': fold_mae_l,
        })
        print(f'Fold {fold} | XGB RMSE: {fold_rmse_x:.4f} | LGB RMSE: {fold_rmse_l:.4f}')

    overall = {
        'xgb_rmse_oof': rmse(y, xgb_oof),
        'lgb_rmse_oof': rmse(y, lgb_oof),
        'xgb_mae_oof': mean_absolute_error(y, xgb_oof),
        'lgb_mae_oof': mean_absolute_error(y, lgb_oof)
    }
    print('Overall OOF:', overall)
    return xgb_models[-1], lgb_models[-1], xgb_oof, lgb_oof, fold_metrics, overall

def plot_actual_vs_pred(dates, y_true, y_pred, out_file, title='Actual vs Predicted'):
    plt.figure(figsize=(12,5))
    plt.plot(dates, y_true, label='Actual', linewidth=1)
    plt.plot(dates, y_pred, label='Predicted', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()

def feature_importance_plot(model, feat_names, out_file, top_n=30):
    try:
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]
        names = np.array(feat_names)[idx]
        vals = imp[idx]
        plt.figure(figsize=(8, max(4, len(names)*0.25)))
        plt.barh(names[::-1], vals[::-1])
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig(out_file, dpi=200)
        plt.close()
    except Exception as e:
        print('Feature importance plot failed:', e)

def model_one_series(df, store=None, dept=None, out_dir=Path('results'), n_splits=5, holdout_ratio=0.10):
    sel = df.copy()
    if store is not None:
        sel = sel[sel['Store']==store]
    if dept is not None:
        sel = sel[sel['Dept']==dept]
    sel = sel.sort_values('Date').reset_index(drop=True)
    if sel.shape[0] < 10:
        print('Not enough rows for modelling, skipping.')
        return None

    sel = create_time_features(sel)
    sel = create_lags_rolls_group(sel, target_col='Weekly_Sales')
    num_cols = sel.select_dtypes(include=[np.number]).columns.tolist()
    sel[num_cols] = sel[num_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # seasonal plot
    try:
        spath = out_dir / f'seasonal_decompose_store{store}_dept{dept}.png'
        if seasonal_decompose_plot(sel['Weekly_Sales'].rolling(window=4).mean().dropna(), period=52, out_file=spath):
            print('Saved seasonal decomposition to', spath)
    except Exception:
        pass

    exclude = ['Date','Weekly_Sales']
    if 'Store' in sel.columns: exclude.append('Store')
    if 'Dept' in sel.columns: exclude.append('Dept')
    feature_cols = [c for c in sel.columns if c not in exclude]

    # handle categorical 'Type' if present
    if 'Type' in sel.columns:
        sel['Type'] = sel['Type'].astype('category').cat.codes

    X = sel[feature_cols]
    y = sel['Weekly_Sales']
    dates = sel['Date']

    print('Using features:', feature_cols)

    xgb_params = {'n_estimators':500, 'learning_rate':0.05, 'max_depth':6, 'subsample':0.8, 'colsample_bytree':0.8, 'random_state':42, 'n_jobs':-1}
    lgb_params = {'n_estimators':1000, 'learning_rate':0.03, 'max_depth':8, 'subsample':0.8, 'colsample_bytree':0.8, 'random_state':42, 'n_jobs':-1}

    xgb_model, lgb_model, xgb_oof, lgb_oof, fold_metrics, overall = time_series_cv_train(X, y, n_splits=n_splits, xgb_params=xgb_params, lgb_params=lgb_params)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_base = out_dir / f'model_store{store}_dept{dept}'
    joblib.dump(xgb_model, str(model_base)+'_xgb.joblib')
    joblib.dump(lgb_model, str(model_base)+'_lgb.joblib')
    print('Saved models to', out_dir)

    holdout_size = max(1, int(len(sel) * holdout_ratio))
    holdout_df = sel.iloc[-holdout_size:].reset_index(drop=True)
    X_hold = holdout_df[feature_cols]
    y_hold = holdout_df['Weekly_Sales']
    dates_hold = holdout_df['Date']

    pred_xgb = xgb_model.predict(X_hold)
    pred_lgb = lgb_model.predict(X_hold)
    pred_avg = 0.5*(pred_xgb + pred_lgb)

    print('Holdout RMSE XGB:', rmse(y_hold, pred_xgb), 'LGB:', rmse(y_hold, pred_lgb), 'AVG:', rmse(y_hold, pred_avg))

    plot_actual_vs_pred(dates_hold, y_hold, pred_xgb, out_dir / f'actual_vs_pred_store{store}_dept{dept}_xgb.png', title=f'Store{store}_Dept{dept} XGBoost')
    plot_actual_vs_pred(dates_hold, y_hold, pred_lgb, out_dir / f'actual_vs_pred_store{store}_dept{dept}_lgb.png', title=f'Store{store}_Dept{dept} LightGBM')
    plot_actual_vs_pred(dates_hold, y_hold, pred_avg, out_dir / f'actual_vs_pred_store{store}_dept{dept}_avg.png', title=f'Store{store}_Dept{dept} Ensemble Avg')

    feature_importance_plot(xgb_model, feature_cols, out_dir / f'feature_importance_store{store}_dept{dept}_xgb.png')
    feature_importance_plot(lgb_model, feature_cols, out_dir / f'feature_importance_store{store}_dept{dept}_lgb.png')

    holdout_out = holdout_df[['Date']].copy()
    holdout_out['actual'] = y_hold
    holdout_out['pred_xgb'] = pred_xgb
    holdout_out['pred_lgb'] = pred_lgb
    holdout_out['pred_avg'] = pred_avg
    holdout_out.to_csv(out_dir / f'holdout_predictions_store{store}_dept{dept}.csv', index=False)

    oof_out = sel[['Date']].copy()
    oof_out['oof_xgb'] = xgb_oof
    oof_out['oof_lgb'] = lgb_oof
    oof_out.to_csv(out_dir / f'oof_predictions_store{store}_dept{dept}.csv', index=False)

    pd.DataFrame(fold_metrics).to_csv(out_dir / f'cv_fold_metrics_store{store}_dept{dept}.csv', index=False)

    return {
        'store': store, 'dept': dept,
        'holdout_rmse_xgb': float(rmse(y_hold, pred_xgb)),
        'holdout_rmse_lgb': float(rmse(y_hold, pred_lgb)),
        'holdout_rmse_avg': float(rmse(y_hold, pred_avg))
    }

def main(args):
    train_path = Path(args.train)
    test_path = Path(args.test) if args.test else None
    stores_path = Path(args.stores) if args.stores else None
    features_path = Path(args.features) if args.features else None
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, test, stores, features = load_and_merge(train_path, test_path, stores_path, features_path)
    print('Loaded train shape:', train.shape)

    results = []
    if args.all:
        pairs = train[['Store','Dept']].drop_duplicates().sort_values(['Store','Dept'])
        start = time.time()
        for _, row in pairs.iterrows():
            s = int(row['Store']); d = int(row['Dept'])
            print(f'\n--- Modeling Store {s} Dept {d} ---')
            try:
                res = model_one_series(train, store=s, dept=d, out_dir=out_dir, n_splits=args.n_splits, holdout_ratio=args.holdout_ratio)
                if res:
                    results.append(res)
            except Exception as e:
                print('Failed for', s, d, e)
        pd.DataFrame(results).to_csv(out_dir / 'all_pairs_results.csv', index=False)
        print('Completed all pairs in', round(time.time()-start,1), 'seconds')
    else:
        if args.store is None or args.dept is None:
            first = train[['Store','Dept']].drop_duplicates().iloc[0]
            args.store = int(first['Store']); args.dept = int(first['Dept'])
            print('No store/dept provided; using first pair:', args.store, args.dept)
        res = model_one_series(train, store=args.store, dept=args.dept, out_dir=out_dir, n_splits=args.n_splits, holdout_ratio=args.holdout_ratio)
        if res:
            pd.DataFrame([res]).to_csv(out_dir / f'result_store{args.store}_dept{args.dept}.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Walmart Sales Forecasting pipeline (fixed)')
    parser.add_argument('--train', required=True, help='train.csv with Weekly_Sales')
    parser.add_argument('--test', required=False, help='test.csv (optional)')
    parser.add_argument('--stores', required=False, help='stores.csv (optional)')
    parser.add_argument('--features', required=False, help='features.csv (optional)')
    parser.add_argument('--output', default='results', help='output directory')
    parser.add_argument('--n_splits', type=int, default=5, help='TimeSeriesSplit folds')
    parser.add_argument('--holdout_ratio', type=float, default=0.10, help='fraction of last data for holdout evaluation')
    parser.add_argument('--store', type=int, default=None, help='store id to model (single)')
    parser.add_argument('--dept', type=int, default=None, help='dept id to model (single)')
    parser.add_argument('--all', action='store_true', help='model all store-dept pairs (slower)')
    args = parser.parse_args()
    main(args)
