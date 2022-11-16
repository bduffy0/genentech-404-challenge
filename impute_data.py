import traceback
from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn
from hyperimpute.plugins.imputers.plugin_hyperimpute import HyperImputePlugin
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MinMaxScaler

import gpboost as gpb

neuro_features = ['Hippocampus', 'Entorhinal', 'Fusiform', 'MidTemp']
vent_wb_features = ['Ventricles', 'WholeBrain']
cog_features = ['CDRSB', 'MMSE', 'ADAS13']
age_dx_features = ['AGE', 'DX_num']
ID_VISCODE_COLS = ['RID_HASH', 'VISCODE']


def load_data(parent_path):
    PATH = Path(f'{parent_path}/data/')
    df_gt = pd.read_csv(PATH / 'dev_set.csv').sort_values(['RID_HASH', 'VISCODE'], ignore_index=True)
    df_dev_1 = pd.read_csv(PATH / 'dev_1.csv').sort_values(['RID_HASH', 'VISCODE'], ignore_index=True)
    df_dev_2 = pd.read_csv(PATH / 'dev_2.csv').sort_values(['RID_HASH', 'VISCODE'], ignore_index=True)
    df_dev_3 = pd.read_csv(PATH / 'dev_3.csv').sort_values(['RID_HASH', 'VISCODE'], ignore_index=True)
    df_test_a = pd.read_csv(PATH / 'test_A.csv').sort_values(['RID_HASH', 'VISCODE'], ignore_index=True)
    df_test_b = pd.read_csv(PATH / 'test_B.csv').sort_values(['RID_HASH', 'VISCODE'], ignore_index=True)
    sample = pd.read_csv(PATH / 'sample_submission.csv', index_col=0)
    return df_gt, df_dev_1, df_dev_2, df_dev_3, df_test_a, df_test_b, sample


def withinsubjectknn(input_df, n_neigh=2, ignore_cols=None, drop_cols=ID_VISCODE_COLS):
    """ KNN imputation for running within subject """

    if ignore_cols is None:
        ignore_cols = []

    ignore_col_data = {}
    for col in ignore_cols:
        ignore_col_data[col] = input_df[col]

    im = sklearn.impute.KNNImputer(n_neighbors=n_neigh, weights='distance')

    out_df = input_df.drop(drop_cols, axis=1)
    input_df_id_dropped = input_df.drop(drop_cols, axis=1)

    not_nan_cols = ~(input_df_id_dropped.isna().sum() == input_df.shape[0])
    cols = not_nan_cols[not_nan_cols].index.values.tolist()

    # scale
    df_dropped_na = input_df_id_dropped[cols]
    ss = sklearn.preprocessing.StandardScaler()

    X_scaled = ss.fit_transform(df_dropped_na)
    df_scaled = pd.DataFrame(X_scaled, index=df_dropped_na.index, columns=df_dropped_na.columns)

    y = im.fit_transform(df_scaled[cols])
    out_trans = ss.inverse_transform(y)
    out_df.loc[:, cols] = out_trans
    out_df[ID_VISCODE_COLS] = input_df[ID_VISCODE_COLS]

    for col in ignore_col_data:
        out_df[col] = ignore_col_data[col]

    return out_df


def scale_cols(df, ignore_cols=False):
    """scale columns using min max scaler"""

    if ignore_cols:
        ignore_data = df[ignore_cols]
        df = df.drop(ignore_cols, 1)

    scaler = MinMaxScaler()
    scaler.fit(df)

    data = scaler.transform(df)
    df_out = pd.DataFrame(data, columns=df.columns)

    if ignore_cols:
        df_out[ignore_cols] = ignore_data

    return scaler, df_out


def fill_missing_subj_mean(df, col_names=('APOE4', 'PTGENDER_num', 'PTEDUCAT')):
    df_fill = df.copy()
    for col_name in col_names:
        new_col = df_fill[col_name].fillna(df_fill.groupby('RID_HASH')[col_name].transform('mean'))
        df_fill[col_name] = new_col
    return df_fill


def fill_missing_age(df):
    def _fill_age_subject(subjdf):
        baseline = subjdf[~subjdf['AGE'].isna()]
        mths = subjdf['VISCODE'] / 12

        if not baseline.empty:
            bl_age = baseline['AGE'].iloc[0]

            bl_mths = baseline['VISCODE'].iloc[0] / 12
            new_age = bl_age - bl_mths + mths
            subjdf['AGE'] = new_age
        return subjdf

    df_fill = df.copy()
    new_df = df_fill.groupby('RID_HASH').apply(_fill_age_subject)

    return new_df


def hash_ID(df, n_features):
    h = FeatureHasher(n_features=n_features, input_type='string')
    df_hash = pd.DataFrame((h.fit_transform(df['RID_HASH'])).toarray())
    data_np = np.concatenate([df, df_hash], axis=1)
    out_df = pd.DataFrame(data_np)
    out_df.columns = df.columns.values.tolist() + list(range(n_features))
    return h, out_df


def fill_within_subject_constant_columns(input_df):
    """ fill constant columns"""
    input_df = fill_missing_age(input_df)
    subj_const_cols = ['APOE4', 'PTGENDER_num', 'PTEDUCAT']
    input_df = fill_missing_subj_mean(input_df, col_names=subj_const_cols)
    return input_df


# to standardize output results from different mixed model APIs
class Result:
    random_effects = None


class GPBEstimator:
    """ Grouped random effects estimator """

    def __init__(self, groups=None, params=None):

        self.scale = True
        self.result = Result()
        self.groups = groups
        self.params = params
        self.init_model(groups)

    def init_model(self, groups):
        self.model = gpb.GPModel(group_data=groups, likelihood='gaussian')

    def fit(self, X, y, groups=None):
        self.scaler = sklearn.preprocessing.StandardScaler()

        if self.scale:
            X = self.scaler.fit_transform(X)

        if groups is not None:
            self.init_model(groups)

        # self.model = gpb.GPModel(group_data=groups, likelihood='gaussian')
        self.model.fit(X=X, y=y, params=self.params)

        self.result.random_effects = set([x[0] for x in self.model.group_data.tolist()])

    def predict(self, X, groups=None):
        groups = pd.Series(groups)

        if self.scale:
            X = self.scaler.transform(X)

        out = self.model.predict(X_pred=X, group_data_pred=groups)['mu']

        return out


def fit_gpb(input_df, feature, X_features, params=None):
    """drop rows that we cannot fit and """

    fitting_df = input_df.dropna(subset=[feature] + X_features)
    groups = fitting_df['RID_HASH']
    model = GPBEstimator(groups=groups, params=params)
    X = pd.DataFrame(fitting_df[X_features])
    y = fitting_df[feature].values
    model.fit(X.values, y, groups=groups)
    return model


def predict_feature_mixed_effects_row_gbp_helper(in_df, feature, model, x_features):
    missing = in_df[feature].isna()
    ID = in_df['RID_HASH'].iloc[0]

    if missing.sum() > 0:
        missing_df = in_df[missing]
        preds = pd.DataFrame(model.predict(pd.DataFrame(missing_df[x_features]), groups=[ID] * len(missing_df)))
        preds.index = in_df.index
        in_df.loc[:, feature] = preds
    return in_df


def predict_row(in_row, model, feature, x_features):
    """ predict single row only"""

    hash_in_model = in_row['RID_HASH'].iloc[0] in model.result.random_effects
    feature_missing = in_row[feature].isna().iloc[0]

    # check if there are missing vals and if that subject has fitted random effects
    if in_row.isna().sum().sum() > 0 and hash_in_model and feature_missing:
        newrow = predict_feature_mixed_effects_row_gbp_helper(in_row, feature=feature, model=model,
                                                              x_features=x_features)
    else:
        newrow = in_row

    return newrow


def gbp_pipeline(input_df, features, feature_combs, params=None):
    for x_features in feature_combs:
        for feature_num, feature in enumerate(features):
            model = fit_gpb(input_df, feature=feature, X_features=x_features, params=params)

            for i, row in input_df.iterrows():
                in_row = pd.DataFrame(row).T
                hash_in_model = in_row['RID_HASH'].iloc[0] in model.result.random_effects
                feature_missing = in_row[feature].isna().iloc[0]
                if in_row.isna().sum().sum() > 0 and hash_in_model and feature_missing:
                    res = predict_row(in_row, model, feature, x_features)
                    input_df.loc[i, :] = res.iloc[0]

    return input_df


def subject_level_pipeline(input_df):
    """1. fill cols that are constant across subject
       2. fit mixed model
       3. run KNN within subject """

    # 1. Fill constant columns
    input_df = fill_within_subject_constant_columns(input_df)

    # 2. Prepare and fit mixed model
    mixed_model = partial(gbp_pipeline, params={"std_dev": True,
                                                "optimizer_cov": "gradient_descent", "lr_cov": 0.1,
                                                "use_nesterov_acc": True,
                                                "maxit": 100})

    # fit different features based on feature combinations
    input_df = mixed_model(input_df, features=neuro_features,
                           feature_combs=[['WholeBrain'] + age_dx_features,
                                          age_dx_features, ['AGE']])

    input_df = mixed_model(input_df, features=vent_wb_features,
                           feature_combs=[neuro_features + age_dx_features,
                                          age_dx_features, ['AGE']])

    input_df = mixed_model(input_df, features=cog_features,
                           feature_combs=[neuro_features + age_dx_features,
                                          vent_wb_features + age_dx_features,
                                          ['WholeBrain'] + age_dx_features])

    # run KNN within subject first to all cols except DX_num and then to DX_num with k=1
    input_df = input_df.groupby('RID_HASH').apply(
        partial(withinsubjectknn, n_neigh=2, ignore_cols=['DX_num'], drop_cols=['RID_HASH']))

    out_df = input_df.groupby('RID_HASH').apply(
        partial(withinsubjectknn, n_neigh=1, ignore_cols=set(input_df.columns) - set(['DX_num']),
                drop_cols=['RID_HASH']))

    return out_df


def save_test_results(out_df, aindex, bindex, outpath='submissions/sub_42.csv'):
    pred_a = out_df.loc[aindex]
    pred_b = out_df.loc[bindex]

    stacked_a = pred_a.stack()
    stacked_a.index = ['_'.join(x) + f'_test_A' for x in stacked_a.index]

    stacked_b = pred_b.stack()
    stacked_b.index = ['_'.join(x) + f'_test_B' for x in stacked_b.index]

    df_pred = pd.concat([stacked_a, stacked_b], axis=0)

    assert out_df.isna().sum().sum() == 0

    df_out = df_pred.loc[sample.index]
    df_out.name = 'Predicted'

    df_out.to_csv(outpath)


def prep_test_data(df_gt, df_test_a, df_test_b):
    aindex = df_test_a[ID_VISCODE_COLS].astype('str').agg('_'.join, axis=1)
    bindex = df_test_b[ID_VISCODE_COLS].astype('str').agg('_'.join, axis=1)

    df_test_a = df_test_a.set_index(aindex)
    df_test_b = df_test_b.set_index(bindex)

    test_concat = pd.concat([df_test_a, df_test_b], axis=0)

    gt_df_pred_concat = pd.concat([df_gt, test_concat])

    return gt_df_pred_concat, aindex, bindex


def run_inference(df_gt, df_test_a, df_test_b, params):
    # prepare data - concat
    gt_df_pred_concat, aindex, bindex = prep_test_data(df_gt, df_test_a, df_test_b)

    # run subject level pipeline
    input_df = subject_level_pipeline(gt_df_pred_concat)
    ncols = input_df.shape[1]

    # hash subject IDs
    init_index = input_df.index
    h, input_df = hash_ID(input_df, n_features=params['n_features'])
    input_df.index = init_index
    data_noid = input_df.drop(ID_VISCODE_COLS, axis=1)
    id_data = input_df[ID_VISCODE_COLS]
    num_subj_cols = input_df.shape[1] - ncols
    subj_cols = list(range(num_subj_cols))
    ignore_cols = subj_cols

    # min-max scale cols
    scaler, data_scaled = scale_cols(data_noid, ignore_cols=ignore_cols)

    # run hyperimpute to fill missing that are still missing
    plugin = HyperImputePlugin(regression_seed=["xgboost_regressor", "catboost_regressor", "random_forest_regressor",
                                                'linear_regression'], random_state=130)
    out = plugin.fit_transform(data_scaled.copy())
    out = pd.DataFrame(out, columns=data_noid.columns)

    # invert scaling
    df_dr_out = out.drop(subj_cols, axis=1)
    out = scaler.inverse_transform(df_dr_out)
    out = pd.DataFrame(out, columns=df_dr_out.columns, index=data_noid.index)
    out[ID_VISCODE_COLS] = id_data

    # save output
    save_test_results(out, aindex, bindex, params['out_path'])


if __name__ == '__main__':
    df_gt, df_dev_1, df_dev_2, df_dev_3, df_test_a, df_test_b, sample = load_data('.')
    params = {'n_features': 64, 'out_path': './submission.csv'}

    run_inference(df_gt, df_test_a, df_test_b, params)
