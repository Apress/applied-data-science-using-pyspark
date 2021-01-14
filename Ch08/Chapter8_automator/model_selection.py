"""
    model_selection.py

    Use this module to pick the best model based on user selected stats. It creates a final output file - 'metrics.xlsx' in the model output folder which contains the stats for each model. In addition, it also tells you which model is Champion and Challenger.

"""

import pandas as pd
# from sklearn.externals import joblib
import joblib
import numpy as np
import glob
import os

def select_model(user_id, mdl_ltrl, model_selection_criteria, dataset_to_use):
    df = pd.DataFrame({},columns=['roc_train', 'accuracy_train', 'ks_train', 'roc_valid', 'accuracy_valid', 'ks_valid', 'roc_test', 'accuracy_test', 'ks_test', 'roc_oot1', 'accuracy_oot1', 'ks_oot1', 'roc_oot2', 'accuracy_oot2', 'ks_oot2'])
    current_dir = os.getcwd()
    os.chdir('/home/' + user_id + '/' + 'mla_' + mdl_ltrl)
    for file in glob.glob('*metrics.z'):
        l = joblib.load(file)
        df.loc[str(file.split('_')[0])] = l

    for file in glob.glob('*metrics.z'):
        os.remove(file)

    os.chdir(current_dir)
    df.index = df.index.set_names(['model_type'])
    df = df.reset_index()
    model_selection_criteria = model_selection_criteria.lower()
    column_to_sort = model_selection_criteria + '_' + dataset_to_use.lower()
    checker_value = 0.03

    if model_selection_criteria == 'ks':
        checker_value = checker_value * 100

    df['counter'] = (np.abs(df[column_to_sort] - df[model_selection_criteria + '_train']) > checker_value).astype(int) +                     (np.abs(df[column_to_sort] - df[model_selection_criteria + '_valid']) > checker_value).astype(int) +                     (np.abs(df[column_to_sort] - df[model_selection_criteria + '_test']) > checker_value).astype(int) +                     (np.abs(df[column_to_sort] - df[model_selection_criteria + '_oot1']) > checker_value).astype(int) +                     (np.abs(df[column_to_sort] - df[model_selection_criteria + '_oot2']) > checker_value).astype(int)


    df = df.sort_values(['counter', column_to_sort], ascending=[True, False]).reset_index(drop=True)

    df['selected_model'] = ''
    df.loc[0,'selected_model'] = 'Champion'
    df.loc[1,'selected_model'] = 'Challenger'

    df.to_excel('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/metrics.xlsx')
    return df
