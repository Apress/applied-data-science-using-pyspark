"""
    validation_and_plots.py

    This code is used to perform
    1. Model validation
    2. Generate ROC chart
    3. Generate KS Chart
    4. Confusion matrix
"""



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import glob
import os
import pandas as pd
import seaborn as sns
from pandas import ExcelWriter
from metrics_calculator import *

# Generate ROC chart

def draw_roc_plot(user_id, mdl_ltrl, y_score, y_true, model_type, data_type):

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label = 1)
    roc_auc = metrics.auc(fpr,tpr)
    plt.title(str(model_type) + ' Model - ROC for ' + str(data_type) + ' data' )
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc = 'lower right')
    print('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type) + '/' + str(model_type) + ' Model - ROC for ' + str(data_type) + ' data.png')
    plt.savefig('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type) + '/' + str(model_type) + ' Model - ROC for ' + str(data_type) + ' data.png', bbox_inches='tight')
    plt.close()

# Generate KS Chart

def draw_ks_plot(user_id, mdl_ltrl, model_type):

    writer = ExcelWriter('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type) + '/KS_Charts.xlsx')

    for filename in glob.glob('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type) + '/KS ' + str(model_type) + ' Model*.xlsx'):
        excel_file = pd.ExcelFile(filename)
        (_, f_name) = os.path.split(filename)
        (f_short_name, _) = os.path.splitext(f_name)
        for sheet_name in excel_file.sheet_names:
            df_excel = pd.read_excel(filename, sheet_name=sheet_name)
            df_excel = df_excel.style.apply(highlight_max, subset=['spread'], color='#e6b71e')
            df_excel.to_excel(writer, f_short_name, index=False)
            worksheet = writer.sheets[f_short_name]
            worksheet.conditional_format('C2:C11', {'type': 'data_bar','bar_color': '#34b5d9'})#,'bar_solid': True
            worksheet.conditional_format('E2:E11', {'type': 'data_bar','bar_color': '#366fff'})#,'bar_solid': True
        os.remove(filename)
    writer.save()

# Confusion matrix

def draw_confusion_matrix(user_id, mdl_ltrl, y_pred, y_true, model_type, data_type):

    AccuracyValue =  metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
    PrecisionValue = metrics.precision_score(y_pred=y_pred, y_true=y_true)
    RecallValue = metrics.recall_score(y_pred=y_pred, y_true=y_true)
    F1Value = metrics.f1_score(y_pred=y_pred, y_true=y_true)

    plt.title(str(model_type) + ' Model - Confusion Matrix for ' + str(data_type) + ' data \n \n Accuracy:{0:.3f}   Precision:{1:.3f}   Recall:{2:.3f}   F1 Score:{3:.3f}\n'.format(AccuracyValue, PrecisionValue, RecallValue, F1Value))
    cm = metrics.confusion_matrix(y_true=y_true,y_pred=y_pred)
    sns.heatmap(cm, annot=True, fmt='g'); #annot=True to annotate cells
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    print('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type) + '/' + str(model_type) + ' Model - Confusion Matrix for ' + str(data_type) + ' data.png')
    plt.savefig('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type) + '/' + str(model_type) + ' Model - Confusion Matrix for ' + str(data_type) + ' data.png', bbox_inches='tight')
    plt.close()

# Model validation

def model_validation(user_id, mdl_ltrl, data, y, model, model_type, data_type):

    start_time = time.time()

    pred_data = model.transform(data)
    print('model output predicted')

    roc_data, accuracy_data, ks_data, y_score, y_pred, y_true, decile_table = calculate_metrics(pred_data,y,data_type)
    draw_roc_plot(user_id, mdl_ltrl, y_score, y_true, model_type, data_type)
    decile_table.to_excel('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + str(model_type) + '/KS ' + str(model_type) + ' Model ' + str(data_type) + '.xlsx',index=False)
    draw_confusion_matrix(user_id, mdl_ltrl, y_pred, y_true, model_type, data_type)
    print('Metrics computed')

    l = [roc_data, accuracy_data, ks_data]
    end_time = time.time()
    print("Model validation process completed in :  %s seconds" % (end_time-start_time))
    return l
