"""
    feature_selection.py

    The code here is used to perform final feature selection. The code is executed depending on the following criteria

    Criteria 1 (Number of variables less than or equal to 30) - The code is invoked to produce feature importance plot.
    Criteria 2 (Number of variables between 30 and 300) - The code is invoked and it uses Random Forest output to pick the best variables. Reduces the final variable list to 30.
    Criteria 3 (Number of variables more than 300) - MLA variable reduction is invoked and the output of it is used for the final variables.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The module below is used to draw the feature importance plot
def draw_feature_importance(user_id, mdl_ltrl, importance_df):

    importance_df = importance_df.sort_values('Importance_Score')
    plt.figure(figsize=(15,15))
    plt.title('Feature Importances')
    plt.barh(range(len(importance_df['Importance_Score'])), importance_df['Importance_Score'], align='center')
    plt.yticks(range(len(importance_df['Importance_Score'])), importance_df['name'])
    plt.ylabel('Variable Importance')
    plt.savefig('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + 'Features selected for modeling.png', bbox_inches='tight')
    plt.close()
    return None

# The module below is used to save the feature importance as a excel file
def save_feature_importance(user_id, mdl_ltrl, importance_df):
    importance_df.drop('idx',axis=1,inplace=True)
    importance_df = importance_df[0:30]
    importance_df.to_excel('/home/' + user_id + '/' + 'mla_' + mdl_ltrl + '/' + 'feature_importance.xlsx')
    draw_feature_importance(user_id, mdl_ltrl, importance_df)
    return None

# The module below is used to calculate the feature importance for each variables based on the Random Forest output. The feature importance is used to reduce the final variable list to 30.
def ExtractFeatureImp(featureImp, dataset, featuresCol):
    """
    Takes in a feature importance from a random forest / GBT model and map it to the column names
    Output as a pandas dataframe for easy reading
    rf = RandomForestClassifier(featuresCol="features")
    mod = rf.fit(train)
    ExtractFeatureImp(mod.featureImportances, train, "features")
    """

    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['Importance_Score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('Importance_Score', ascending = False))
