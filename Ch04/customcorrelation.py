# Import the estimator and transformer class
from pyspark.ml import Transformer
# Parameter sharing class. We will use this for input column
from pyspark.ml.param.shared import HasInputCol
# Statistics class to calculate correlation
from pyspark.mllib.stat import Statistics
import pandas as pd
# custom class definition
class CustomCorrelation(Transformer, HasInputCol):
    """
    A custom function to calculate the correlation between two variables.
    
    Parameters:
    -----------
    inputCol: default value (None)
        Feature column name to be used for the correlation purpose. The input column should be assembled vector.
        
    correlation_type: 'pearson' or 'spearman'
    
    correlation_cutoff: float, default value (0.7), accepted values 0 to 1
        Columns more than the specified cutoff will be displayed in the output dataframe. 
    """
    
    # Initialize parameters for the function
    def __init__(self, inputCol=None, correlation_type='pearson', correlation_cutoff=0.7):
        
        super(CustomCorrelation, self).__init__()
        
        assert inputCol, "Please provide a assembled feature column name"
        #self.inputCol is class parameter
        self.inputCol = inputCol 
        
        assert correlation_type == 'pearson' or correlation_type == 'spearman', "Please provide a valid option for correlation type. 'pearson' or 'spearman'. "
        #self.correlation_type is class parameter
        self.correlation_type = correlation_type
        
        assert 0.0 <= correlation_cutoff <= 1.0, "Provide a valid value for cutoff. Accepted range is 0 to 1" 
        #self.correlation_cutoff is class parameter
        self.correlation_cutoff = correlation_cutoff
            
    # Transformer function, method inside a class, '_transform' - protected parameter
    def _transform(self, df):
        
        for k, v in df.schema[self.inputCol].metadata["ml_attr"]["attrs"].items():
            features_df = pd.DataFrame(v)
            
        column_names = list(features_df['name'])
        df_vector = df.rdd.map(lambda x: x[self.inputCol].toArray())
        
        #self.correlation_type is class parameter
        matrix = Statistics.corr(df_vector, method=self.correlation_type)
        
        # apply pandas dataframe operation on the fit output
        corr_df = pd.DataFrame(matrix, columns=column_names, index=column_names)
        final_corr_df = pd.DataFrame(corr_df.abs().unstack().sort_values(kind='quicksort')).reset_index()
        final_corr_df.rename({'level_0': 'col1', 'level_1': 'col2', 0: 'correlation_value'}, axis=1, inplace=True)
        final_corr_df = final_corr_df[final_corr_df['col1'] != final_corr_df['col2']]
        
        #shortlisted dataframe based on custom cutoff
        shortlisted_corr_df = final_corr_df[final_corr_df['correlation_value'] > self.correlation_cutoff]
        return corr_df, shortlisted_corr_df
