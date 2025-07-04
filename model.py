import json
import warnings

import category_encoders as ce
import lightgbm
import numpy as np
import pandas as pd

class FNIBotModel:
    def __init__(
        self,
        model_type: str = None,
        model_params: dict = None,
        monotone_constraints: dict = None,
        eval_period: int = 100,
        target_column: str = None,
        feature_columns: list = None,
        categorical_columns: list = None
    ):
        
        # Log Info
        print(f'Initializing {model_type} model using LGBM {lightgbm.__version__}')
        assert len(feature_columns) == len(set(feature_columns)), "feature_columns contains duplicates"
        if monotone_constraints is not None:
            print(f'Using monotone constraints: {monotone_constraints}')
            monotone_list = [monotone_constraints.get(x, 0) for x in feature_columns]
            model_params['monotone_constraints'] = monotone_list
            
        print(f'Using model parameters: {model_params}')
        
        self.model_type = model_type
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.categorical_encoder = None
        self.required_columns = self.feature_columns + [self.target_column]
        self.eval_period = eval_period
        self.model_params = model_params
        self.dtypes_map = None
        
        if model_type == 'approval':
            self.model = lightgbm.LGBMClassifier(**model_params)
        elif model_type == 'rate':
            self.model = lightgbm.LGBMRegressor(**model_params)
        else:
            raise RuntimeError(f'Invalid model type: {model_type}')

    def fit(self, raw_train, raw_val=None):
        """
        Function to fit/train the model using the training data
        
        Paratmeters
        -----------
        raw_train: Data Frame like
            Input Data Frame containing the training data & target
        raw_val: Data Frame like (optional)
            Validation dataframe for early stopping

        Returns
        -------
        Object
            Reference to the instance object
        """
        
        fit_kwargs = {
            'callbacks': [lightgbm.log_evaluation(period=self.eval_period)]
        }
        
        x_train, y_train = self._training_preprocess(raw_train)

        if raw_val is not None:
            x_val = self._inference_preprocess(raw_val)
            y_val = self.get_target_data(raw_val)
            fit_kwargs['eval_set'] = [(x_val, y_val)]
        else:
            fit_kwargs['eval_set'] = [(x_train, y_train)]

        if 'monotone_constraints' in self.model_params:
            assert len(self.model_params['monotone_constraints'])==len(x_train.columns.tolist()), "Monotone Constraints do not match"
        
        self.model.fit(x_train, y_train, **fit_kwargs)

        return self

    def predict(self, raw_df):
        """
        Function to make predictions using the trained model
        
        Paratmeters
        -----------
        raw_df: Data Frame like
            Input Data Frame for which predictions need to be made
            
        Returns
        -------
        array
            Predictions for the given input data
        """
        
        x_data = self._inference_preprocess(raw_df)
        if self.model_type == 'rate':
            return self.model.predict(x_data)
        else:
            return self.model.predict_proba(x_data)[:, 1]
    
    def get_target_data(self, raw_df: pd.DataFrame):
        """
        Function to get target data from a dataframe
        
        Paratmeters
        -----------
        raw_df: Data Frame like
            Raw Data Frame which needs to be preprocessed
            
        Returns
        -------
        Series
            Target data series
            
        """
     
        if self.model_type == 'rate':
            return raw_df.loc[:, self.target_column].copy()
        elif self.model_type == 'approval':
            return raw_df.loc[:, self.target_column] == 'Approved'
        else:
            return None
        
    
    def _training_preprocess(self, raw_df: pd.DataFrame):
        """
        Function to execute encoding and preprocessing related to the rate estimation process for both the input and target data
        
        Paratmeters
        -----------
        raw_df: Data Frame like
            Raw Data Frame which needs to be preprocessed
            
        Returns
        -------
        Data Frame
            Pre processed data frame for input data that will be fed as the input to the model
        array
            Pre processed array for output data that will be fed as the input to the model
            
        """
        raw_df = raw_df.copy()
        
        x_data = self._x_preprocess(raw_df)
        y_data = self.get_target_data(raw_df)
        
        # Store data original data types
        self.dtypes_map = x_data.dtypes.apply(lambda x: x.name).to_dict()

        print('generating model encoders')
        self.categorical_encoder = ce.TargetEncoder(cols=self.categorical_columns)
        x_data = self.categorical_encoder.fit_transform(x_data, y_data)
                
        dtype_str = ', '.join([f"{col}: {dtype}" for col, dtype in x_data.dtypes.to_dict().items()])
        print(f'Dtypes after training pre-processing: {dtype_str}')
        
        return x_data, y_data
    
    def _inference_preprocess(self, raw_df: pd.DataFrame):
        """
        Function to execute encoding and preprocessing for test/inference data
        
        Paratmeters
        -----------
        raw_df: Data Frame like
            Raw Data Frame (test/inference data) which needs to be preprocessed
            
        Returns
        -------
        Data Frame
            Pre processed data frame for input data that will be fed as the input to the model
            
        """
        raw_df = raw_df.copy()
        
        x_data = self._x_preprocess(raw_df)
        
        x_data = x_data.astype(self.dtypes_map)

        x_data = self.categorical_encoder.transform(x_data)
        
        dtype_str = ', '.join([f"{col}: {dtype}" for col, dtype in x_data.dtypes.to_dict().items()])
        print(f'Dtypes after inference pre-processing: {dtype_str}')
        
        return x_data

    def _x_preprocess(self, raw_df: pd.DataFrame, return_target=False):
        """
        Function to execute preprocessing related to the rate estimation process for the input data
        - Subsetting the data frame to only include the input variables used in the model
        - Coverting categorical variables to string
        
        Paratmeters
        -----------
        raw_df: Data Frame like
            Raw Data Frame which needs to be preprocessed
            
        Returns
        -------
        Data Frame
            Pre processed data frame that will be fed as the input to the model
            
        """
        # Make sure we have all feature columns
        missing_columns = set(self.feature_columns) - set(raw_df.columns)
        assert len(missing_columns) == 0, "Missing columns: " + ','.join(missing_columns)
                
        x_data = raw_df[self.feature_columns].copy()
        
        # Make sure strings are handled correctly
        for col_name, col_data in x_data.select_dtypes(include=['category', 'object']).items():
            x_data[col_name] = x_data[col_name].apply(lambda x: str(x))
            x_data[col_name] = col_data.astype('string')
        
        return x_data