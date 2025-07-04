from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from lightgbm import LGBMRegressor

from fnibot_model.fnibot_model import FNIBotModel

class TestModelTraining():

    @pytest.fixture
    def model_config(self):
        """
        Create model_config tests to use
        """
        config = {
            'feature_columns': ['f1', 'f2', 'f3'],
            'categorical_columns': [],
            'model_params': {
                'n_estimators': 10,
                'num_leaves': 6,
                'min_child_samples': 1,
                'early_stopping_round': 50,
                'first_metric_only': True,
                'seed': 42},
            'approval_model_params': {
                'metric': 'auc'},
            'rate_model_params': {
                'metric': 'mae',
                'objective': 'huber'}
        }

        yield config
        
    @pytest.fixture
    def sample_data(self):
        """
        Create sample training data
        """
        sample_data = pd.DataFrame({
            'f1': [1, 2, 1, 2, 5, 6],
            'f2': [1.2, 2.1, 3.8, 4.4, 9.2, 8.8],
            'f3': [4, 2, 5, 8, 7, 2],
            'f4': [11, 12, 0, 1, 22, 13],
            't_approval': ['Approved', 'Declined', 'Approved', 'Declined', 'Approved', 'Declined'],
            't_rate': [3.5, 12.2, 3.5, 9.9, 15.8, 5.4]
        })
        
        yield sample_data
     
    @pytest.mark.parametrize('model_type', ['rate', 'approval'])
    def test_prediction_format(self, model_type, model_config, sample_data):
        model_config['model_type'] = model_type
        model_config['target_column'] = f't_{model_type}'
        model_config['model_params'].update(model_config[f'{model_type}_model_params'])
        del model_config['rate_model_params']
        del model_config['approval_model_params']
        
        print(model_config)
        
        model = FNIBotModel(**model_config)
        model.fit(raw_train=sample_data, raw_val=sample_data)
        
        pred = model.predict(sample_data)
        assert pred.shape == (6,), 'Wrong output shape'
        assert pred.dtype == float, 'Wrong output value type'
        assert type(pred) == np.ndarray, 'Wrong output type'
            
    @pytest.mark.parametrize('model_type', ['rate', 'approval'])
    def test_model_trains(self, model_type, model_config, sample_data):
        model_config['model_type'] = model_type
        model_config['target_column'] = f't_{model_type}'
        model_config['model_params'].update(model_config[f'{model_type}_model_params'])
        del model_config['rate_model_params']
        del model_config['approval_model_params']
        
        print(model_config)
        
        model = FNIBotModel(**model_config)
        model.fit(raw_train=sample_data, raw_val=sample_data)
        
        pred = model.predict(sample_data)
        
        assert model.model.n_iter_ > 1, 'No training iterations'
        assert model.model.best_iteration_ > 1, 'Best iteration not > 1'
        
    def test_binary_output_range(self, model_config, sample_data):
        model_type = 'approval'
        model_config['model_type'] = model_type
        model_config['target_column'] = f't_{model_type}'
        model_config['model_params'].update(model_config[f'{model_type}_model_params'])
        del model_config['rate_model_params']
        del model_config['approval_model_params']
        
        model = FNIBotModel(**model_config)
        model.fit(raw_train=sample_data, raw_val=sample_data)
        
        pred = model.predict(sample_data)
        assert max(pred) <= 1.0, f'{model_type} prediction > 1'
        assert min(pred) >= 0.0, f'{model_type} prediction < 0'
        
        
        
        