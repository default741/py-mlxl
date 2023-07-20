import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer, fbeta_score, roc_curve, auc, f1_score, roc_auc_score, recall_score, precision_score, accuracy_score,
    confusion_matrix, classification_report, precision_recall_curve
)

from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, RepeatedEditedNearestNeighbours, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN

from scipy.stats import ks_2samp

import optuna
import joblib


class _Read_Data_File:
    """Protected class to Read Data with respect to its File Type. Currently Supports
    three file types namely CSV, EXCEL (xlsx), Parquet.

    Methods:
        _read_csv_type: Reads CSV file type
        _read_excel_type: Reads XLSX file type
        _read_parquet_type: Reads PARQUET file type
    """

    @staticmethod
    def _read_csv_type(file_path: str, params: dict) -> pd.DataFrame:
        """Reads CSV file type using Pandas Library (read_csv).

        Args:
            file_path (str): Path to read the file from.
            params (dict): Extra Parameters for the Method

        Returns:
            pd.DataFrame: Raw Data File
        """
        return pd.read_csv(file_path, **params)


class _Utils:
    """Utility Class for Helping Fulctions

    Methods:
        _fbeta_eq_two_scorer: Calculates FBeta with beta=2 for weightage on Recall.
        _fbeta_eq_half_scorer: Calculates FBeta with beta=0.5 for weightage on Precision.
        _roc_threshold: Calculate Cutoff Threshold based on ROC Curve.
        _pr_threshold: Calculate Cutoff Threshold based on Precision-Recall Curve.
        _get_scorers: Returns Scoring Metrics.
    """

    @staticmethod
    def _fbeta_eq_two_scorer(ytrue: np.array, ypred: np.array) -> float:
        """
        Calculates FBeta with beta=2 for weightage on Recall.

        Args:
            yture (np.array): True Target Values
            ypred (np.array): Predicted Target Values

        Returns:
            float: FBeta Score
        """
        return fbeta_score(ytrue, ypred, beta=2)

    @staticmethod
    def _fbeta_eq_half_scorer(ytrue: np.array, ypred: np.array) -> float:
        """
        Calculates FBeta with beta=0.5 for weightage on Precision.

        Args:
            yture (np.array): True Target Values
            ypred (np.array): Predicted Target Values

        Returns:
            float: FBeta Score
        """
        return fbeta_score(ytrue, ypred, beta=0.5)

    @staticmethod
    def _get_scorers() -> dict:
        """
        Returns Scoring Metrics.

        Returns:
            dict: Scoring Metrics
        """

        return {
            'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1',
            'f2': make_scorer(_Utils._fbeta_eq_two_scorer, greater_is_better=True),
            'f0.5': make_scorer(_Utils._fbeta_eq_half_scorer, greater_is_better=True),
            'roc_auc': 'roc_auc',
        }


class _Parameter_Suggestion:

    @staticmethod
    def _random_forest_classifier(trial: object):
        params = {
            'max_depth': trial.suggest_int('rf_max_depth', 2, 32, log=True),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 502, step=50),
            'max_leaf_nodes': trial.suggest_categorical('rf_max_leaf_nodes', [None] + [i for i in range(2, 75, 5)]),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 50, 205, step=5),
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 600, step=100)
        }

        return RandomForestClassifier(random_state=0, n_jobs=-1, **params)

    @staticmethod
    def _gradient_boosting_classifier(trial: object):
        params = {
            'n_estimators': trial.suggest_int('gb_n_estimators', 100, 600, step=100),
            'max_depth': trial.suggest_int('gb_max_depth', 2, 32, log=True),
            'min_samples_split': trial.suggest_int('gb_min_samples_split', 2, 502, step=50),
            'min_samples_leaf': trial.suggest_int('gb_min_samples_leaf', 50, 205, step=5),
            'max_leaf_nodes': trial.suggest_categorical('gb_max_leaf_nodes', [None] + [i for i in range(2, 75, 5)])
        }

        return GradientBoostingClassifier(random_state=0, **params)

    @staticmethod
    def _xgboost_classifier(trial: object):
        params = {
            'max_depth': trial.suggest_int('xgb_max_depth', 2, 32, log=True),
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 600, step=100)
        }

        return XGBClassifier(random_state=0, n_jobs=-1, **params)

    @staticmethod
    def _lightgbm_classifier(trial: object):
        params = {
            'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 600, step=100),
            'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 6, 51, step=5),
            'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.5, 1, step=0.1),
            'max_depth': trial.suggest_int('lgbm_max_depth', -1, 11, step=3)
        }

        if params['max_depth'] == -1:
            params['num_leaves'] = 31

        elif params['max_depth'] > -1 and params['max_depth'] < 7:
            params['num_leaves'] = int(2 ** params['max_depth'])

        else:
            params['num_leaves'] = int(2 ** params['max_depth'] / 2)

        return LGBMClassifier(random_state=0, n_jobs=-1, **params)

    @staticmethod
    def _logistic_regression(trial: object):
        params = {}

        return LogisticRegression(random_state=0, n_jobs=-1, **params)

    @staticmethod
    def _svm_rbf_classifier(trial: object):
        params = {}

        return SVC(random_state=0, n_jobs=-1, **params)

    @staticmethod
    def _mlp_classifier(trial: object):
        params = {}

        return MLPClassifier(random_state=0, n_jobs=-1, **params)


class _Hyper_Parameter_Tuning:

    def __init__(self, tuning_config: dict, data_dict: dict, sample_schema: dict, selected_features: dict, tuning_parameter: str = 'accuracy'):
        self._tuning_config = tuning_config
        self._data_dict = data_dict
        self._tuning_parameter = tuning_parameter
        self._sample_schema = sample_schema
        self._selected_features = selected_features

        self._model_schema = {
            'random_forest': _Parameter_Suggestion._random_forest_classifier,
            'gradient_boosting': _Parameter_Suggestion._gradient_boosting_classifier,
            'xgboost': _Parameter_Suggestion._xgboost_classifier,
            'lightgbm': _Parameter_Suggestion._lightgbm_classifier,
            'logistic_regression': _Parameter_Suggestion._logistic_regression,
            'svm_rbf': _Parameter_Suggestion._svm_rbf_classifier,
            'mlp_classifier': _Parameter_Suggestion._mlp_classifier
        }

    def _objective(self, trial):
        sampler_method, pipeline = None, None

        model_name = trial.suggest_categorical(
            'model_name', self._tuning_config['model'])
        feature_set_name = trial.suggest_categorical(
            'dataset_name', self._tuning_config['feature_set'])

        if not self._tuning_config['sample_name'][0] == 'balanced_data (None)':
            sample_method = trial.suggest_categorical(
                'sample_method', self._sample_schema.keys())

        X_train, y_train = self._data_dict['X_train'][list(
            self._selected_features[feature_set_name])], self._data_dict['y_train'].values.ravel()

        classifier = self._model_schema[model_name](trial=trial)

        if sampler_method is not None:
            pipeline = imbpipeline(
                steps=[('sample', self._sample_schema[sample_method]), ('model', classifier)])

        else:
            pipeline = imbpipeline(steps=[('model', classifier)])

        cv_fold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv_fold, scoring=_Utils._get_scorers(), n_jobs=-1)

        return np.mean(cv_results[f'test_{self._tuning_parameter[0]}'])

    def _initiate_study(self, n_trials: int = 100):
        study = optuna.create_study(
            direction='maximize', study_name='Optuna Study v1', sampler=optuna.samplers.RandomSampler(seed=0))
        study.optimize(self._objective, n_trials=n_trials)

        return study


class Tuning:

    def __init__(self) -> None:

        self._sample_schema = {
            'smote (up)': SMOTE(random_state=0, n_jobs=-1),
            'adasyn (up)': ADASYN(random_state=0, n_jobs=-1),
            'svm_smote (up)': SVMSMOTE(random_state=0, n_jobs=-1),
            'boderline_smote (up)': BorderlineSMOTE(random_state=0, n_jobs=-1),
            'k_means_smote (up)': KMeansSMOTE(random_state=0, n_jobs=-1),

            'near_miss (down)': NearMiss(sampling_strategy='majority', n_jobs=-1),
            'renn (down)': RepeatedEditedNearestNeighbours(sampling_strategy='majority', n_jobs=-1),
            'tomek_links (down)': TomekLinks(sampling_strategy='majority', n_jobs=-1),

            'smote_tomek (combine)': SMOTETomek(random_state=0, n_jobs=-1),
            'smote_enn (combine)': SMOTEENN(random_state=0, n_jobs=-1)
        }

    def compile_tuning(
        self, file_paths: dict, tuning_parameter: str, n_trials: int, save_path: str, **kwargs
    ) -> None:
        """Compiles all the Methods to run the HyperParameter Tuning.

        Args:
            file_path (str): Data File Path
            data_type (str): Balance of Data
            kpi_sorting (list): Sorting Metrics
            save_path (dict): Path to Save File
        """

        selected_features_dict = dict()

        selected_features = _Read_Data_File._read_csv_type(
            file_path=file_paths['selected_feature'], params=kwargs)

        for cols in selected_features.columns:
            selected_features_dict[cols] = list(
                selected_features[cols].dropna())

        baseline_results = _Read_Data_File._read_csv_type(
            file_path=file_paths['baseline_result'], params=kwargs)

        data_dict = {
            'X_train': _Read_Data_File._read_csv_type(file_path=file_paths['X_train'], params=kwargs),
            'y_train': _Read_Data_File._read_csv_type(file_path=file_paths['y_train'], params=kwargs)
        }

        baseline_results = baseline_results.iloc[:10][[
            'feature_set', 'sample_name', 'model']].to_dict(orient='list')
        baseline_results = {key: list(set(value))
                            for key, value in baseline_results.items()}

        print(baseline_results)
        print()
        print(selected_features_dict)

        optuna_study = _Hyper_Parameter_Tuning(
            tuning_config=baseline_results, data_dict=data_dict, sample_schema=self._sample_schema, tuning_parameter=tuning_parameter, selected_features=selected_features_dict)._initiate_study(n_trials=n_trials)

        print(optuna_study.best_trial)

        # report = self.generate_model_report(
        #     best_trials=optuna_study.best_trials, data_dict=data_dict, num_deciles=num_deciles)

        # self.save_results(optuna_study=optuna_study,
        #                   report=report, save_path=save_path)

        # print(f'\nParameter Tuning Finished\n{"-" * 100}')


if __name__ == '__main__':

    # Example Configuration File for Baseline Modelling.
    config = {
        'file_paths': {
            'X_train': './X_train_v0.0.csv',
            'y_train': './y_train_v0.0.csv',
            'selected_feature': './selected_feature_list_v0.0.csv',
            'baseline_result': './baseline_results.csv'
        },
        'tuning_parameter': ['accuracy'],
        'n_trials': 5,
        'save_path': '../results/tuning/tuning_results_v1.joblib'
    }

    Tuning().compile_tuning(**config)
