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
    def get_ks_score(ytrue: np.array, yprob: np.array):
        df = pd.DataFrame({'true_class': ytrue, 'pred_prob': yprob})

        return ks_2samp(df[df['true_class'] == 0]['pred_prob'], df[df['true_class'] == 1]['pred_prob']).statistic


    @staticmethod
    def _roc_threshold(y_train: np.array, y_prob: np.array) -> float:
        """
        Calculate Cutoff Threshold based on ROC Curve.

        Args:
            y_train (np.array): True Target Values
            y_prob (np.array): Predicted Target Values

        Returns:
            float: Threshold Value
        """

        fpr, tpr, threshold = roc_curve(y_train, y_prob)
        index = np.arange(len(tpr))

        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=index), 'threshold': pd.Series(threshold, index=index)})
        roc = roc.iloc[(roc['tf'] - 0).abs().argsort()[:1]]

        return list(roc['threshold'])[0]


    @staticmethod
    def _pr_threshold(y_train: np.array, y_prob: np.array) -> float:
        """
        Calculate Cutoff Threshold based on Precision-Recall Curve.

        Args:
            y_train (np.array): True Target Values
            y_prob (np.array): Predicted Target Values

        Returns:
            float: Threshold Value
        """

        precision, recall, threshold = precision_recall_curve(y_train, y_prob)
        index = np.arange(len(recall[:-1]))

        roc = pd.DataFrame({'tf': pd.Series(precision[:-1] - recall[:-1], index=index), 'threshold': pd.Series(threshold, index=index)})
        roc = roc.iloc[(roc['tf'] - 0).abs().argsort()[:1]]

        return list(roc['threshold'])[0]


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



class _Model_Report:

    @staticmethod
    def _get_train_test_data(data_dict: dict):
        return (data_dict['X_train'], data_dict['y_train'], data_dict['X_test'], data_dict['y_test'])


    @staticmethod
    def _get_model_params(params: dict):
        if params['model_name'] == 'lightgbm':
            if params['lgbm_max_depth'] == -1:
                params['lgbm_num_leaves'] = 31

            elif params['lgbm_max_depth'] > -1 and params['lgbm_max_depth'] < 7:
                params['lgbm_num_leaves'] = int(2 ** params['lgbm_max_depth'])

            else:
                params['lgbm_num_leaves'] = int(2 ** params['lgbm_max_depth'] / 2)

            return {'_'.join(key.split('_')[1:]): value for key, value in params.items() if key not in ['model_name', 'dataset_name', 'sample_method']}

        return {'_'.join(key.split('_')[1:]): value for key, value in params.items() if key not in ['model_name', 'dataset_name', 'sample_method']}


    @staticmethod
    def _get_classifier(model_name: str, params: dict) -> object:
        model_schema = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier,
            'logistic_regression': LogisticRegression,
            'svm_rbf': SVC,
            'mlp_classifier': MLPClassifier
        }

        return model_schema[model_name](random_state=0, **params)


    @staticmethod
    def _train_pipeline(X_train: pd.DataFrame, y_train: np.array, model: object, sample_method: object):
        pipeline = None

        if sample_method is not None:
            pipeline = imbpipeline(steps=[('sample', sample_method), ('model', model)])
        else:
            pipeline = imbpipeline(steps=[('model', model)])

        cv_fold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv_fold, scoring=_Utils._get_scorers(), n_jobs=-1)

        pipeline.fit(X_train, y_train)

        return pipeline, cv_results


    @staticmethod
    def _generate_predictions(pipeline: object, data_tuple: tuple):
        X_train, y_train, X_test, y_test = data_tuple

        results = {
            'y_prob_train': pipeline.predict_proba(X_train),

            'y_true_test': y_test,
            'y_pred_test': pipeline.predict(X_test),
            'y_prob_test': pipeline.predict_proba(X_test)
        }

        thresh_roc = _Utils._roc_threshold(y_train=y_train, y_prob=results['y_prob_train'][:, 1])
        thresh_pr = _Utils._pr_threshold(y_train=y_train, y_prob=results['y_prob_train'][:, 1])

        results['y_pred_opt'] = [1 if j > thresh_roc else 0 for j in results['y_prob_test'][:, 1]]
        results['y_pred_pr'] = [1 if j > thresh_pr else 0 for j in results['y_prob_test'][:, 1]]

        return results, thresh_pr, thresh_roc


    @staticmethod
    def _get_kpi_metrics(params: dict, cv_results: dict, y_test: np.array, results: dict, thresh_roc: float, thresh_pr: float):
        cv_result_data = [
            params['model_name'], params['dataset_name'], params.get('sample_method', 'balanced_data (None)'), 0.5, 'CV Scores',
            round(np.mean(cv_results["test_accuracy"]), 4), round(np.mean(cv_results["test_precision"]), 4), round(np.mean(cv_results["test_recall"]), 4),
            round(np.mean(cv_results["test_f1"]), 4), round(np.mean(cv_results["test_roc_auc"]), 4)
        ]

        test_result_data = [
            params['model_name'], params['dataset_name'], params.get('sample_method', 'balanced_data (None)'), 0.5, 'Test Data',
            round(accuracy_score(y_test, results['y_pred_test']), 4), round(precision_score(y_test, results['y_pred_test']), 4),
            round(recall_score(y_test, results['y_pred_test']), 4), round(f1_score(y_test, results['y_pred_test']), 4),
            round(roc_auc_score(y_test, results['y_prob_test'][:, 1]), 4)
        ]

        roc_result_data = [
            params['model_name'], params['dataset_name'], params.get('sample_method', 'balanced_data (None)'), round(thresh_roc, 3), 'ROC Threshold',
            round(accuracy_score(y_test, results['y_pred_opt']), 4), round(precision_score(y_test, results['y_pred_opt']), 4),
            round(recall_score(y_test, results['y_pred_opt']), 4), round(f1_score(y_test, results['y_pred_opt']), 4),
            round(roc_auc_score(y_test, results['y_prob_test'][:, 1]), 4)
        ]

        pr_result_data = [
            params['model_name'], params['dataset_name'], params.get('sample_method', 'balanced_data (None)'), round(thresh_pr, 3), 'PR Threshold',
            round(accuracy_score(y_test, results['y_pred_pr']), 4), round(precision_score(y_test, results['y_pred_pr']), 4),
            round(recall_score(y_test, results['y_pred_pr']), 4), round(f1_score(y_test, results['y_pred_pr']), 4),
            round(roc_auc_score(y_test, results['y_prob_test'][:, 1]), 4)
        ]

        metrics = [cv_result_data, test_result_data, roc_result_data, pr_result_data]

        return pd.DataFrame(data=metrics, columns=[
            'Model Name', 'Feature Set', 'Sampling Method', 'Cut-Off Threshold', 'Category', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'
        ])


    @staticmethod
    def _get_decile_data(y_prob: np.array, y_pred: np.array, y_test: np.array, num_deciles: int) -> pd.DataFrame:
        prob_defect = pd.DataFrame({'probabilities': y_prob, 'actual_class': y_test, 'pred_class': y_pred})
        labels = [f'D{i}' for i in range(1, num_deciles + 1)]

        prob_defect['Decile'] = pd.qcut(prob_defect['probabilities'], num_deciles, labels=labels)
        prob_defect['Range'] = pd.qcut(prob_defect['probabilities'], num_deciles, retbins = True)[0]

        report = prob_defect.groupby('Decile').agg({
            'probabilities': 'mean', 'actual_class': 'sum', 'pred_class': 'sum', 'Decile': 'size'
        })

        report = report.rename(columns={
            'probabilities': 'Avg. Prob', 'actual_class': 'Actual Defective', 'pred_class': 'Predictive Defective', 'Decile': 'Total Engines'
        })

        report['Avg. Prob'] = round(report['Avg. Prob'], 3)
        report['Range'] = prob_defect['Range'].unique().sort_values()

        report['Actual Non Defective'] = report['Total Engines'] - report['Actual Defective']
        report['Predicted Non Defective'] = report['Total Engines'] - report['Predictive Defective']
        report['Defect Rate'] = report['Actual Defective'] / report['Total Engines']
        report['Defect Rate'] = round(report['Defect Rate'], 3)

        report = report.loc[:, [
            'Range', 'Total Engines', 'Avg. Prob', 'Defect Rate', 'Actual Defective', 'Actual Non Defective',
            'Predictive Defective', 'Predicted Non Defective'
        ]]

        return report.reset_index()


    @staticmethod
    def _main(best_trials: list, data_dict: dict, sample_schema: dict, num_deciles: int):
        model_reports = list()

        for trial in best_trials:
            feature_set = data_dict[trial.params['dataset_name']]
            sample_method, decile_report = None, dict()

            if 'sample_method' in trial.params.keys():
                sample_method = sample_schema[trial.params['sample_method']]

            X_train, y_train, X_test, y_test = _Model_Report._get_train_test_data(feature_set)

            model_params = _Model_Report._get_model_params(params=trial.params)
            model = _Model_Report._get_classifier(model_name=trial.params['model_name'], params=model_params)

            pipeline, cv_results = _Model_Report._train_pipeline(X_train=X_train, y_train=y_train, model=model, sample_method=sample_method)

            data_tuple = (X_train, y_train, X_test, y_test)
            results, thresh_pr, thresh_roc = _Model_Report._generate_predictions(pipeline, data_tuple)

            metrics_df = _Model_Report._get_kpi_metrics(
                params=trial.params, cv_results=cv_results, y_test=y_test, results=results, thresh_roc=thresh_roc, thresh_pr=thresh_pr
            )

            for category in ['y_pred_test', 'y_pred_opt', 'y_pred_pr']:
                decile_report[category] = _Model_Report._get_decile_data(
                    y_prob=results['y_prob_test'][:, 1], y_pred=results[category], y_test=results['y_true_test'], num_deciles=num_deciles
                )

            model_reports.append({
                'Trial Number': f'{trial.number}', 'Feature List': list(X_train.columns), 'KPI Metrics': metrics_df, 'Results': results,
                'Decile Reports': decile_report
            })

        return model_reports



class _Hyper_Parameter_Tuning:

    def __init__(self, tuning_config: dict, data_dict: dict, sample_schema: dict, tuning_parameter: str = 'accuracy'):
        self._tuning_config = tuning_config
        self._data_dict = data_dict
        self._tuning_parameter = tuning_parameter
        self._sample_schema = sample_schema

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

        model_name = trial.suggest_categorical('model_name', self._tuning_config['model'])
        feature_set_name = trial.suggest_categorical('dataset_name', self._tuning_config['data_set'])

        if not self._tuning_config['sample_name'][0] == 'balanced_data (None)':
            sample_method = trial.suggest_categorical('sample_method', self._sample_schema.keys())

        X_train, y_train = self._data_dict[feature_set_name]['X_train'], self._data_dict[feature_set_name]['y_train']

        classifier = self._model_schema[model_name](trial=trial)

        if sampler_method is not None:
            pipeline = imbpipeline(steps=[('sample', self._sample_schema[sample_method]), ('model', classifier)])

        else:
            pipeline = imbpipeline(steps=[('model', classifier)])

        cv_fold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv_fold, scoring=_Utils._get_scorers(), n_jobs=-1)

        return np.mean(cv_results[f'test_{self._tuning_parameter}'])


    def _initiate_study(self, n_trials: int = 100):
        study = optuna.create_study(direction='maximize', study_name='Optuna Study v1', sampler=optuna.samplers.RandomSampler(seed=0))
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

    def load_datasets(self, file_path: str) -> dict:
        """
        Load Feature Selected Datasets.

        Args:
            file_path (str): Path of saved Data

        Returns:
            dict: Feature Selected Data
        """
        return joblib.load(file_path)['input_data']


    def load_baseline_results(self, file_path: str) -> dict:
        """
        Load Feature Selected Datasets.

        Args:
            file_path (str): Path of saved Data

        Returns:
            dict: Feature Selected Data
        """
        return joblib.load(file_path)['best_result']


    def tune_models(self, baseline_results: dict, data_dict: dict, tuning_parameter: str, n_trials: int) -> object:
        return _Hyper_Parameter_Tuning(
            tuning_config=baseline_results, data_dict=data_dict, sample_schema=self._sample_schema, tuning_parameter=tuning_parameter
        )._initiate_study(n_trials=n_trials)


    def generate_model_report(self, best_trials: list, data_dict: dict, num_deciles: int) -> None:
        report = _Model_Report._main(best_trials=best_trials, data_dict=data_dict, sample_schema=self._sample_schema, num_deciles=num_deciles)


    def save_results(self, optuna_study: object, report: list, save_path: str) -> None:
        """Save the Baseline Excel and the Best Results as a Joblib file.

        Args:
            results (list): Results Data List
            sort_by (list): Sort Baseline Report by KPI
            save_path (dict): Path to save results.
        """
        joblib.dump({'optuna_study': optuna_study, 'report': report}, save_path)


    def compile_tuning(self, file_paths: dict, tuning_parameter: str, n_trials: int, num_deciles: int, save_path: str) -> None:
        """Compiles all the Methods to run the HyperParameter Tuning.

        Args:
            file_path (str): Data File Path
            data_type (str): Balance of Data
            kpi_sorting (list): Sorting Metrics
            save_path (dict): Path to Save File
        """

        data_dict = self.load_datasets(file_path=file_paths['data_set'])
        baseline_results = self.load_baseline_results(file_path=file_paths['baseline_result'])

        optuna_study = self.tune_models(baseline_results=baseline_results, data_dict=data_dict, tuning_parameter=tuning_parameter, n_trials=n_trials)
        report = self.generate_model_report(best_trials=optuna_study.best_trials, data_dict=data_dict, num_deciles=num_deciles)

        self.save_results(optuna_study=optuna_study, report=report, save_path=save_path)

        print(f'\nParameter Tuning Finished\n{"-" * 100}')



if __name__ == '__main__':

    # Example Configuration File for Baseline Modelling.
    config = {
        'file_paths': {
            'data_set': '../data/interim_data/feature_selected_data_v1.joblib',
            'baseline_result': '../results/baseline/baseline_best_results_v1.joblib'
        },
        'tuning_parameter': 'accuracy',
        'n_trials': 5,
        'num_deciles': 10,
        'save_path': '../results/tuning/tuning_results_v1.joblib'
    }

    Tuning().compile_tuning(**config)