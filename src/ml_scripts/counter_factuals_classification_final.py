import logging
from dice_ml import Dice, Data, Model
from utils.utils_classification import _Model_Utility
from func_timeout import func_timeout, FunctionTimedOut

import pandas as pd
import numpy as np


class _Feature_Importance:
    """Class generates feature importance for given query instances.

    Methods:
        gen_local_feature_importance : Generates local feature importance for each Query Instance.
        gen_global_feature_importance : Generates global feature importance w.r.t all Query Instances given.
    """
    @staticmethod
    def _gen_local_feature_importance(cf_object: object, query_instances: pd.DataFrame, total_CFs: int, desired_class: int) -> list:
        """Generates feature importance for each query instance in the given query_instances dataframe.
            Cf's must be >= 12 to generate local feature importance.

        Args:
            cf_object (object): Dice object.
            query_instances (pd.DataFrame): DataFrame of query_instances
            total_CFs (int): No:of CF's to generate while calculating feature importance.
            desired_class (int): Output class desired.

        Returns:
            list: List of query_instances with their respective Feature Importance
        """

        local_feature_impt = []

        try:
            for i in range(len(query_instances)):
                importance = cf_object.local_feature_importance(query_instances[i:i+1], total_CFs=total_CFs, desired_class=desired_class)

                local_feature_impt.append({
                    'query_instance_index': i,
                    'query_instance_data': pd.DataFrame(query_instances[i:i+1]),
                    'query_instance_impt': importance.local_importance
                })
        except Exception as e:
            print(e)

        return local_feature_impt

    @staticmethod
    def _gen_global_feature_importance(cf_object: object, query_instances: pd.DataFrame, total_CFs: int, desired_class: int) -> list:
        """Generates feature importance for all query_instances at once.
             Cf's must be >= 12 and query_instances must be >= 12 to generate global feature importance.

        Args:
            cf_object (object): Dice object
            query_instances (pd.DataFrame): DataFrame of query_instances
            total_CFs (int): No:of CF's to generate while calculating feature importance.
            desired_class (int): Output class desired

        Returns:
            list: List of Variables with their respective Feature Importance
        """
        global_feature_impt = dict()
        try:
            global_feature_impt = cf_object.global_feature_importance(query_instances, total_CFs=total_CFs, desired_class=desired_class).summary_importance
        except Exception as e:
            print(e)
        return global_feature_impt

class CounterFactualClassification:
    """Class generates Counterfactuals for given query instances.

    Methods:
        __init__: Class Initialization Method.
        generate_cf_object : This method is used to create counter factual object using Data and Model objects.
        generate_cf_meta_object : This method creates counter factual meta object by generating counter factuals.
        calculate_min_standardized_difference : Method used to calculate the Min difference b/w standard actual and standard CF values.
        calculate_min_actual_difference : Method used to calculate the Min difference b/w actual and CF values.
        generate_cf_actual : Driver method that makes use of 'generate_cf_meta_object' method to create CFs.
        generate_counter_factuals : This method makes use of 'generate_cf_actual' method to generate counter factuals.
    """

    def __init__(self, model_meta_data: dict, desired_class: int, total_cfs: int, num_query_instances: int, method: str = 'genetic') -> None:
        """Class Initialization Method

        Args:
            model_meta_data (dict): Meta Data for CounterFactuals generation (
                Required Keys: [
                    'model_data': pd.DataFrame, 'numerical_columns': list, 'target_variable': np.array, 'model_object': object, 'test_data': pd.DataFrame
                ]
            )
            method (str): The type of method to use while calculating Counterfactuals. It can take values as "genetic"/"random"
            desired_class (int): Output class desired.
            num_of_CFs (int): Number of Counterfactuals to generate for each query_instance
            num_query_instances (int): Number of instances to take from x_test
        """

        self.model_meta_data = model_meta_data
        self.method = method

        self.desired_class = desired_class
        self.total_cfs = total_cfs
        self.num_query_instances = num_query_instances

    def generate_cf_object(self) -> tuple:
        """
        This function is responsible for creating counter_factual object.

        Returns : tuple of counter_factual object and query instances.
        """
        query_instances = self.model_meta_data['test_data'].iloc[:self.num_query_instances, :]

        data_object = Data(
            dataframe=self.model_meta_data['model_data'], continuous_features=self.model_meta_data['numerical_columns'],
            outcome_name=self.model_meta_data['target_variable_name']
        )

        model_object = Model(model=self.model_meta_data['model_object'], backend='sklearn', model_type='classifier')
        cf_object = Dice(data_object, model_object, method=self.method)
        return cf_object, query_instances


    def calculate_min_standardized_difference(self,feature: str, actual_ft_instance: int, cf_vals_ft: list) -> int:
        """
        Function to calculate standardized minimum distance b/w actual instance value and counter factuals for each query instance.

        Args:
        feature (str): feautre name for which the CF's have been generated.
        actual_ft_instance (int): actual feature value for a particular instance
        cf_vals_ft (list): specific feature's counter factual values of a particular instance

        return:
        min_diff (int): Standardized Minimum distance b/w actual instance value and counter factuals.
        """
        distance_vals_standard = [(val/self.model_meta_data['X_train'][feature].mean()) - (actual_ft_instance/self.model_meta_data['X_train'][feature].mean()) for val in cf_vals_ft]
        minimum_val_index = np.argmin(np.abs(distance_vals_standard))
        min_diff_standard = distance_vals_standard[minimum_val_index]

        return np.abs(min_diff_standard)

    def calculate_min_actual_difference(actual_ft_instance: int, cf_vals_ft: list) -> int:
        """
        Function to calculate minimum distance b/w actual instance value and counter factuals for each query instance.

        Args:
        actual_ft_instance (int): actual feature value for a particular instance
        cf_vals_ft (list): specific feature's counter factual values of a particular instance

        return:
        min_diff (int): Actual Minimum distance b/w actual instance value and counter factuals.
        """
        distance_vals = [val - actual_ft_instance for val in cf_vals_ft]
        minimum_val_index = np.argmin(np.abs(distance_vals))
        min_diff = distance_vals[minimum_val_index]

        return min_diff

    def generate_cf_meta_obj(self,feature:str,cf_object:object, query_instance:pd.DataFrame, permitted_range:dict = None) -> tuple:
        """
        Function to generate counter factuals.

        Args:
        feature (str): Feature to vary
        permitted_range (dict): Permitted range dictionary
        cf_object (object): Counter Factual object
        query_instance (pd.DataFrame): query instance for which counter factuals to be generated.

        Returns:
        cf_vals_ft (list): CF values generated for the query instance by allowing Feature to change.
        cf_df_instance_ft (pd.DataFrame): CF dataframe generated by varying only given feature for the query instance.
        """
        cf_vals_ft = list()
        cf_df_instance_ft = pd.DataFrame(columns = self.model_meta_data['model_data'].columns)  ## Counter Factual Dataframe
        if (permitted_range is not None) and (feature in permitted_range.keys()):
            cf_object_meta = cf_object.generate_counterfactuals(query_instance,self.total_cfs, self.desired_class, None, {feature : permitted_range[feature]},[feature])
        else:
            cf_object_meta = cf_object.generate_counterfactuals(query_instance,self.total_cfs,self.desired_class,None,None,[feature])

        if cf_object_meta.cf_examples_list[0].final_cfs_df is not None:
            cf_vals_ft = cf_vals_ft + list(cf_object_meta.cf_examples_list[0].final_cfs_df[feature])
            for cf_no in range(self.total_cfs):
                cf_df_instance_ft.loc[len(cf_df_instance_ft)] = list(cf_object_meta.cf_examples_list[0].final_cfs_df.values[cf_no])

        return cf_vals_ft, cf_df_instance_ft


    def generate_cf_actual(self, cf_object: object ,query_instances: pd.DataFrame,features_to_vary: list , permitted_range: dict = None)-> tuple:
        """
        This function is responsible for generating counter factuals.

        Args:
        cf_object (object): Counter_factual object
        query_instances (pd.DataFrame): query instances for which the CF's have to be generated.
        features_to_vary (list): Features to vary 1 at a time while generating CFs.
        permitted_range (dict): Permitted_range of feature values which are allowed to vary.

        Returns:
        final_feature_magnitude_change (list): actual magnitude change values for each query instance.
        final_feature_magnitude_change_standard (list): Standardized magnitude change values for each query instance.
        cf_df (pd.DataFrame): Counter factuals dataframe for all query instances.
        """
        final_feature_magnitude_change = list()
        final_feature_magnitude_change_standard = list()
        cf_df_final = pd.DataFrame(columns = self.model_meta_data['model_data'].columns)  ## Counter Factual Dataframe
        cf_df_instance = pd.DataFrame(columns = self.model_meta_data['model_data'].columns)
        for idx in range(self.num_query_instances):
            instance_dict = dict()
            instance_dict_std = dict()
            for feature in features_to_vary:
                if feature in self.model_meta_data['numerical_columns']:
                    print('Feature - ', feature )
                    #logging.info('Feature - ', feature)
                    actual_ft_instance = query_instances.iloc[idx:idx+1,:][feature ].values[0]
                    try:
                        cf_vals_ft, cf_df_instance_ft = func_timeout(1*60,CounterFactualClassification.generate_cf_meta_obj, args = (
                                                                            self,feature,cf_object,query_instances.iloc[idx:idx+1,:],permitted_range
                                                                            ))
                        print("cf_vals_ft - \n", cf_vals_ft)
                        cf_df_instance = pd.concat([cf_df_instance,cf_df_instance_ft], axis = 0)
                        instance_dict[feature] = CounterFactualClassification.calculate_min_actual_difference(actual_ft_instance = actual_ft_instance, cf_vals_ft = cf_vals_ft)
                        instance_dict_std[feature] = CounterFactualClassification.calculate_min_standardized_difference(self, feature = feature, actual_ft_instance = actual_ft_instance, cf_vals_ft = cf_vals_ft)

                    except(FunctionTimedOut,Exception) as e:
                        print(e)

                else:
                    print(f'{feature} is not in numerical format. Only numerical features are expected.')

            cf_df_final = pd.concat([cf_df_final,cf_df_instance])
            instance_dict_std = sorted(instance_dict_std.items(), key=lambda x: x[1])
            final_feature_magnitude_change.append({idx:instance_dict})
            final_feature_magnitude_change_standard.append({idx: instance_dict_std})

        return final_feature_magnitude_change, final_feature_magnitude_change_standard, cf_df_final


    def generate_counter_factuals(self, features_to_vary: list = None, permitted_range: dict = None) -> tuple:
        """function to generate counterfactuals.

        Args:
            features_to_vary (list): list of features which can be varied/modified while generating counterfactuals
            permitted_range (dict): This is the dictionary with feature_names as keys and permitted value range for those features as values(list).

        Returns:
            final_feature_magnitude_change (list): actual magnitude change values for each query instance.
            final_feature_magnitude_change_standard (list): Standardized magnitude change values for each query instance.
            cf_df (pd.DataFrame): Counter factuals dataframe for all query instances.
            local_feature_impt (list): List of query_instance wise respective Feature Importance.
            global_feature_impt (list): global feature importance w.r.t all Query Instances given.
        """

        cf_object, query_instances = CounterFactualClassification.generate_cf_object(self)
        if (features_to_vary is not None and permitted_range is not None) or (features_to_vary is None and permitted_range is not None):
            if features_to_vary is not None:
                features_to_vary = features_to_vary
            else:
                features_to_vary = self.model_meta_data['numerical_columns']
            final_feature_magnitude_change, final_feature_magnitude_change_standard,cf_df = CounterFactualClassification.generate_cf_actual(self,cf_object=cf_object,
                                                                                                    query_instances = query_instances,
                                                                                                    features_to_vary = features_to_vary,
                                                                                                    permitted_range = permitted_range)

        elif (features_to_vary is not None and permitted_range is None) or (features_to_vary is None and permitted_range is None):
            if features_to_vary is not None:
                features_to_vary = features_to_vary
            else:
                features_to_vary = self.model_meta_data['numerical_columns']
            final_feature_magnitude_change,final_feature_magnitude_change_standard,cf_df = CounterFactualClassification.generate_cf_actual(self, cf_object=cf_object,
                                                                                                    query_instances = query_instances,
                                                                                                    features_to_vary = features_to_vary)
        ### Generating Feature Importance.
        local_feature_impt = _Feature_Importance._gen_local_feature_importance(cf_object, query_instances, self.total_cfs, self.desired_class)
        global_feature_impt = _Feature_Importance._gen_global_feature_importance(cf_object, query_instances, self.total_cfs, self.desired_class)

        return final_feature_magnitude_change, final_feature_magnitude_change_standard, cf_df, local_feature_impt, global_feature_impt





if __name__ == '__main__':
    features_to_vary = ['Basecoat Humidity', 'Basecoat Temperature', 'Clearcoat Humidity', 'Clearcoat Temperature']
    permitted_range = {'Basecoat Humidity': [65, 75], 'Clearcoat Humidity': [65, 75], 'Basecoat Temperature': [23, 29], 'Clearcoat Temperature': [23, 29]}

    desired_class = 0
    total_cfs = 12
    num_query_instances = 12

    model_meta_data_config = {
        'model_name' : 'random_forest',
        'data_path' : './data/paint_quality_assurance_data.csv',
        'file_type' : 'csv',
        'target' : 'Dust Defect',
        'filter_value' : 1,
        'drop_cols_list' : ['Body Id']
    }

    model_meta_data = _Model_Utility.get_trained_data_model(**model_meta_data_config)

    cfc = CounterFactualClassification(model_meta_data=model_meta_data, desired_class=desired_class, total_cfs=total_cfs, num_query_instances=num_query_instances)
    magnitude_change, magnitude_change_standard, cf_df, local_feature_impt, global_feature_impt = cfc.generate_counter_factuals(features_to_vary=features_to_vary, permitted_range=permitted_range)


    print("Local feature importance - \n", local_feature_impt)
    print("Global feature imp - \n", global_feature_impt)
    print("Magnitude change - \n", magnitude_change)
    print("Standardized Magnitude change - \n",magnitude_change_standard)
    print("Exporting Counter Factuals DF ....")
    cf_df.to_csv('Classification_CF.csv',encoding='utf-8')
    print("Done")