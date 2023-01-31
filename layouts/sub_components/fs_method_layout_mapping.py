from dash import html, dcc
from src.dash_utils.config_input import fs_conf

import dash_bootstrap_components as dbc


class _Selection_Methods:

    @staticmethod
    def get_anova_f_value_selection_layout():
        return html.Div(
            [
                dbc.Label("ANOVA-F Selection Method (Params):"),
                html.Div(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Selected Number of Features:"),
                                dbc.Input(type="number", placeholder=1, value=0.5,
                                          id='anova_f_value_selection_params', max=fs_conf.transformed_data.shape[1], min=1, step=1),
                            ], className="mb-3",
                        )
                    ]
                ),
            ], style={'margin-top': '5px'}
        )

    @staticmethod
    def get_mutual_info_classif_selection_layout():
        return html.Div(
            [
                dbc.Label("Mutual Info. Classif (Params):"),
                html.Div(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Selected Number of Features:"),
                                dbc.Input(type="number", placeholder=1, value=0.5,
                                          id='mutual_info_classif_selection_params', max=fs_conf.transformed_data.shape[1], min=1, step=1),
                            ], className="mb-3",
                        )
                    ]
                ),
            ], style={'margin-top': '5px'}
        )

    # @staticmethod
    # def get_drop_unique_value_columns_layout():
    #     return html.Div(
    #         [
    #             dbc.Label("Drop Unique Value Columns (Params):"),
    #             html.Div(
    #                 [
    #                     dbc.InputGroup(
    #                         [
    #                             dbc.InputGroupText("Unique Value Threshold:"),
    #                             dbc.Input(type="number", placeholder=1,
    #                                       value=1, id='drop_unique_value_columns_params', min=0, step=0.1)
    #                         ], className="mb-3",
    #                     )
    #                 ]
    #             ),
    #         ], style={'margin-top': '5px'}
    #     )

    # @staticmethod
    # def get_data_imputation_layout():
    #     return html.Div(
    #         [
    #             dbc.Label("Imputation Type (Params):"),
    #             dcc.Dropdown(
    #                 id='data_imputation_params',
    #                 value=list(dt_conf.imputation_methods)[0],
    #                 multi=False,
    #                 options=[
    #                     {'label': method.capitalize(), 'value': method} for method in list(dt_conf.imputation_methods)
    #                 ],
    #                 placeholder='Select Method'
    #             ),
    #         ], style={'margin-top': '5px'}
    #     )

    # @staticmethod
    # def get_feature_scaling_layout():
    #     return html.Div(
    #         [
    #             dbc.Label("Feature Scaling Type (Params):"),
    #             dcc.Dropdown(
    #                 id='feature_scaling_params',
    #                 value=list(dt_conf.feature_scaling_methods)[0],
    #                 multi=False,
    #                 options=[
    #                     {'label': method.capitalize(), 'value': method} for method in list(dt_conf.feature_scaling_methods)
    #                 ],
    #                 placeholder='Select Features'
    #             )
    #         ], style={'margin-top': '5px'}
    #     )

    # @staticmethod
    # def get_feature_transformer_layout():
    #     return html.Div(
    #         [
    #             dbc.Label("Feature Transform Type (Params):"),
    #             dcc.Dropdown(
    #                 id='feature_transformer_params',
    #                 value=list(dt_conf.feature_transform_methods)[0],
    #                 multi=False,
    #                 options=[
    #                     {'label': method.capitalize(), 'value': method} for method in list(dt_conf.feature_transform_methods)
    #                 ],
    #                 placeholder='Select Features'
    #             ),
    #         ], style={'margin-top': '5px'}
    #     )


method_layout_mapping = {
    'anova_f_value_selection': _Selection_Methods.get_anova_f_value_selection_layout(),
    'mutual_info_classif_selection': _Selection_Methods.get_mutual_info_classif_selection_layout(),
    # 'drop_unique_value_columns': _Selection_Methods.get_drop_unique_value_columns_layout(),
    # 'data_imputation': _Selection_Methods.get_data_imputation_layout(),
    # 'feature_scaling': _Selection_Methods.get_feature_scaling_layout(),
    # 'feature_transformer': _Selection_Methods.get_feature_transformer_layout(),
}
