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
                                dbc.Input(type="number", placeholder=1, value=15,
                                          id='anova_f_value_selection_params', max=30, min=1, step=1),
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
                                dbc.Input(type="number", placeholder=1, value=15,
                                          id='mutual_info_classif_selection_params', max=30, min=1, step=1),
                            ], className="mb-3",
                        )
                    ]
                ),
            ], style={'margin-top': '5px'}
        )

    @staticmethod
    def get_logit_selection_layout():
        return html.Div(
            [
                dbc.Label("Logit Selection Fit Method (Params):"),
                dcc.Dropdown(
                    id='logit_selection_params',
                    value=list(fs_conf.logit_fit_methods)[0],
                    multi=False,
                    options=[
                        {'label': method.capitalize(), 'value': method} for method in list(fs_conf.logit_fit_methods)
                    ],
                    placeholder='Select Method'
                ),
            ], style={'margin-top': '5px'}
        )

    @staticmethod
    def get_permutation_impt_selection_layout():
        return html.Div(
            [
                dbc.Label("Permutation Importance Model Selection (Params):"),
                dcc.Dropdown(
                    id='permutation_impt_selection_params',
                    value=list(fs_conf.perm_impt_model_list)[0],
                    multi=True,
                    options=[
                        {'label': method.capitalize(), 'value': method} for method in list(fs_conf.perm_impt_model_list)
                    ],
                    placeholder='Select Method'
                ),
                html.Div(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Selected Number of Features:"),
                                dbc.Input(type="number", placeholder=1, value=15,
                                          id='permutation_impt_selection_feat_num_params', max=30, min=1, step=1),
                            ], className="mb-3",
                        )
                    ], style={'margin-top': '10px'}
                ),
            ], style={'margin-top': '5px'}
        )

    @staticmethod
    def get_recursive_feature_elimination_layout():
        return html.Div(
            [
                dbc.Label("Permutation Importance Model Selection (Params):"),
                dcc.Dropdown(
                    id='recursive_feature_elimination_params',
                    value=list(fs_conf.rfe_model_list)[0],
                    multi=True,
                    options=[
                        {'label': method.capitalize(), 'value': method} for method in list(fs_conf.rfe_model_list)
                    ],
                    placeholder='Select Method'
                ),
                html.Div(
                    [
                        dbc.Label(
                            "If 0 then Step Value will be determined Heuristically:"),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Selected Number of Features:"),
                                dbc.Input(type="number", placeholder=1, value=15,
                                          id='recursive_feature_elimination_feat_num_params', max=30, min=0, step=1),
                            ], className="mb-3",
                        )
                    ]
                ),
                html.Div(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Specify Step Value for RFE:"),
                                dbc.Input(type="number", placeholder=1, value=1,
                                          id='recursive_feature_elimination_step_value_params', max=30, min=1, step=1),
                            ], className="mb-3",
                        )
                    ]
                ),
            ], style={'margin-top': '5px'}
        )

    @staticmethod
    def get_model_based_importance_layout():
        return html.Div(
            [
                dbc.Label("Model Based Importance Model Selection (Params):"),
                dcc.Dropdown(
                    id='model_based_importance_params',
                    value=list(fs_conf.model_based_impt_model_list)[0],
                    multi=True,
                    options=[
                        {'label': method.capitalize(), 'value': method} for method in list(fs_conf.model_based_impt_model_list)
                    ],
                    placeholder='Select Method'
                ),
                html.Div(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Selected Number of Features:"),
                                dbc.Input(type="number", placeholder=1, value=15,
                                          id='model_based_importance_feat_num_params', max=30, min=1, step=1),
                            ], className="mb-3",
                        )
                    ], style={'margin-top': '10px'}
                ),
            ], style={'margin-top': '5px'}
        )

    @staticmethod
    def get_regularization_selection_layout():
        return html.Div(
            [
                dbc.Label(
                    "Regularization Based Selection Model Selection (Params):"),
                dcc.Dropdown(
                    id='regularization_selection_params',
                    value=list(fs_conf.reg_based_model_list)[0],
                    multi=True,
                    options=[
                        {'label': method.capitalize(), 'value': method} for method in list(fs_conf.reg_based_model_list)
                    ],
                    placeholder='Select Method'
                ),
                html.Div(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Selected Number of Features:"),
                                dbc.Input(type="number", placeholder=1, value=15,
                                          id='regularization_selection_feat_num_params', max=30, min=1, step=1),
                            ], className="mb-3",
                        )
                    ], style={'margin-top': '10px'}
                ),
            ], style={'margin-top': '5px'}
        )

    @staticmethod
    def get_boruta_selection_layout():
        return html.Div(
            [
                dbc.Label(
                    "Boruta Model Selection (Params):"),
                dcc.Dropdown(
                    id='boruta_selection_params',
                    value=list(fs_conf.boruta_model_list)[0],
                    multi=True,
                    options=[
                        {'label': method.capitalize(), 'value': method} for method in list(fs_conf.boruta_model_list)
                    ],
                    placeholder='Select Method'
                ),
            ], style={'margin-top': '5px'}
        )

    @staticmethod
    def get_sequencial_forward_selection_layout():
        return html.Div(
            [
                dbc.Label(
                    "Sequencial Forward Selection Model Selection (Params):"),
                dcc.Dropdown(
                    id='sequencial_forward_selection_params',
                    value=list(fs_conf.sfs_model_list)[0],
                    multi=True,
                    options=[
                        {'label': method.capitalize(), 'value': method} for method in list(fs_conf.sfs_model_list)
                    ],
                    placeholder='Select Method'
                ),
                html.Div(
                    [
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Selected Number of Features:"),
                                dbc.Input(type="number", placeholder=1, value=15,
                                          id='sequencial_forward_selection_feat_num_params', max=30, min=1, step=1),
                            ], className="mb-3",
                        )
                    ], style={'margin-top': '10px'}
                ),

                dbc.Label(
                    "Sequencial Forward Selection Scoring Metric (Params):"),
                dcc.Dropdown(
                    id='sequencial_forward_selection_scoring_mertric_params',
                    value=list(fs_conf.sfs_scoring_metrics)[0],
                    multi=False,
                    options=[
                        {'label': method.capitalize(), 'value': method} for method in list(fs_conf.sfs_scoring_metrics)
                    ],
                    placeholder='Select Method'
                ),
            ], style={'margin-top': '5px'}
        )


method_layout_mapping = {
    'anova_f_value_selection': _Selection_Methods.get_anova_f_value_selection_layout(),
    'mutual_info_classif_selection': _Selection_Methods.get_mutual_info_classif_selection_layout(),
    'logit_selection': _Selection_Methods.get_logit_selection_layout(),
    'permutation_impt_selection': _Selection_Methods.get_permutation_impt_selection_layout(),
    'recursive_feature_elimination': _Selection_Methods.get_recursive_feature_elimination_layout(),
    'model_based_importance': _Selection_Methods.get_model_based_importance_layout(),
    'regularization_selection': _Selection_Methods.get_regularization_selection_layout(),
    'boruta_selection': _Selection_Methods.get_boruta_selection_layout(),
    'sequencial_forward_selection': _Selection_Methods.get_sequencial_forward_selection_layout(),
}
