from dash import Input, Output, State, html, dcc

from .sub_components.dt_conf_input import dt_upload_layout, dt_form, dt_data_info
from .sub_components.fs_conf_input import fs_upload_layout, fs_form, fs_data_info

from src.dash_utils.utils import UtilityTools
from src.dash_utils.config_input import dt_conf, ml_config, fs_conf
from src.data_transform import DataTransform
from src.feature_selection import FeatureSelection

from server import app
from hashlib import sha256

import json

import dash_bootstrap_components as dbc
import pandas as pd
import datetime as dt


data_transform_layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [dbc.Label(['Upload Raw Data File:']), dt_upload_layout], width=4),
                                dbc.Col([
                                    html.Div(
                                        [
                                            html.Span(['File Name: ', 'NA'],
                                                      id='dt-uploaded-file-name'),
                                            html.Br(),
                                            html.Span(
                                                ['Date Uploaded: ', 'NA'])
                                        ], id='dt-uploaded-file-meta-data', style={'margin-left': '10px'}
                                    )
                                ], width=8, style={'border-left': '1px solid black'}),
                            ]
                        ),

                        dbc.Row([dt_form])
                    ], width=6
                ),

                dbc.Col([], width=1),

                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Button(
                                    [
                                        html.I(
                                            className='fa-regular fa-file-code'),
                                        html.Span(
                                            [' Show Current Configuration (JSON)'])
                                    ], id='dt-show-current-conf-btn', color='primary', style={
                                        'margin-top': '15px', 'margin-bottom': '15px', 'width': '100%', 'fontSize': '0.8rem', 'height': 'auto'
                                    }
                                ),

                                dbc.Modal(
                                    [
                                        dbc.ModalHeader(dbc.ModalTitle(
                                            "Modal with scrollable body")),
                                        dbc.ModalBody(
                                            id='dt-show-current-conf'),
                                        dbc.ModalFooter(
                                            dbc.Button(
                                                "Close",
                                                id="close-body-scroll",
                                                className="ms-auto",
                                                n_clicks=0,
                                            )
                                        ),
                                    ],
                                    id="modal-body-scroll",
                                    scrollable=True,
                                    is_open=False,
                                ),
                            ]
                        ),

                        dbc.Row(
                            [
                                dbc.Col([dt_data_info])
                            ], style={'overflow-y': 'scroll', 'height': 'auto'}
                        ),
                    ], width=5
                ),
            ]
        )
    ]
)

feature_selection_layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [dbc.Label(['Upload Transformed Data File:']), fs_upload_layout], width=5),
                                dbc.Col([
                                    html.Div(
                                        [
                                            html.Span(['File Name: ', 'NA'],
                                                      id='fs-uploaded-file-name'),
                                            html.Br(),
                                            html.Span(
                                                ['Date Uploaded: ', 'NA'])
                                        ], id='fs-uploaded-file-meta-data', style={'margin-left': '10px'}
                                    )
                                ], width=7, style={'border-left': '1px solid black'}),
                            ]
                        ),

                        dbc.Row([fs_form])
                    ], width=6
                ),

                dbc.Col([], width=1),

                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Button(
                                    [
                                        html.I(
                                            className='fa-regular fa-file-code'),
                                        html.Span(
                                            [' Show Current Configuration (JSON)'])
                                    ], id='fs-show-current-conf-btn', color='primary', style={
                                        'margin-top': '15px', 'margin-bottom': '15px', 'width': '100%', 'fontSize': '0.8rem', 'height': 'auto'
                                    }
                                ),

                                dbc.Button(
                                    [
                                        html.I(
                                            className='fa-regular fa-file-code'),
                                        html.Span(
                                            [' Show Selected Features (JSON)'])
                                    ], id='fs-show-selected_feat-btn', color='primary', disabled=True, style={
                                        'margin-top': '15px', 'margin-bottom': '15px', 'width': '100%', 'fontSize': '0.8rem', 'height': 'auto'
                                    }
                                ),

                                dbc.Modal(
                                    [
                                        dbc.ModalHeader(dbc.ModalTitle(
                                            "Modal with scrollable body")),
                                        dbc.ModalBody(
                                            id='fs-show-current-conf'),
                                        dbc.ModalFooter(
                                            dbc.Button(
                                                "Close",
                                                id="fs-close-body-scroll",
                                                className="ms-auto",
                                                n_clicks=0,
                                            )
                                        ),
                                    ],
                                    id="fs-modal-body-scroll",
                                    scrollable=True,
                                    is_open=False,
                                ),

                                dbc.Modal(
                                    [
                                        dbc.ModalHeader(dbc.ModalTitle(
                                            "Selected Features")),
                                        dbc.ModalBody(
                                            id='fs-show-selected-features-json'),
                                        dbc.ModalFooter(
                                            dbc.Button(
                                                "Close",
                                                id="fs-close-select-feat-modal",
                                                className="ms-auto",
                                                n_clicks=0,
                                            )
                                        ),
                                    ],
                                    id="fs-show-select-feat-modal",
                                    scrollable=True,
                                    is_open=False,
                                ),
                            ]
                        ),

                        dbc.Row(
                            [
                                dbc.Col([fs_data_info])
                            ], style={'overflow-y': 'scroll', 'height': 'auto'}
                        ),
                    ], width=5
                ),
            ]
        )
    ]
)

accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                children=[
                    data_transform_layout], title=f"Data Transformation Details, Status - {ml_config.dt_initial_status}", id='dt-accordion-title'
            ),
            dbc.AccordionItem(
                children=[
                    feature_selection_layout], title=f"Feature Selection Details, Status - {ml_config.dt_initial_status}", id='fs-accordion-title'
            ),
            dbc.AccordionItem(
                children=[], title=f"Baseline Modeling Details, Status - {ml_config.dt_initial_status}", id='bm-accordion-title'
            ),
            dbc.AccordionItem(
                children=[], title=f"Hyper-Parameter Tuning Details, Status - {ml_config.dt_initial_status}", id='hpt-accordion-title'
            ),
        ],
        flush=True,
        start_collapsed=True,
        always_open=True
    ),
)

classification_heading_jumbotron = html.Div(
    dbc.Container(
        [
            html.H2("Classification XL", className="display-3"),
            html.P(
                "Sub-Modules: Data Transformation, Feature Selection, Baseline Modeling, Hyper-Parameter Tuning",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(
                "Classification is defined as admitting, understanding, and grouping objects. It classifies the "
                "datasets into pre-set classes. It is perhaps performed on both structured and unstructured data. "
                "The process starts with anticipating the class of given points. In classification machine "
                "learning, algorithms work to classify the data and datasets into respective and relevant categories."
            )
        ],
        fluid=True,
        className="py-3",
    ),
    className="p-3 bg-light rounded-3",
)


Classification_Layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([classification_heading_jumbotron])
                ]),
                dbc.Row([
                    dbc.Col(
                        [accordion], width=12)
                ], style={'margin-top': '32px'})
            ])
        ])
    ])
])

# --------------------------------------------- ## CALLBACK METHODS ## --------------------------------------------- #
# --------------------------------------------- ## DATA TRANSFORMATION CALLBACK ## --------------------------------------------- #


@app.callback(
    Output('dt-uploaded-file-meta-data', 'children'),
    [
        Input('dt-raw-data-upload', 'contents')
    ],
    [
        State('dt-raw-data-upload', 'filename'),
    ]
)
def dt_upload_raw_data(file_content, file_name):
    if file_content is not None:
        return UtilityTools.parse_contents(file_content, file_name, 'dt')

    default_response_page = html.Div(
        [
            html.Span(['File Name: ', 'NA'],
                      id='dt-uploaded-file-name'),
            html.Br(),
            html.Span(['Date Uploaded: ', 'NA'])
        ], id='dt-uploaded-file-meta-data', style={'margin-top': '10px'}
    )

    return default_response_page


@app.callback(
    Output('dt-data-file-contents', 'children'),
    [
        Input('dt-uploaded-file-name', 'children')
    ]
)
def dt_show_feature_info_list(file_name_children):
    file_name = file_name_children[-1]

    if not (file_name == 'NA'):
        raw_data_uploaded = pd.read_csv(f'./data/raw/{file_name}')
        dt_conf.raw_data = raw_data_uploaded.copy()

        dt_conf.dt_conf_input['file_path'] = f'./data/raw/{file_name}'
        dt_conf.dt_conf_input['file_type'] = file_name.split('.')[-1]

        data_dtypes = raw_data_uploaded.dtypes
        data_dtypes = pd.DataFrame(
            {'Feature Name': data_dtypes.index, 'Feature Data Type': list(map(str, data_dtypes.values))})

        feat_info_table = dbc.Table.from_dataframe(
            data_dtypes, striped=True, bordered=True, hover=True)

        return feat_info_table

    return html.Div([html.Span([html.Em(['None File Uploaded!'])])])


@app.callback(
    [
        Output('dt-feature-list-target', 'options'),
        Output('dt-feature-list-categorical', 'options'),
        Output('drop_redundent_columns_params', 'options')
    ],
    [
        Input('dt-uploaded-file-name', 'children')
    ]
)
def dt_show_feature_list_target(file_name_children):
    if not (file_name_children[-1] == 'NA'):
        feature_list_options = [{'label': col, 'value': col}
                                for col in list(dt_conf.raw_data.columns)]

        return (feature_list_options, feature_list_options, feature_list_options)

    return ([], [], [])


@app.callback(
    [
        Output('dt-show-current-conf', 'children'),
        Output('dt-accordion-title', 'title')
    ],
    [
        Input('dt-save-conf-btn', 'n_clicks'),
        Input('dt-feature-list-target', 'value'),
        Input('dt-feature-list-categorical', 'value'),

        Input('drop_redundent_columns', 'value'),
        Input('drop_redundent_columns_params', 'value'),

        Input('drop_null_columns', 'value'),
        Input('drop_null_columns_params', 'value'),

        Input('drop_unique_value_columns', 'value'),
        Input('drop_unique_value_columns_params', 'value'),

        Input('data_imputation', 'value'),
        Input('data_imputation_params', 'value'),

        Input('feature_scaling', 'value'),
        Input('feature_scaling_params', 'value'),

        Input('feature_transformer', 'value'),
        Input('feature_transformer_params', 'value'),

        Input('dt-outlier-removal-checkbox', 'value'),
        Input('dt-contamination-factor-input', 'value')
    ]
)
def dt_update_config(
    save_btn, target_value, cat_feat_list, drc_bool, drop_redundent_columns_params, dnc_bool, drop_null_columns_params, duvc_bool,
    drop_unique_value_columns_params, di_bool, data_imputation_params, fs_bool, feature_scaling_params,
    ft_bool, feature_transformer_params, outlier_removal_bool, contamination_factor
):
    if save_btn:
        methods_params_dict = {
            'drop_redundent_columns': drc_bool,
            'drop_null_columns': dnc_bool,
            'drop_unique_value_columns': duvc_bool,
            'data_imputation': di_bool,
            'feature_scaling': fs_bool,
            'feature_transformer': ft_bool,
        }

        params_dict = {
            'drop_redundent_columns': {
                'custom_columns_list': drop_redundent_columns_params,
            },
            'drop_null_columns': {
                'null_percent_threshold': drop_null_columns_params
            },
            'drop_unique_value_columns': {
                'unique_value_threshold': drop_unique_value_columns_params
            },
            'data_imputation': {
                'imputation_method': data_imputation_params
            },
            'feature_scaling': {
                'scaling_method': feature_scaling_params
            },
            'feature_transformer': {
                'transforming_method': feature_transformer_params
            },
        }

        dt_conf.dt_conf_input['target_feature'] = target_value
        dt_conf.dt_conf_input['feature_list'] = cat_feat_list

        dt_conf.dt_conf_input['remove_outlier'] = outlier_removal_bool
        dt_conf.dt_conf_input['contamination_factor'] = contamination_factor

        for method_name in dt_conf.transform_method:
            if methods_params_dict[method_name]:
                dt_conf.dt_conf_input['transform_conf'].append({
                    'method_name': f'{method_name}',
                    'params': params_dict[method_name]
                })

        _unique_serial_number = sha256(
            str(ml_config.dt_serial_number).encode('utf-8')).hexdigest()

        ml_config.dt_serial_number += 1
        ml_config.dt_current_version += 0.1
        ml_config.dt_initial_status = 'Configuration Saved'

        ml_config.status.loc[len(ml_config.status.index)] = [_unique_serial_number, dt.datetime.now(), ml_config.dt_current_version,
                                                             'Data Transformation', json.dumps(dt_conf.dt_conf_input), ml_config.dt_initial_status]

    current_status = f"Data Transformation Details, Status - {ml_config.dt_initial_status}"

    json_output = html.Div(
        [
            html.Pre(children=["JSON Config: {}".format(
                json.dumps(dt_conf.dt_conf_input, indent=4))])
        ]
    )

    return [json_output, current_status]


@app.callback(
    Output("dt-save-conf-file-success", "is_open"),
    [
        Input("dt-save-conf-btn", "n_clicks")
    ],
    [
        State("dt-save-conf-file-success", "is_open")
    ],
)
def toggle_conf_save_alert(save_conf_bth, save_conf_alert_state):
    if save_conf_bth:
        return not save_conf_alert_state

    return save_conf_alert_state


@app.callback(
    Output("dt-run-started", "is_open"),
    [
        Input("run-data-transform", "n_clicks")
    ],
    [
        State("dt-run-started", "is_open")
    ],
)
def toggle_dt_run_started_alert(run_transform_btn, run_status_alert_state):
    if run_transform_btn:
        return not run_status_alert_state
    return run_status_alert_state


@app.callback(
    [
        Output('drop_unique_value_columns_params', 'valid'),
        Output('drop_unique_value_columns_params', 'invalid')
    ],
    [
        Input('drop_unique_value_columns_params', 'value')
    ]
)
def valid_dt_duvc_threshold(drop_unique_value_columns_params):
    if isinstance(drop_unique_value_columns_params, (int, float)):
        if isinstance(drop_unique_value_columns_params, float) and drop_unique_value_columns_params <= 1:
            return True, False

        elif isinstance(drop_unique_value_columns_params, int) and drop_unique_value_columns_params >= 1:
            return True, False

    return False, True


@app.callback(
    Output("modal-body-scroll", "is_open"),
    [
        Input("dt-show-current-conf-btn", "n_clicks"),
        Input("close-body-scroll", "n_clicks"),
    ],
    [
        State("modal-body-scroll", "is_open")
    ],
)
def toggle_show_dt_conf_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('dt-contamination-factor-input', 'disabled'),
    [
        Input('dt-outlier-removal-checkbox', 'value')
    ]
)
def toggle_outlier_dropping(value):
    return not value


@app.callback(
    Output("dt-run-complete", "is_open"),
    [
        Input("run-data-transform", "n_clicks"),
    ],
    [
        State("dt-run-complete", "is_open")
    ],
)
def toggle_dt_run_started_alert(run_transform_btn, run_status_alert_state):
    if run_transform_btn:
        transformed_data = DataTransform().compile_pipeline(**dt_conf.dt_conf_input)
        transformed_data.to_csv(
            f'./data/processed/transformed_data_v{ml_config.dt_current_version}.csv', index=False)

        _unique_serial_number = sha256(
            str(ml_config.dt_serial_number).encode('utf-8')).hexdigest()

        ml_config.dt_serial_number += 1
        ml_config.dt_current_version += 0.1
        ml_config.dt_initial_status = 'Run Completed Successfully'

        ml_config.status.loc[len(ml_config.status.index)] = [_unique_serial_number, dt.datetime.now(), ml_config.dt_current_version,
                                                             'Data Transformation', json.dumps(dt_conf.dt_conf_input), ml_config.dt_initial_status]

        print(ml_config.status)

        return not run_status_alert_state

    return run_status_alert_state


@app.callback(
    Output("dt-download", "data"),
    [
        Input("download-dt-data", "n_clicks")
    ]
)
def download_dt_data(download_btn):
    if download_btn:
        df = pd.read_csv('./data/processed/transformed_data.csv')
        return dcc.send_data_frame(df.to_csv, filename=f"transformed_data_v{ml_config.dt_serial_number}.csv", index=False)


# --------------------------------------------- ## FEATURE SELECTION CALLBACK ## --------------------------------------------- #

@app.callback(
    Output('fs-uploaded-file-meta-data', 'children'),
    [
        Input('fs-raw-data-upload', 'contents')
    ],
    [
        State('fs-raw-data-upload', 'filename'),
    ]
)
def fs_upload_raw_data(file_content, file_name):
    if file_content is not None:
        return UtilityTools.parse_contents(file_content, file_name, 'fs')

    default_response_page = html.Div(
        [
            html.Span(['File Name: ', 'NA'],
                      id='fs-uploaded-file-name'),
            html.Br(),
            html.Span(['Date Uploaded: ', 'NA'])
        ], id='fs-uploaded-file-meta-data', style={'margin-top': '10px'}
    )

    return default_response_page


@app.callback(
    Output('fs-data-file-contents', 'children'),
    [
        Input('fs-uploaded-file-name', 'children')
    ]
)
def fs_show_feature_info_list(file_name_children):
    file_name = file_name_children[-1]

    if not (file_name == 'NA'):
        raw_data_uploaded = pd.read_csv(f'./data/raw/{file_name}')
        fs_conf.transformed_data = raw_data_uploaded.copy()

        fs_conf.fs_conf_input['file_path'] = f'./data/raw/{file_name}'
        fs_conf.fs_conf_input['file_type'] = file_name.split('.')[-1]

        data_dtypes = raw_data_uploaded.dtypes
        data_dtypes = pd.DataFrame(
            {'Feature Name': data_dtypes.index, 'Feature Data Type': list(map(str, data_dtypes.values))})

        feat_info_table = dbc.Table.from_dataframe(
            data_dtypes, striped=True, bordered=True, hover=True, id='fs-feature-list-table')

        return feat_info_table

    return html.Div([html.Span([html.Em(['None File Uploaded!'])])])


@app.callback(
    Output('fs-feature-list-target', 'options'),
    [
        Input('fs-uploaded-file-name', 'children')
    ]
)
def fs_show_feature_list_target(file_name_children):
    if not (file_name_children[-1] == 'NA'):
        feature_list_options = [{'label': col, 'value': col}
                                for col in list(fs_conf.transformed_data.columns)]

        return feature_list_options

    return []


@app.callback(
    Output('fs-variance-thresh-input', 'disabled'),
    [
        Input('fs-remove-low-variance-feat-checkbox', 'value')
    ]
)
def toggle_variance_feat_dropping(value):
    return not value


@app.callback(
    [
        Output('fs-corr-thresh-input', 'disabled'),
        Output('fs-corr-method-list', 'disabled')
    ],
    [
        Input('fs-remove-high-corr-feat-checkbox', 'value')
    ]
)
def toggle_corr_feat_dropping(value):
    return (not value, not value)


@app.callback(
    [
        Output('fs-show-current-conf', 'children'),
        Output('fs-accordion-title', 'title')
    ],
    [
        Input('fs-save-conf-btn', 'n_clicks'),
        Input('fs-feature-list-target', 'value'),
        Input('fs-test-size-input', 'value'),

        Input('fs-remove-low-variance-feat-checkbox', 'value'),
        Input('fs-variance-thresh-input', 'value'),

        Input('fs-remove-multi-collinear-feat-checkbox', 'value'),
        Input('fs-remove-high-corr-feat-checkbox', 'value'),
        Input('fs-corr-thresh-input', 'value'),
        Input('fs-corr-method-list', 'value'),

        Input('fs-run-method-radio-select', 'value'),

        Input('anova_f_value_selection', 'value'),
        Input('anova_f_value_selection_params', 'value'),

        Input('mutual_info_classif_selection', 'value'),
        Input('mutual_info_classif_selection_params', 'value'),

        Input('logit_selection', 'value'),
        Input('logit_selection_params', 'value'),

        Input('permutation_impt_selection', 'value'),
        Input('permutation_impt_selection_params', 'value'),
        Input('permutation_impt_selection_feat_num_params', 'value'),

        Input('recursive_feature_elimination', 'value'),
        Input('recursive_feature_elimination_params', 'value'),
        Input('recursive_feature_elimination_feat_num_params', 'value'),
        Input('recursive_feature_elimination_step_value_params', 'value'),

        Input('model_based_importance', 'value'),
        Input('model_based_importance_params', 'value'),
        Input('model_based_importance_feat_num_params', 'value'),

        Input('regularization_selection', 'value'),
        Input('regularization_selection_params', 'value'),
        Input('regularization_selection_feat_num_params', 'value'),

        Input('boruta_selection', 'value'),
        Input('boruta_selection_params', 'value'),

        Input('sequencial_forward_selection', 'value'),
        Input('sequencial_forward_selection_params', 'value'),
        Input('sequencial_forward_selection_feat_num_params', 'value'),
        Input('sequencial_forward_selection_scoring_mertric_params', 'value'),
    ]
)
def fs_update_config(
    save_btn, target_value, test_size, variance_bool, variance_thresh, multi_corr_bool, corr_bool, corr_thresh, corr_method, run_parallel_bool,
    anova_f_bool, anova_f_params, mic_bool, mic_params, logit_bool, logit_params, perm_bool, perm_model, perm_feat_num, rfe_bool, rfe_model,
    rfe_feat_num, rfe_step, mbi_bool, mbo_model, mbi_feat_num, rs_bool, ri_model, ri_feat_num, boruta_bool, boruta_model, sfs_bool, sfs_model,
    sfs_feat_num, sfs_metric
):
    if save_btn:
        methods_params_dict = {
            'anova_f_value_selection': anova_f_bool,
            'mutual_info_classif_selection': mic_bool,
            'logit_selection': logit_bool,
            'permutation_impt_selection': perm_bool,
            'recursive_feature_elimination': rfe_bool,
            'model_based_importance': mbi_bool,
            'regularization_selection': rs_bool,
            'boruta_selection': boruta_bool,
            'sequencial_forward_selection': sfs_bool,
        }

        params_dict = {
            'anova_f_value_selection': {
                'num_feat': anova_f_params
            },
            'mutual_info_classif_selection': {
                'num_feat': mic_params
            },
            'logit_selection': {
                'fit_method': logit_params if logit_params != 'default' else None
            },
            'permutation_impt_selection': {
                'model_list': perm_model if isinstance(perm_model, list) else [perm_model],
                'num_feat': perm_feat_num
            },
            'recursive_feature_elimination': {
                'model_list': rfe_model if isinstance(rfe_model, list) else [rfe_model],
                'num_feat': rfe_feat_num,
                'step_value': rfe_step if rfe_step != 0 else None
            },
            'model_based_importance': {
                'model_list': mbo_model if isinstance(mbo_model, list) else [mbo_model],
                'num_feat': mbi_feat_num
            },
            'regularization_selection': {
                'model_list': ri_model if isinstance(ri_model, list) else [ri_model],
                'num_feat': ri_feat_num
            },
            'boruta_selection': {
                'model_list': boruta_model if isinstance(boruta_model, list) else [boruta_model]
            },
            'sequencial_forward_selection': {
                'model_list': sfs_model if isinstance(sfs_model, list) else [sfs_model],
                'num_feat': sfs_feat_num,
                'scoring_metric': sfs_metric
            },
        }

        fs_conf.fs_conf_input['target_feature'] = target_value
        fs_conf.fs_conf_input['test_size'] = test_size

        fs_conf.fs_conf_input['drop_low_variance_features'] = variance_bool
        fs_conf.fs_conf_input['variance_thresh'] = variance_thresh

        fs_conf.fs_conf_input['drop_multicolliner_features'] = multi_corr_bool
        fs_conf.fs_conf_input['drop_high_corr_features'] = corr_bool
        fs_conf.fs_conf_input['corr_threshold'] = corr_thresh
        fs_conf.fs_conf_input['corr_method'] = corr_method

        fs_conf.fs_conf_input['run_parallel'] = bool(run_parallel_bool)

        for method_name in fs_conf.selection_methods:
            if methods_params_dict[method_name]:
                fs_conf.fs_conf_input['feature_select_conf'].append({
                    'select_method': f'{method_name}',
                    'params': params_dict[method_name]
                })

    current_status = f"Feature Selection Details, Status - {ml_config.fs_initial_status}"

    json_output = html.Div(
        [
            html.Pre(children=["JSON Config: {}".format(
                json.dumps(fs_conf.fs_conf_input, indent=4))])
        ]
    )

    return [json_output, current_status]


@app.callback(
    Output("fs-save-conf-file-success", "is_open"),
    [
        Input("fs-save-conf-btn", "n_clicks")
    ],
    [
        State("fs-save-conf-file-success", "is_open")
    ],
)
def toggle_conf_save_alert(save_conf_bth, save_conf_alert_state):
    if save_conf_bth:
        return not save_conf_alert_state

    return save_conf_alert_state


@app.callback(
    Output("fs-modal-body-scroll", "is_open"),
    [
        Input("fs-show-current-conf-btn", "n_clicks"),
        Input("fs-close-body-scroll", "n_clicks"),
    ],
    [
        State("fs-modal-body-scroll", "is_open")
    ],
)
def toggle_show_fs_conf_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [
        Output("fs-run-complete", "is_open"),
        Output('fs-show-selected_feat-btn', 'disabled'),
        Output('fs-show-selected-features-json', 'children')
    ],
    [
        Input("run-feature-selection", "n_clicks"),
    ],
    [
        State("fs-run-complete", "is_open")
    ],
)
def toggle_fs_run_started_alert(run_selection_btn, run_status_alert_state):
    if run_selection_btn:
        selected_features = FeatureSelection().compile_selection(**fs_conf.fs_conf_input)

        _unique_serial_number = sha256(
            str(ml_config.dt_serial_number).encode('utf-8')).hexdigest()

        ml_config.dt_serial_number += 1
        ml_config.dt_current_version += 0.1
        ml_config.dt_initial_status = 'Run Completed Successfully'

        ml_config.status.loc[len(ml_config.status.index)] = [_unique_serial_number, dt.datetime.now(), ml_config.dt_current_version,
                                                             'Data Transformation', json.dumps(dt_conf.dt_conf_input), ml_config.dt_initial_status]

        print(ml_config.status)

        json_output = html.Div(
            [
                html.Pre(children=["JSON Config: {}".format(
                    json.dumps(selected_features, indent=4))])
            ]
        )

        return [not run_status_alert_state, False, json_output]

    return [run_status_alert_state, True, html.Div()]


@app.callback(
    Output("fs-show-select-feat-modal", "is_open"),
    [
        Input("fs-show-selected_feat-btn", "n_clicks"),
        Input("fs-close-select-feat-modal", "n_clicks"),
    ],
    [
        State("fs-show-select-feat-modal", "is_open")
    ],
)
def toggle_show_fs_conf_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
