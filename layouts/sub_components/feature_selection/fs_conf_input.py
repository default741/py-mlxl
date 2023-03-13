from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from .fs_method_layout_mapping import method_layout_mapping
from src.dash_utils.config_input import fs_conf, ml_config

from src.dash_utils.utils import UtilityTools

from server import app

import json

import datetime as dt
import pandas as pd


fs_upload_layout = html.Div(
    [
        dcc.Upload(
            [
                dbc.Button(
                    [
                        html.I(className='fa-solid fa-upload'),
                        html.Span([" Upload"])
                    ], id="fs-open-upload-data", style={'width': '100%'}
                ),
            ], id='fs-raw-data-upload', multiple=False
        ),
    ]
)

fs_target_feature_input = html.Div(
    [
        html.Hr(),
        dbc.Label("Select Dependent Variable (Target):"),
        dcc.Dropdown(id='fs-feature-list-target', multi=False,
                     placeholder='Select Feature'),

        dbc.Label("Add Test Size (For Train-Test Split):",
                  style={'margin-top': '5px'}),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Test Data Size:"),
                dbc.Input(type="number", placeholder=1,
                          value=0.2, id='fs-test-size-input', max=1, min=0,
                          step=0.1),
                dbc.InputGroupText("% (Percent)"),
            ], className="mb-3",
        ),
    ], style={'margin-top': '5px'}
)

remove_low_variance_layout = html.Div(
    [
        html.Hr(),
        dbc.Label(['Low Variance Features:']),
        dbc.Checkbox(
            label='Remove Low Variance Features from Data (Using Variance Threshold)',
            value=False,
            id="fs-remove-low-variance-feat-checkbox",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Variance Threshold:"),
                dbc.Input(type="number", placeholder=1,
                          value=0.1, id='fs-variance-thresh-input', max=1, min=0,
                          step=0.1, disabled=True),
                dbc.InputGroupText("% (Percent)"),
            ], className="mb-3",
        ),
    ], style={'margin-top': '10px'}
)

remove_high_corr_layout = html.Div(
    [
        html.Hr(),
        dbc.Label(['Highly Correlated Features:']),
        dbc.Checkbox(
            label='Remove Multi-Collinear Features from Data (Stand-Alone)',
            value=False,
            id="fs-remove-multi-collinear-feat-checkbox",
        ),
        dbc.Checkbox(
            label='Remove Highly Correlated Features from Data',
            value=False,
            id="fs-remove-high-corr-feat-checkbox",
        ),

        dbc.InputGroup(
            [
                dbc.InputGroupText("Correlation Threshold:"),
                dbc.Input(type="number", placeholder=1,
                          value=0.1, id='fs-corr-thresh-input', max=1, min=0,
                          step=0.1, disabled=True),
                dbc.InputGroupText("% (Percent)"),
            ], className="mb-3",
        ),

        dbc.Label("Select Correlation Method:"),
        dcc.Dropdown(
            id='fs-corr-method-list',
            multi=False,
            value=list(fs_conf.correlation_methods)[0],
            options=[
                {'label': method.capitalize(), 'value': method} for method in list(fs_conf.correlation_methods)
            ],
            placeholder='Select Method',
            disabled=True
        ),
    ], style={'margin-top': '10px'}
)

run_parallel_layout = html.Div(
    [
        html.Hr(),
        dbc.Label("Select One Execution Method"),
        dbc.RadioItems(
            options=[
                {"label": "Run Parallel (Fast Run, Uses Multiple Cores)",
                 "value": 1},
                {"label": "Run Sequencially (Slow Run, Uses One Core)",
                 "value": 0},
            ],
            value=1,
            id="fs-run-method-radio-select",
        ),
    ]
)

fs_method_acc_item = [
    dbc.AccordionItem(
        [
            html.Div(
                [
                    dbc.Checkbox(
                        id=f'{method}',
                        label="Add this Selection Method to Pipeline!",
                        value=False,
                    ),

                    method_layout_mapping[method]
                ]
            )
        ], title=f"{idx + 1}. {' '.join(list(map(lambda x: x.capitalize(), method.split('_'))))}",
    )
    for idx, method in enumerate(fs_conf.selection_methods)
]

fs_methods_select_accordion = html.Div(
    [
        html.Hr(),
        dbc.Label("Select Transformation Methods to Apply:"),
        dbc.Accordion(children=fs_method_acc_item,
                      start_collapsed=True, flush=False),
    ], style={'margin-top': '10px'}
)

fs_btn_group = html.Div(
    [
        html.Hr(),

        dbc.Row(
            [
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-floppy-disk'),
                            html.Span([' Save Configuration'])
                        ], color='primary', id='fs-save-conf-btn', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.8rem'}
                    )
                ], width=4),

                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-play'),
                            html.Span([' Run Feature Selection'])
                        ], id='run-feature-selection', color='primary', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.8rem'}
                    )
                ], width=4),

                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-download'),
                            html.Span([' Download Selection Zip'])
                        ], id='download-fs-data', color='primary', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.6rem'}
                    ),
                    dcc.Download(id="fs-download")
                ], width=4),
            ]
        )
    ]
)

alert_tab_layout = html.Div(
    [
        dbc.Alert(
            "Configuration Save Successfully!",
            id="fs-save-conf-file-success", is_open=False, duration=2000,
        ),

        dbc.Alert(
            "Data Transformation Pipeline Starting to Execute!",
            id="fs-run-started", is_open=False, duration=1000, color="info"
        ),

        dcc.Loading(
            children=[
                dbc.Alert(
                    "Data Transformation Pipeline Run Complete!",
                    id="fs-run-complete", is_open=False, duration=2000,
                ),
            ], id="fs-ls-loading-2", type="cube", fullscreen=True, style={'margin-top': '5px', 'margin-botttom': '5px'}
        ),


    ],
    id='dt-show-alert', style={'margin-top': '5px', 'margin-bottom': '5px'}
)

fs_form = html.Div(
    [fs_target_feature_input, remove_low_variance_layout, remove_high_corr_layout, run_parallel_layout,
     fs_methods_select_accordion, alert_tab_layout, fs_btn_group])

fs_data_info = html.Div([
    html.Div([html.Span([html.Em(['None File Uploaded!'])])])
], id='fs-data-file-contents', style={'margin-top': '10px'})


# --------------------------------------------- ## FEATURE SELECTION CALLBACK ## --------------------------------------------- #


@app.callback(
    Output('fs-uploaded-file-meta-data', 'children'),
    [
        Input('fs-raw-data-upload', 'contents'),
        Input('use-dt-final-dataset', 'value')
    ],
    [
        State('fs-raw-data-upload', 'filename'),
    ]
)
def fs_upload_raw_data(file_content, use_dt_data_bool, file_name):
    if use_dt_data_bool:
        date_uploaded = dt.datetime.now().strftime('%d %B, %Y %H:%M:%S')
        previous_file = f'transformed_data_v{ml_config.dt_current_version - 0.1}.csv'

        pd.read_csv(f'./data/processed/{previous_file}').to_csv(
            f'./data/raw/{previous_file}', index=False)

        return html.Div([
            html.Span(['File Name: ', previous_file],
                      id=f'fs-uploaded-file-name'),
            html.Br(),
            html.Span(
                ['Date Uploaded: ', date_uploaded])
        ])

    if file_content is not None:
        return UtilityTools.parse_contents(file_content, file_name, 'fs')

    default_response_page = html.Div([
        html.Span(['File Name: ', 'NA'],
                  id='fs-uploaded-file-name'),
        html.Br(),
        html.Span(['Date Uploaded: ', 'NA'])
    ])

    return default_response_page


@app.callback(
    [
        Output('fs-data-file-contents', 'children'),
        Output('fs-feature-list-target', 'options')
    ],
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

        feature_list_options = [{'label': col, 'value': col}
                                for col in list(fs_conf.transformed_data.columns)]

        return (feat_info_table, feature_list_options)

    return (html.Div([html.Span([html.Em(['None File Uploaded!'])])]), [])


@app.callback(
    Output('fs-variance-thresh-input', 'disabled'),
    [
        Input('fs-remove-low-variance-feat-checkbox', 'value')
    ]
)
def fs_toggle_variance_feat_dropping(value):
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
def fs_toggle_corr_feat_dropping(value):
    return (not value, not value)


@app.callback(
    Output("fs-save-conf-file-success", "is_open"),
    [
        Input("fs-save-conf-btn", "n_clicks")
    ],
    [
        State("fs-save-conf-file-success", "is_open")
    ],
)
def fs_toggle_conf_save_alert(save_conf_bth, save_conf_alert_state):
    if save_conf_bth:
        return not save_conf_alert_state

    return save_conf_alert_state


@app.callback(
    [
        Output('fs-show-current-conf', 'children'),
        Output('fs-accordion-title', 'title'),
        Output('fs-reset-current-conf-modal', 'n_clicks')
    ],
    [
        Input('fs-save-conf-btn', 'n_clicks'),
        Input('fs-feature-list-target', 'value'),
        Input('fs-test-size-input', 'value'),
        Input('fs-reset-current-conf-modal', 'n_clicks'),

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
    save_btn, target_value, test_size, reset_conf_btn, variance_bool, variance_thresh, multi_corr_bool, corr_bool, corr_thresh, corr_method, run_parallel_bool,
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

    if reset_conf_btn:
        fs_conf.refresh_config()
        reset_conf_btn = 0

    current_status = f"Feature Selection Details, Status - {ml_config.fs_initial_status}"

    json_output = html.Div(
        [
            html.Pre(children=["JSON Config: {}".format(
                json.dumps(fs_conf.fs_conf_input, indent=4))])
        ]
    )

    return [json_output, current_status, reset_conf_btn]
