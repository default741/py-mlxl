from dash import html, dcc, Input, State, Output
import dash_bootstrap_components as dbc

import pandas as pd
import datetime as dt

from .dt_method_layout_mapping import method_layout_mapping

from src.dash_utils.config_input import dt_conf, ml_config
from src.dash_utils.utils import UtilityTools

import json

from hashlib import sha256
from server import app


dt_upload_layout = html.Div(
    [
        dcc.Upload(
            [
                dbc.Button(
                    [
                        html.I(className='fa-solid fa-upload'),
                        html.Span([" Upload"])
                    ], id="dt-open-upload-data", style={'width': '100%'}
                ),
            ], id='dt-raw-data-upload', multiple=False
        ),
    ]
)

dt_target_feature_input = html.Div(
    [
        html.Hr(),
        dbc.Label("Select Dependent Variable (Target):"),
        dcc.Dropdown(id='dt-feature-list-target', multi=False,
                     placeholder='Select Feature'),
    ], style={'margin-top': '5px'}
)

categorical_feature_select = html.Div(
    [
        dbc.Label("Select Categorical Features:"),
        dcc.Dropdown(id='dt-feature-list-categorical', multi=True,
                     placeholder='Select Features'),
    ], style={'margin-top': '10px'}
)

remove_outliers_layout = html.Div(
    [
        html.Hr(),
        dbc.Label(['Outlier Removal:']),
        dbc.Checkbox(
            label='Remove Outliers from Data (Using Isolation Forest)',
            value=False,
            id="dt-outlier-removal-checkbox",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Contamination Factor:"),
                dbc.Input(type="number", placeholder=1,
                          value=0.1, id='dt-contamination-factor-input', max=1, min=0,
                          step=0.01, disabled=True),
                dbc.InputGroupText("(For Isolation Forest)"),
            ], className="mb-3",
        ),
    ], style={'margin-top': '10px'}
)

dt_method_acc_item = [
    dbc.AccordionItem(
        [
            html.Div(
                [
                    dbc.Checkbox(
                        id=f'{method}',
                        label="Add this Transformation Method to Pipeline!",
                        value=False,
                    ),

                    method_layout_mapping[method]
                ]
            )
        ], title=f"{idx + 1}. {' '.join(list(map(lambda x: x.capitalize(), method.split('_'))))}",
    )
    for idx, method in enumerate(dt_conf.transform_method)
]

dt_methods_select_accordion = html.Div(
    [
        html.Hr(),
        dbc.Label("Select Transformation Methods to Apply:"),
        dbc.Accordion(children=dt_method_acc_item,
                      start_collapsed=True, flush=False),
    ], style={'margin-top': '10px'}
)

dt_btn_group = html.Div(
    [
        html.Hr(),

        dbc.Row(
            [
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-floppy-disk'),
                            html.Span([' Save Configuration'])
                        ], color='primary', id='dt-save-conf-btn', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.6rem'}
                    )
                ], width=4),

                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-play'),
                            html.Span([' Run Transformation'])
                        ], id='run-data-transform', color='primary', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.6rem'}
                    )
                ], width=4),

                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-download'),
                            html.Span([' Download Transformation Zip'])
                        ], id='download-dt-data', color='primary', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.6rem'}
                    ),
                    dcc.Download(id="dt-download")
                ], width=4),
            ]
        )
    ]
)

alert_tab_layout = html.Div(
    [
        dbc.Alert(
            "Configuration Save Successfully!",
            id="dt-save-conf-file-success", is_open=False, duration=2000,
        ),

        dbc.Alert(
            "Data Transformation Pipeline Starting to Execute!",
            id="dt-run-started", is_open=False, duration=1000, color="info"
        ),

        dcc.Loading(
            children=[
                dbc.Alert(
                    "Data Transformation Pipeline Run Complete!",
                    id="dt-run-complete", is_open=False, duration=2000,
                ),
            ], id="ls-loading-2", type="cube", fullscreen=True, style={'margin-top': '5px', 'margin-botttom': '5px'}
        ),


    ],
    id='dt-show-alert', style={'margin-top': '5px', 'margin-bottom': '5px'}
)

dt_form = html.Div(
    [dt_target_feature_input, categorical_feature_select, remove_outliers_layout, dt_methods_select_accordion, alert_tab_layout, dt_btn_group])

dt_data_info = html.Div([
    html.Div([html.Span([html.Em(['None File Uploaded!'])])])
], id='dt-data-file-contents', style={'margin-top': '10px'})


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

    default_response_page = html.Div([
        html.Span(['File Name: ', 'NA'],
                  id='dt-uploaded-file-name'),
        html.Br(),
        html.Span(['Date Uploaded: ', 'NA'])
    ])

    return default_response_page


@app.callback(
    [
        Output('dt-data-file-contents', 'children'),

        Output('dt-feature-list-target', 'options'),
        Output('dt-feature-list-categorical', 'options'),
        Output('drop_redundent_columns_params', 'options')
    ],
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

        feature_list_options = [{'label': col, 'value': col}
                                for col in list(dt_conf.raw_data.columns)]

        return (feat_info_table, feature_list_options, feature_list_options, feature_list_options)

    return (html.Div([html.Span([html.Em(['None File Uploaded!'])])]), [], [], [])


@app.callback(
    Output('dt-contamination-factor-input', 'disabled'),
    [
        Input('dt-outlier-removal-checkbox', 'value')
    ]
)
def dt_toggle_outlier_dropping(value):
    return not value


@app.callback(
    Output("dt-save-conf-file-success", "is_open"),
    [
        Input("dt-save-conf-btn", "n_clicks")
    ],
    [
        State("dt-save-conf-file-success", "is_open")
    ],
)
def dt_toggle_conf_save_alert(save_conf_bth, save_conf_alert_state):
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
def dt_toggle_run_started_alert(run_transform_btn, run_status_alert_state):
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
def dt_valid_duvc_threshold(drop_unique_value_columns_params):
    if isinstance(drop_unique_value_columns_params, (int, float)):
        if isinstance(drop_unique_value_columns_params, float) and drop_unique_value_columns_params <= 1:
            return True, False

        elif isinstance(drop_unique_value_columns_params, int) and drop_unique_value_columns_params >= 1:
            return True, False

    return False, True


@app.callback(
    [
        Output('dt-show-current-conf', 'children'),
        Output('dt-accordion-title', 'title'),
        Output('dt-reset-current-conf-modal', 'n_clicks')
    ],
    [
        Input('dt-save-conf-btn', 'n_clicks'),
        Input('dt-feature-list-target', 'value'),
        Input('dt-feature-list-categorical', 'value'),
        Input('dt-reset-current-conf-modal', 'n_clicks'),

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
    save_btn, target_value, cat_feat_list, reset_btn_click, drc_bool, drop_redundent_columns_params, dnc_bool, drop_null_columns_params, duvc_bool,
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

    if reset_btn_click:
        dt_conf.refresh_config()
        reset_btn_click = 0

    current_status = f"Data Transformation Details, Status - {ml_config.dt_initial_status}"

    json_output = html.Div(
        [
            html.Pre(children=["JSON Config: {}".format(
                json.dumps(dt_conf.dt_conf_input, indent=4))])
        ]
    )

    return [json_output, current_status, reset_btn_click]
