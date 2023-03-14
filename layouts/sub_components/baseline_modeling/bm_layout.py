from dash import Input, Output, State, html, dcc

from .bm_conf_input import get_upload_button, bm_form

from src.dash_utils.utils import UtilityTools
from src.dash_utils.config_input import bm_conf, ml_config
from src.ml_scripts.baseline_modeling import BaselineModelling

from server import app
from hashlib import sha256
from zipfile import ZipFile

import json
import joblib

import dash_bootstrap_components as dbc
import pandas as pd
import datetime as dt


baseline_modeling_layout = html.Div(
    [
        dbc.Row([
            dbc.Col([
                dbc.Checkbox(
                    id='use-fs-final-data-file',
                    label="Use the Feature Selection Joblib file from the previous Step.",
                    value=False,
                ),
            ], width=12)
        ]),

        dbc.Row([
            dbc.Col(
                [html.Div([html.H4('OR')], style={'text-align': 'center'})], width=4)
        ]),

        dbc.Row([
            dbc.Col(
                [dbc.Label(['Upload Feature Selection Joblib Data File:'])], width=12),
        ]),

        dbc.Row([
            dbc.Col([
                get_upload_button(id_type='xtrain', value_str='X-Train'),

                html.Div(
                    [
                        html.Div([
                            html.Span(['File Name: ', 'NA'],
                                      id='bm-uploaded-file-name-xtrain'),
                            html.Br(),
                            html.Span(
                                ['Date Uploaded: ', 'NA'])
                        ])
                    ], id='bm-uploaded-file-meta-data-xtrain', style={'margin-top': '10px'}
                )
            ], width=4),

            dbc.Col([
                get_upload_button(id_type='ytrain', value_str='Y-Train'),

                html.Div(
                    [
                        html.Div([
                            html.Span(['File Name: ', 'NA'],
                                      id='bm-uploaded-file-name-ytrain'),
                            html.Br(),
                            html.Span(
                                ['Date Uploaded: ', 'NA'])
                        ])
                    ], id='bm-uploaded-file-meta-data-ytrain', style={'margin-top': '10px'}
                )
            ], width=4),

            dbc.Col([
                get_upload_button(id_type='select-feat',
                                  value_str='Features Selected'),

                html.Div(
                    [
                        html.Div([
                            html.Span(['File Name: ', 'NA'],
                                      id='bm-uploaded-file-name-select-feat'),
                            html.Br(),
                            html.Span(
                                ['Date Uploaded: ', 'NA'])
                        ])
                    ], id='bm-uploaded-file-meta-data-select-feat', style={'margin-top': '10px'}
                )
            ], width=4),
        ]),

        html.Hr(),

        dbc.Row([
            dbc.Col([bm_form], width=6),

            dbc.Col([], width=1),

            dbc.Col([
                dbc.Row([
                    dbc.Button(
                        [
                            html.I(
                                className='fa-regular fa-file-code'),
                            html.Span(
                                [' Show Current Configuration (JSON)'])
                        ], id='bm-show-current-conf-btn', color='primary', style={
                            'margin-top': '15px', 'margin-bottom': '15px', 'width': '100%', 'fontSize': '0.8rem', 'height': 'auto'
                        }
                    ),

                    dbc.Button(
                        [
                            html.I(
                                className='fa-regular fa-file-code'),
                            html.Span(
                                [' Show Baseline Modeling Results'])
                        ], id='bm-show-baseline-results-btn', color='primary', disabled=True, style={
                            'margin-top': '15px', 'margin-bottom': '15px', 'width': '100%', 'fontSize': '0.8rem', 'height': 'auto'
                        }
                    ),

                    dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle(
                                            "Feature Selection Configuration")),
                            dbc.ModalBody(
                                id='bm-show-current-conf'),
                            dbc.ModalFooter(
                                [
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Reset Conf.",
                                                id="bm-reset-current-conf-modal",
                                                className="ms-auto",
                                                n_clicks=0, style={'margin-right': '10px'}
                                            ),

                                            dbc.Button(
                                                "Close",
                                                id="bm-close-current-conf-modal",
                                                className="ms-auto",
                                                n_clicks=0,
                                            )
                                        ]
                                    )
                                ]
                            ),
                        ], id="bm-current-conf-modal-obj", scrollable=True, is_open=False,
                    ),

                    dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle(
                                            "Selected Features")),
                            dbc.ModalBody(
                                id='bm-show-baseline-results-json'),
                            dbc.ModalFooter(
                                dbc.Button(
                                    "Close",
                                    id="bm-close-baseline-results-modal",
                                    className="ms-auto",
                                    n_clicks=0,
                                )
                            ),
                        ],
                        id="bm-show-baseline-results-modal",
                        scrollable=True,
                        is_open=False,
                        fullscreen=True
                    ),
                ]),

                html.Hr(),

                dbc.Row([
                    dcc.Dropdown(
                        id='bm-feature-set-list', multi=False, placeholder='Select Feature Set', disabled=True
                    ),
                ]),

                dbc.Row([
                    html.Div([
                        html.Span(['No File Uploaded!'])
                    ], id='bm-show-feature-set-list')
                ], style={'margin-top': '10px'})
            ], width=5)
        ])
    ]
)


# --------------------------------------------- ## BASELINE MODELING CALLBACK ## --------------------------------------------- #


@app.callback(
    Output('bm-uploaded-file-meta-data-xtrain', 'children'),
    [
        Input('bm-raw-xtrain-data-upload', 'contents'),
        Input('use-fs-final-data-file', 'value')
    ],
    [
        State('bm-raw-xtrain-data-upload', 'filename'),
    ]
)
def bm_upload_raw_data_xtrain(file_content, use_fs_data_bool, file_name):
    if use_fs_data_bool:
        date_uploaded = dt.datetime.now().strftime('%d %B, %Y %H:%M:%S')
        previous_file = f'X_train_v{ml_config.fs_current_version - 0.1}.csv'

        X_train_data = joblib.load(
            './data/downloadables/feature_selection_files/feature_selected_meta.joblib')['meta_data']['X_train']

        X_train_data.to_csv(f'./data/raw/{previous_file}', index=False)
        bm_conf.bm_conf_input['file_path']['X_train'] = f'./data/raw/{previous_file}'

        return html.Div([
            html.Span(['File Name: ', previous_file],
                      id=f'bm-uploaded-file-name-xtrain'),
            html.Br(),
            html.Span(
                ['Date Uploaded: ', date_uploaded])
        ])

    if file_content is not None:
        bm_conf.bm_conf_input['file_path']['X_train'] = f'./data/raw/{file_name}'

        return UtilityTools.parse_contents(file_content, file_name, 'bm', '-xtrain')

    default_response_page = html.Div([
        html.Span(['File Name: ', 'NA'],
                  id='bm-uploaded-file-name-xtrain'),
        html.Br(),
        html.Span(['Date Uploaded: ', 'NA'])
    ])

    return default_response_page


@app.callback(
    Output('bm-uploaded-file-meta-data-ytrain', 'children'),
    [
        Input('bm-raw-ytrain-data-upload', 'contents'),
        Input('use-fs-final-data-file', 'value')
    ],
    [
        State('bm-raw-ytrain-data-upload', 'filename'),
    ]
)
def bm_upload_raw_data_ytrain(file_content, use_fs_data_bool, file_name):
    if use_fs_data_bool:
        date_uploaded = dt.datetime.now().strftime('%d %B, %Y %H:%M:%S')
        previous_file = f'y_train_v{ml_config.fs_current_version - 0.1}.csv'

        y_train_data = joblib.load(
            './data/downloadables/feature_selection_files/feature_selected_meta.joblib')['meta_data']['y_train']

        pd.DataFrame({'target_label': y_train_data}).to_csv(
            f'./data/raw/{previous_file}', index=False)

        bm_conf.bm_conf_input['file_path']['y_train'] = f'./data/raw/{previous_file}'

        return html.Div([
            html.Span(['File Name: ', previous_file],
                      id=f'bm-uploaded-file-name-ytrain'),
            html.Br(),
            html.Span(
                ['Date Uploaded: ', date_uploaded])
        ])

    if file_content is not None:
        bm_conf.bm_conf_input['file_path']['y_train'] = f'./data/raw/{file_name}'

        return UtilityTools.parse_contents(file_content, file_name, 'bm', '-ytrain')

    default_response_page = html.Div([
        html.Span(['File Name: ', 'NA'],
                  id='bm-uploaded-file-name-ytrain'),
        html.Br(),
        html.Span(['Date Uploaded: ', 'NA'])
    ])

    return default_response_page


@app.callback(
    Output('bm-uploaded-file-meta-data-select-feat', 'children'),
    [
        Input('bm-raw-select-feat-data-upload', 'contents'),
        Input('use-fs-final-data-file', 'value')
    ],
    [
        State('bm-raw-select-feat-data-upload', 'filename'),
    ]
)
def bm_upload_raw_data_select_feat(file_content, use_fs_data_bool, file_name):
    if use_fs_data_bool:
        date_uploaded = dt.datetime.now().strftime('%d %B, %Y %H:%M:%S')
        previous_file = f'selected_feature_list_v{ml_config.fs_current_version - 0.1}.csv'

        select_feat_data = joblib.load(
            './data/downloadables/feature_selection_files/feature_selected_meta.joblib')['meta_data']['selected_features']

        select_feat_data = {f'{outer_key}-{inner_key}': feat_list for outer_key,
                            outer_value in select_feat_data.items() for inner_key, feat_list in outer_value.items()}

        select_feat_data = pd.concat([pd.DataFrame(
        )] + [pd.DataFrame({key: value}) for key, value in select_feat_data.items()], axis=1)

        select_feat_data.to_csv(
            f'./data/raw/{previous_file}', index=False)

        bm_conf.bm_conf_input['file_path'][
            'selected_feature'] = f'./data/raw/{previous_file}'

        return html.Div([
            html.Span(['File Name: ', previous_file],
                      id=f'bm-uploaded-file-name-select-feat'),
            html.Br(),
            html.Span(
                ['Date Uploaded: ', date_uploaded])
        ])

    if file_content is not None:
        bm_conf.bm_conf_input['file_path'][
            'selected_feature'] = f'./data/raw/{file_name}'

        return UtilityTools.parse_contents(file_content, file_name, 'bm', '-select-feat')

    default_response_page = html.Div([
        html.Span(['File Name: ', 'NA'],
                  id='bm-uploaded-file-name-select-feat'),
        html.Br(),
        html.Span(['Date Uploaded: ', 'NA'])
    ])

    return default_response_page


@app.callback(
    Output("bm-current-conf-modal-obj", "is_open"),
    [
        Input("bm-show-current-conf-btn", "n_clicks"),
        Input("bm-close-current-conf-modal", "n_clicks"),
    ],
    [
        State("bm-current-conf-modal-obj", "is_open")
    ],
)
def toggle_show_bm_conf_modal(show_btn, close_btn, open_state):
    if show_btn or close_btn:
        return not open_state
    return open_state


@app.callback(
    Output("bm-show-baseline-results-modal", "is_open"),
    [
        Input("bm-show-baseline-results-btn", "n_clicks"),
        Input("bm-close-baseline-results-modal", "n_clicks"),
    ],
    [
        State("bm-show-baseline-results-modal", "is_open")
    ],
)
def bm_toggle_show_baseline_results_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [
        Output("bm-run-complete", "is_open"),
        Output('bm-show-baseline-results-btn', 'disabled'),
        Output('bm-show-baseline-results-json', 'children'),
    ],
    [
        Input("run-baseline-modeling", "n_clicks"),
    ],
    [
        State("bm-run-complete", "is_open")
    ],
)
def bm_toggle_run_started_alert(run_transform_btn, run_status_alert_state):
    if run_transform_btn:
        UtilityTools._create_folder(
            'baseline_modeling_files', './data/downloadables')

        BaselineModelling().compile_baseline(**bm_conf.bm_conf_input)
        baseline_results = pd.read_excel(
            './data/downloadables/baseline_modeling_files/baseline_results.xlsx')

        _unique_serial_number = sha256(
            str(ml_config.dt_serial_number).encode('utf-8')).hexdigest()

        ml_config.bm_serial_number += 1
        ml_config.bm_current_version += 0.1
        ml_config.bm_initial_status = 'Run Completed Successfully'

        ml_config.status.loc[len(ml_config.status.index)] = [_unique_serial_number, dt.datetime.now(), ml_config.bm_current_version,
                                                             'Data Transformation', json.dumps(bm_conf.bm_conf_input), ml_config.bm_initial_status]

        feat_info_table = dbc.Table.from_dataframe(
            baseline_results, striped=True, bordered=True, hover=True)

        return not run_status_alert_state, False, feat_info_table

    return run_status_alert_state, True, html.Div([html.Span([html.Em(['None File Uploaded!'])])])
