from dash import Input, Output, State, html, dcc
from .fs_conf_input import fs_upload_layout, fs_form, fs_data_info

from src.dash_utils.utils import UtilityTools
from src.dash_utils.config_input import fs_conf, ml_config
from src.ml_scripts.feature_selection import FeatureSelection

from server import app
from hashlib import sha256
from zipfile import ZipFile

import json
import joblib

import dash_bootstrap_components as dbc
import datetime as dt
import pandas as pd


feature_selection_layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row([
                            dbc.Col([
                                dbc.Checkbox(
                                    id='use-dt-final-dataset',
                                    label="Use the Transformed Dataset from the previous Step.",
                                    value=False,
                                ),
                            ], width=12)
                        ]),

                        dbc.Row(
                            [
                                dbc.Col(
                                    [dbc.Label(['Upload Transformed Data File:']), fs_upload_layout], width=5),
                                dbc.Col([
                                    html.Div(
                                        [
                                            html.Div([
                                                html.Span(['File Name: ', 'NA'],
                                                          id='fs-uploaded-file-name'),
                                                html.Br(),
                                                html.Span(
                                                    ['Date Uploaded: ', 'NA'])
                                            ])
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
                                            "Feature Selection Configuration")),
                                        dbc.ModalBody(
                                            id='fs-show-current-conf'),
                                        dbc.ModalFooter([
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Reset Conf.",
                                                        id="fs-reset-current-conf-modal",
                                                        className="ms-auto",
                                                        n_clicks=0, style={'margin-right': '10px'}
                                                    ),

                                                    dbc.Button(
                                                        "Close",
                                                        id="fs-close-current-conf-modal",
                                                        className="ms-auto",
                                                        n_clicks=0,
                                                    )
                                                ]
                                            )
                                        ]),
                                    ], id="fs-current-conf-modal-obj", scrollable=True, is_open=False,
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
                                    fullscreen=True
                                ),
                            ]
                        ),

                        dbc.Row(
                            [
                                dbc.Col([fs_data_info])
                            ], style={'overflow-y': 'scroll', 'height': 'auto', 'max-height': '1024px'}
                        )
                    ], width=5
                ),
            ]
        )
    ]
)


# --------------------------------------------- ## FEATURE SELECTION CALLBACK ## --------------------------------------------- #


@app.callback(
    Output("fs-current-conf-modal-obj", "is_open"),
    [
        Input("fs-show-current-conf-btn", "n_clicks"),
        Input("fs-close-current-conf-modal", "n_clicks"),
    ],
    [
        State("fs-current-conf-modal-obj", "is_open")
    ],
)
def fs_toggle_show_conf_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


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
def fs_toggle_show_select_feat_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [
        Output("fs-run-complete", "is_open"),
        Output('fs-show-selected_feat-btn', 'disabled'),
        Output('fs-show-selected-features-json', 'children'),
    ],
    [
        Input("run-feature-selection", "n_clicks"),
    ],
    [
        State("fs-run-complete", "is_open")
    ],
)
def fs_toggle_run_started_alert(run_selection_btn, run_status_alert_state):
    if run_selection_btn:
        UtilityTools._create_folder(
            'feature_selection_files', './data/downloadables')

        selected_features = FeatureSelection().compile_selection(**fs_conf.fs_conf_input)

        feature_list = {f'{outer_key}-{inner_key}': feat_list for outer_key,
                        outer_value in selected_features.items() for inner_key, feat_list in outer_value.items()}

        feature_list = pd.concat([pd.DataFrame(
        )] + [pd.DataFrame({key: value}) for key, value in feature_list.items()], axis=1)

        select_feature_table = dbc.Table.from_dataframe(
            feature_list, striped=True, bordered=True, hover=True)

        _unique_serial_number = sha256(
            str(ml_config.fs_serial_number).encode('utf-8')).hexdigest()

        ml_config.fs_serial_number += 1
        ml_config.fs_current_version += 0.1
        ml_config.fs_initial_status = 'Run Completed Successfully'

        ml_config.status.loc[len(ml_config.status.index)] = [_unique_serial_number, dt.datetime.now(), ml_config.fs_current_version,
                                                             'Data Transformation', json.dumps(fs_conf.fs_conf_input), ml_config.fs_initial_status]

        return [not run_status_alert_state, False, select_feature_table]

    return [run_status_alert_state, True, html.Div()]


@app.callback(
    Output("fs-download", "data"),
    [
        Input("download-fs-data", "n_clicks")
    ]
)
def fs_download_data(download_btn):
    if download_btn:
        feature_select_meta = joblib.load(fs_conf.fs_conf_input['save_path'])

        feature_list = feature_select_meta['meta_data']['selected_features']
        feature_list = {f'{outer_key}-{inner_key}': feat_list for outer_key,
                        outer_value in feature_list.items() for inner_key, feat_list in outer_value.items()}

        feature_list = pd.concat([pd.DataFrame(
        )] + [pd.DataFrame({key: value}) for key, value in feature_list.items()], axis=1)

        feature_list.to_csv(
            f'./data/downloadables/feature_selection_files/selected_feature_list_v{ml_config.fs_current_version - 0.1}.csv', index=False)

        for file_type in ['y_train', 'y_test']:
            pd.DataFrame({'target_label': feature_select_meta['meta_data'][file_type]}).to_csv(
                f'./data/downloadables/feature_selection_files/{file_type}_v{ml_config.fs_current_version - 0.1}.csv', index=False)

        for file_type in ['X_train', 'X_test']:
            feature_select_meta['meta_data'][file_type].to_csv(
                f'./data/downloadables/feature_selection_files/{file_type}_v{ml_config.fs_current_version - 0.1}.csv', index=False)

        with ZipFile('./data/zip_files/feature_selection.zip', 'w') as zip_obj:
            zip_obj.write(
                f'./data/downloadables/feature_selection_files/selected_feature_list_v{ml_config.fs_current_version - 0.1}.csv',
                f'selected_feature_list_v{ml_config.fs_current_version - 0.1}.csv')

            for file_type in ['X_train', 'X_test', 'y_train', 'y_test']:
                zip_obj.write(
                    f'./data/downloadables/feature_selection_files/{file_type}_v{ml_config.fs_current_version - 0.1}.csv',
                    f'{file_type}_v{ml_config.fs_current_version - 0.1}.csv')

            zip_obj.write(
                './data/downloadables/feature_selection_files/feature_selected_meta.joblib', 'feature_selected_meta.joblib')

        return dcc.send_file('./data/zip_files/feature_selection.zip', type='zip')
