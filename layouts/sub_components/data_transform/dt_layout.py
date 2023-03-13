from dash import Input, Output, State, html, dcc

from .dt_conf_input import dt_upload_layout, dt_form, dt_data_info

from src.dash_utils.utils import UtilityTools
from src.dash_utils.config_input import dt_conf, ml_config
from src.ml_scripts.data_transform import DataTransform

from server import app
from hashlib import sha256
from zipfile import ZipFile

import json

import dash_bootstrap_components as dbc
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
                                            html.Div([
                                                html.Span(['File Name: ', 'NA'],
                                                          id='dt-uploaded-file-name'),
                                                html.Br(),
                                                html.Span(
                                                    ['Date Uploaded: ', 'NA'])
                                            ])
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

                                dbc.Button(
                                    [
                                        html.I(
                                            className='fa-regular fa-file'),
                                        html.Span(
                                            [' Show Transformed Data (CSV)'])
                                    ], id='dt-show-transformed-data-btn', color='primary', disabled=True, style={
                                        'margin-top': '10px', 'margin-bottom': '15px', 'width': '100%', 'fontSize': '0.8rem', 'height': 'auto'
                                    }
                                ),

                                dbc.Modal(
                                    [
                                        dbc.ModalHeader(dbc.ModalTitle(
                                            "Data Transform Configuration")),
                                        dbc.ModalBody(
                                            id='dt-show-current-conf'),
                                        dbc.ModalFooter(
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Reset Conf.",
                                                        id="dt-reset-current-conf-modal",
                                                        className="ms-auto",
                                                        n_clicks=0, style={'margin-right': '10px'}
                                                    ),

                                                    dbc.Button(
                                                        "Close",
                                                        id="dt-close-current-conf-modal",
                                                        className="ms-auto",
                                                        n_clicks=0,
                                                    )
                                                ]
                                            )
                                        ),
                                    ],
                                    id="dt-current-conf-modal-obj",
                                    scrollable=True,
                                    is_open=False,
                                ),

                                dbc.Modal(
                                    [
                                        dbc.ModalHeader(dbc.ModalTitle(
                                            "Transformed Data Preview")),
                                        dbc.ModalBody(
                                            id='dt-show-transformed-data'),
                                        dbc.ModalFooter(
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Close",
                                                        id="dt-close-transformed-data-modal",
                                                        className="ms-auto",
                                                        n_clicks=0,
                                                    )
                                                ]
                                            )
                                        ),
                                    ],
                                    id="dt-transformed-data-modal-obj",
                                    scrollable=True,
                                    is_open=False,
                                    fullscreen=True
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


# --------------------------------------------- ## DATA TRANSFORMATION CALLBACK ## --------------------------------------------- #


@app.callback(
    Output("dt-current-conf-modal-obj", "is_open"),
    [
        Input("dt-show-current-conf-btn", "n_clicks"),
        Input("dt-close-current-conf-modal", "n_clicks"),
    ],
    [
        State("dt-current-conf-modal-obj", "is_open")
    ],
)
def dt_toggle_show_conf_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("dt-transformed-data-modal-obj", "is_open"),
    [
        Input("dt-show-transformed-data-btn", "n_clicks"),
        Input("dt-close-transformed-data-modal", "n_clicks"),
    ],
    [
        State("dt-transformed-data-modal-obj", "is_open")
    ],
)
def dt_toggle_show_transformed_data_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [
        Output("dt-run-complete", "is_open"),
        Output('dt-show-transformed-data-btn', 'disabled'),
        Output('dt-show-transformed-data', 'children'),
    ],
    [
        Input("run-data-transform", "n_clicks"),
    ],
    [
        State("dt-run-complete", "is_open")
    ],
)
def dt_toggle_run_started_alert(run_transform_btn, run_status_alert_state):
    if run_transform_btn:
        UtilityTools._create_folder(
            'data_transform_files', './data/downloadables')

        transformed_data = DataTransform().compile_pipeline(**dt_conf.dt_conf_input)
        transformed_data.to_csv(
            f'./data/downloadables/data_transform_files/transformed_data_v{ml_config.dt_current_version}.csv', index=False)

        _unique_serial_number = sha256(
            str(ml_config.dt_serial_number).encode('utf-8')).hexdigest()

        ml_config.dt_serial_number += 1
        ml_config.dt_current_version += 0.1
        ml_config.dt_initial_status = 'Run Completed Successfully'

        ml_config.status.loc[len(ml_config.status.index)] = [_unique_serial_number, dt.datetime.now(), ml_config.dt_current_version,
                                                             'Data Transformation', json.dumps(dt_conf.dt_conf_input), ml_config.dt_initial_status]

        feat_info_table = dbc.Table.from_dataframe(
            transformed_data.loc[:10, :], striped=True, bordered=True, hover=True)

        return not run_status_alert_state, False, feat_info_table

    return run_status_alert_state, True, html.Div([html.Span([html.Em(['None File Uploaded!'])])])


@app.callback(
    Output("dt-download", "data"),
    [
        Input("download-dt-data", "n_clicks")
    ]
)
def dt_download_data(download_btn):
    if download_btn:
        with ZipFile('./data/zip_files/data_transform.zip', 'w') as zip_obj:
            zip_obj.write(
                f'./data/downloadables/data_transform_files/transformed_data_v{ml_config.dt_current_version - 0.1}.csv',
                f'transformed_data_v{ml_config.dt_current_version - 0.1}.csv')
            zip_obj.write(
                './data/downloadables/data_transform_files/transform_pipeline.joblib', 'transform_pipeline.joblib')

        return dcc.send_file('./data/zip_files/data_transform.zip', type='zip')
