from dash import Input, Output, State, html, dcc

from .sub_components.data_transform.dt_layout import data_transform_layout
from .sub_components.feature_selection.fs_layout import feature_selection_layout
from .sub_components.bm_conf_input import get_upload_button, bm_form

from src.dash_utils.utils import UtilityTools
from src.dash_utils.config_input import ml_config, bm_conf

from server import app
from hashlib import sha256

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
                children=[
                    baseline_modeling_layout], title=f"Baseline Modeling Details, Status - {ml_config.dt_initial_status}", id='bm-accordion-title'
            ),
            dbc.AccordionItem(
                children=[], title=f"Hyper-Parameter Tuning Details, Status - {ml_config.dt_initial_status}", id='hpt-accordion-title'
            ),
            dbc.AccordionItem(
                children=[], title=f"Final Model Training Details, Status - {ml_config.dt_initial_status}", id='fmt-accordion-title'
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
                "Sub-Modules: Data Transformation, Feature Selection, Baseline Modeling, Hyper-Parameter Tuning, Final Model Training",
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
            './data/feature_selected_data_v1.joblib')['meta_data']['X_train']

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
            './data/feature_selected_data_v1.joblib')['meta_data']['y_train']

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
            './data/feature_selected_data_v1.joblib')['meta_data']['selected_features']

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
    [
        Output('bm-feature-set-list', 'disabled'),
        Output('bm-feature-set-list', 'options'),
        Output('bm-feature-set-list', 'value'),
        Output('bm-select-feature-set-dropdown', 'options'),
        Output('bm-select-feature-set-dropdown', 'value')
    ],
    [
        Input('bm-uploaded-file-name-select-feat', 'children')
    ]
)
def bm_show_feature_set_list(file_name_children):
    file_name = file_name_children[-1]

    if not (file_name == 'NA'):
        feature_set_options = list()

        raw_data_uploaded = pd.read_csv(f'./data/raw/{file_name}')
        bm_conf.selected_features = raw_data_uploaded.copy()

        bm_conf.bm_conf_input['file_path'][
            'selected_feature'] = f'./data/raw/{file_name}'

        feature_sets = list(raw_data_uploaded.columns)

        for feat_set in feature_sets:
            feature_set_options.append({
                'label': feat_set, 'value': feat_set
            })

        return (False, feature_set_options, feature_sets[0], feature_set_options + [{'label': 'All Datasets', 'value': 'all'}], 'all')

    return (True, [], '', [], '')


@app.callback(
    Output('bm-show-feature-set-list', 'children'),
    [
        Input('bm-feature-set-list', 'value')
    ]
)
def bm_show_select_feature_list_view(select_feature):
    if not bm_conf.selected_features.empty:
        features = list(bm_conf.selected_features[select_feature].dropna())

        return html.Div([
            dbc.ListGroup([dbc.ListGroupItem(feature)
                          for feature in features], numbered=True)
        ])

    return html.Span(['No File Uploaded!'])


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
    [
        Output('bm-show-current-conf', 'children'),
        Output('bm-accordion-title', 'title')
    ],
    [
        Input('bm-save-conf-btn', 'n_clicks'),
        Input('bm-reset-current-conf-modal', 'n_clicks'),
        Input('bm-check-imbalance-radio-select', 'value'),
        Input('bm-select-feature-set-dropdown', 'value'),
        Input('bm-select-model-list-dropdown', 'value'),
        Input('bm-select-sample-list-dropdown', 'value'),
        Input('bm-enable-voting-classifier-checkbox', 'value'),
        Input('bm-select-voting-models', 'value'),
        Input('bm-select-voting-sample', 'value')
    ]
)
def bm_update_config(
    save_conf_btn, reset_conf_btn, imbalance_bool, feature_set, model_list, sample_list, voting_bool, voting_model_list, voting_sample_list
):
    if save_conf_btn:
        if imbalance_bool != 2:
            bm_conf.bm_conf_input['balanced_data'] = bool(imbalance_bool)
        else:
            bm_conf.bm_conf_input['balanced_data'] = False
            bm_conf.bm_conf_input['check_imbalance'] = True
            bm_conf.bm_conf_input['imbalance_class'] = 1
            bm_conf.bm_conf_input['imbalance_threshold'] = 0.0

        bm_conf.bm_conf_input['feature_set_list'] = feature_set if isinstance(
            feature_set, list) else [feature_set]
        bm_conf.bm_conf_input['model_list'] = model_list if isinstance(
            model_list, list) else [model_list]
        bm_conf.bm_conf_input['sample_list'] = sample_list if isinstance(
            sample_list, list) else [sample_list]

        if voting_bool:
            bm_conf.bm_conf_input['enable_voting'] = voting_bool
            bm_conf.bm_conf_input['voting_model_list'] = voting_model_list if isinstance(
                voting_model_list, list) else [voting_model_list]
            bm_conf.bm_conf_input['voting_sample_list'] = voting_sample_list if isinstance(
                voting_sample_list, list) else [voting_sample_list]

    if reset_conf_btn:
        bm_conf.refresh_config()

    current_status = f"Baseline Modeling Details, Status - {ml_config.fs_initial_status}"

    json_output = html.Div(
        [
            html.Pre(children=["JSON Config: {}".format(
                json.dumps(bm_conf.bm_conf_input, indent=4))])
        ]
    )

    return [json_output, current_status]


@app.callback(
    Output("bm-save-conf-file-success", "is_open"),
    [
        Input("bm-save-conf-btn", "n_clicks")
    ],
    [
        State("bm-save-conf-file-success", "is_open")
    ],
)
def toggle_fs_conf_save_alert(save_conf_bth, save_conf_alert_state):
    if save_conf_bth:
        return not save_conf_alert_state

    return save_conf_alert_state


@app.callback(
    Output('bm-select-sample-list-dropdown', 'disabled'),
    [
        Input('bm-check-imbalance-radio-select', 'value')
    ]
)
def toggle_bm_sample_dropdown(imbalance_bool):
    if imbalance_bool == 1:
        return True

    return False


@app.callback(
    [
        Output('bm-select-voting-models', 'disabled'),
        Output('bm-select-voting-sample', 'disabled')
    ],
    [
        Input('bm-enable-voting-classifier-checkbox', 'value')
    ]
)
def toggle_bm_voting_options(voting_bool):
    return [not voting_bool, not voting_bool]

# --------------------------------------------- ## HYPER-PARAMETER TUNING CALLBACK ## --------------------------------------------- #
