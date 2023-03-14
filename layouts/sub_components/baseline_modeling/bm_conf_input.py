from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc

from src.dash_utils.config_input import bm_conf, ml_config

import pandas as pd
import json

from server import app


def get_upload_button(id_type: str, value_str: str):
    return html.Div(
        [
            dcc.Upload(
                [
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-upload'),
                            html.Span([f" Upload {value_str}"])
                        ], id=f"bm-open-upload-{id_type}-data", style={'width': '100%'}
                    ),
                ], id=f'bm-raw-{id_type}-data-upload', multiple=False
            ),
        ]
    )


check_imbalance_layout = html.Div(
    [
        dbc.Label("Is the Data Balanced or Imbalanced?"),
        dbc.RadioItems(
            options=[
                {"label": "Balanced", "value": 1},
                {"label": "Imbalanced", "value": 0},
                {"label": "Don't Know", "value": 2},
            ],
            value=1,
            id="bm-check-imbalance-radio-select",
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Imbalance Class in Data:"),
                dbc.Input(type="number", placeholder=1,
                          value=1, id='bm-imb-class-input', max=1, min=0,
                          step=1, disabled=True),
            ], className="mb-3", style={'margin-top': '10px'}
        ),
        dbc.InputGroup(
            [
                dbc.InputGroupText("Imbalance Class Threshold:"),
                dbc.Input(type="number", placeholder=1,
                          value=0.2, id='bm-imb-threshold-input', max=1, min=0,
                          step=0.1, disabled=True),
                dbc.InputGroupText("% (Percent)"),
            ], className="mb-3"
        ),
    ]
)

bm_select_feature_set = html.Div(
    [
        html.Hr(),
        dbc.Label("Select Feature Set (for Baseline Modeling):"),
        dcc.Dropdown(
            id='bm-select-feature-set-dropdown', multi=True,
            placeholder='Select Feature Set'
        ),
    ]
)

bm_model_list = html.Div(
    [
        dbc.Label("Select Models (for Baseline Modeling):"),
        dcc.Dropdown(
            id='bm-select-model-list-dropdown', multi=True, value='all',
            options=[
                {'label': model.capitalize(), 'value': model} for model in bm_conf.model_list
            ],
            placeholder='Select Model'
        ),
    ], style={'margin-top': '10px'}
)

bm_sample_list = html.Div(
    [
        dbc.Label("Select Sampling Methods (for Baseline Modeling):"),
        dcc.Dropdown(
            id='bm-select-sample-list-dropdown', multi=True, value='all', disabled=True,
            options=[
                {'label': model.capitalize(), 'value': model} for model in bm_conf.sample_list
            ],
            placeholder='Select Sampling Method'
        ),
    ], style={'margin-top': '10px'}
)

voting_classifier_layout = html.Div(
    [
        html.Hr(),
        dbc.Label(['Run Voting Classifier:']),
        dbc.Checkbox(
            label='Enable Voting Classifier',
            value=False,
            id="bm-enable-voting-classifier-checkbox",
        ),

        dbc.Label("Select Voting Models:"),
        dcc.Dropdown(
            id='bm-select-voting-models',
            multi=True,
            value=list(bm_conf.model_list)[:2],
            options=[
                {'label': model.capitalize(), 'value': model} for model in bm_conf.model_list
            ],
            placeholder='Select Model',
            disabled=True
        ),

        dbc.Label("Select Voting Sample:", style={'margin-top': '10px'}),
        dcc.Dropdown(
            id='bm-select-voting-sample',
            multi=False,
            value=list(bm_conf.sample_list)[0],
            options=[
                {'label': model.capitalize(), 'value': model} for model in bm_conf.sample_list
            ],
            placeholder='Select Sampling Method',
            disabled=True
        ),
    ], style={'margin-top': '10px'}
)

kpi_sorting_layout = html.Div(
    [
        html.Hr(),
        dbc.Label("Select KPI to sort results:", style={'margin-top': '10px'}),
        dcc.Dropdown(
            id='bm-select-sorting-kpi',
            multi=False,
            value=list(bm_conf.kpi_list)[0],
            options=[
                {'label': kpi.capitalize(), 'value': kpi} for kpi in bm_conf.kpi_list
            ],
            placeholder='Select Sorting KPI',
        ),
    ]
)

bm_btn_group = html.Div(
    [
        html.Hr(),

        dbc.Row(
            [
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-floppy-disk'),
                            html.Span([' Save Configuration'])
                        ], color='primary', id='bm-save-conf-btn', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.6rem'}
                    )
                ], width=4),

                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-play'),
                            html.Span([' Run Baseline Modeling'])
                        ], id='run-baseline-modeling', color='primary', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.6rem'}
                    )
                ], width=4),

                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-download'),
                            html.Span([' Download Baseline Results'])
                        ], id='download-bm-results-btn', color='primary', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.6rem'}
                    ),
                    dcc.Download(id="bm-download-results")
                ], width=4),
            ]
        )
    ]
)

alert_tab_layout = html.Div(
    [
        dbc.Alert(
            "Configuration Save Successfully!",
            id="bm-save-conf-file-success", is_open=False, duration=2000,
        ),

        dbc.Alert(
            "Data Transformation Pipeline Starting to Execute!",
            id="bm-run-started", is_open=False, duration=1000, color="info"
        ),

        dcc.Loading(
            children=[
                dbc.Alert(
                    "Data Transformation Pipeline Run Complete!",
                    id="bm-run-complete", is_open=False, duration=2000,
                ),
            ], id="bm-ls-loading-2", type="cube", fullscreen=True, style={'margin-top': '5px', 'margin-botttom': '5px'}
        ),


    ],
    id='dt-show-alert', style={'margin-top': '5px', 'margin-bottom': '5px'}
)

bm_form = html.Div(
    [check_imbalance_layout, bm_select_feature_set, bm_model_list, bm_sample_list, voting_classifier_layout, kpi_sorting_layout,
     alert_tab_layout, bm_btn_group])


# --------------------------------------------- ## BASELINE MODELING CALLBACK ## --------------------------------------------- #


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
    Output("bm-save-conf-file-success", "is_open"),
    [
        Input("bm-save-conf-btn", "n_clicks")
    ],
    [
        State("bm-save-conf-file-success", "is_open")
    ],
)
def bm_toggle_conf_save_alert(save_conf_bth, save_conf_alert_state):
    if save_conf_bth:
        return not save_conf_alert_state

    return save_conf_alert_state


@app.callback(
    Output('bm-select-sample-list-dropdown', 'disabled'),
    [
        Input('bm-check-imbalance-radio-select', 'value')
    ]
)
def bm_toggle_sample_dropdown(imbalance_bool):
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
def bm_toggle_voting_options(voting_bool):
    return [not voting_bool, not voting_bool]


@app.callback(
    [
        Output('bm-imb-class-input', 'disabled'),
        Output('bm-imb-threshold-input', 'disabled')
    ],
    [
        Input('bm-check-imbalance-radio-select', 'value')
    ]
)
def bm_toggle_imb_check_inputs(check_imb_value):
    if check_imb_value == 2:
        return (False, False)

    return True, True


@app.callback(
    [
        Output('bm-show-current-conf', 'children'),
        Output('bm-accordion-title', 'title'),
        Output('bm-reset-current-conf-modal', 'n_clicks'),
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
        Input('bm-select-voting-sample', 'value'),
        Input('bm-select-sorting-kpi', 'value')
    ]
)
def bm_update_config(
    save_conf_btn, reset_conf_btn, imbalance_bool, feature_set, model_list, sample_list, voting_bool, voting_model_list, voting_sample_list,
    sorting_kpi
):
    if save_conf_btn:
        if imbalance_bool == 1:
            bm_conf.bm_conf_input['balanced_data'] = bool(imbalance_bool)

        elif imbalance_bool == 0:
            bm_conf.bm_conf_input['balanced_data'] = False

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

        bm_conf.bm_conf_input['kpi_sorting'] = [sorting_kpi]

    if reset_conf_btn:
        reset_conf_btn = 0
        bm_conf.refresh_config()

    current_status = f"Baseline Modeling Details, Status - {ml_config.fs_initial_status}"

    json_output = html.Div(
        [
            html.Pre(children=["JSON Config: {}".format(
                json.dumps(bm_conf.bm_conf_input, indent=4))])
        ]
    )

    return [json_output, current_status, reset_conf_btn]
