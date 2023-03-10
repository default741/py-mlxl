from dash import html, dcc
import dash_bootstrap_components as dbc

from src.dash_utils.config_input import bm_conf


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
    [check_imbalance_layout, bm_select_feature_set, bm_model_list, bm_sample_list, voting_classifier_layout, alert_tab_layout, bm_btn_group])
