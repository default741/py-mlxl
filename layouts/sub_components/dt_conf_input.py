from dash import html, dcc
import dash_bootstrap_components as dbc

from .dt_method_layout_mapping import method_layout_mapping
from src.dash_utils.config_input import dt_conf

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
                            html.Span([' Download Transformed File'])
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
