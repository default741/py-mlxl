from dash import html, dcc
import dash_bootstrap_components as dbc

from src.dash_utils.config_input import fs_conf


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
            disabled=True),
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
                        ], color='primary', id='fs-save-conf-btn', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.6rem'}
                    )
                ], width=4),

                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-play'),
                            html.Span([' Run Feature Selection'])
                        ], id='run-feature-selection', color='primary', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.6rem'}
                    )
                ], width=4),

                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className='fa-solid fa-download'),
                            html.Span([' Download Pickle Data'])
                        ], id='download-fs-pickle-data', color='primary', style={'margin-top': '10px', 'width': '100%', 'fontSize': '0.6rem'}
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
            ], id="fs-ls-loading-2", type="circle", style={'margin-top': '5px', 'margin-botttom': '5px'}
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
