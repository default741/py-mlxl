from dash import Dash, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px

import pandas as pd
import numpy as np

from src.dash_utils.utils import UtilityTools
from src.dash_utils.config_input import eda_conf

from server import app


eda_upload_layout = html.Div(
    [
        dcc.Upload(
            [
                dbc.Button(
                    [
                        html.I(className='fa-solid fa-upload'),
                        html.Span([" Upload"])
                    ], id="eda-open-upload-data", style={'width': '100%'}
                ),
            ], id='eda-raw-data-upload', multiple=False
        ),
    ]
)


modal_graph_null = html.Div(
    [
        dbc.Button(
            [
                html.I(
                    className='fa-regular fa-file-code'),
                html.Span(
                    [' Show Null Data'])
            ], id='open-fs-null', color='primary', style={
                'margin-top': '15px', 'margin-bottom': '15px', 'width': '100%', 'fontSize': '0.8rem', 'height': 'auto'
            }
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Null value Graph")),
                dbc.ModalBody(dcc.Graph(id='modal-graph-null', figure={})),
            ],
            id="modal-fs-null",
            fullscreen=True,

        ),
    ]
)

modal_graph_corr = html.Div(
    [
        dbc.Button(
            [
                html.I(
                    className='fa-regular fa-file-code'),
                html.Span(
                    [' Show Correlation info'])
            ], id='open-fs-corr', color='primary', style={
                'margin-top': '15px', 'margin-bottom': '15px', 'width': '100%', 'fontSize': '0.8rem', 'height': 'auto'
            }
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Correlation Graph")),
                dbc.ModalBody(dcc.Graph(id='modal-graph-corr', figure={})),
            ],
            id="modal-fs-corr",
            fullscreen=True,

        ),
    ]
)


categorical_tab = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Label(['Univariate Analysis:']),
        ], width=5),

        dbc.Col(width=2),

        dbc.Col([
            dbc.Label(['Bi-variate Analysis:']),
        ], width=5)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Div(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='categorical-univariate-feat-list-select',
                                                 multi=False, options=[])
                                ],
                                title="Feature List",
                            ),
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(
                                        id='categorical-univariate-plot-list-select',
                                        multi=False,
                                        options=[
                                            {'label': 'Histogram',
                                                'value': 'Histogram'}
                                        ]
                                    )
                                ],
                                title="Graph Categories Select",
                            ),
                        ], always_open=True
                    )
                )
            ]),
            dbc.Row([
                html.Div(
                    [
                        dbc.Button("Generate Graph",
                                   id="generate-categorical-univariate-graph-btn"),
                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle("Graph")),
                                dbc.ModalBody(
                                    dcc.Graph(id='categorical-univariate-graph-modal-body', figure={})),
                                dbc.ModalFooter([
                                    dbc.Button(id='')
                                ])
                            ],
                            id="categorical-univariate-graph-modal", fullscreen=True
                        ),
                    ]
                )
            ])
        ], width=5),

        dbc.Col(width=2),

        dbc.Col([
            dbc.Row([
                html.Div(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='categorical-bivariate-feat-list-x-select',
                                                 multi=False, options=[])
                                ],
                                title="X-Axis Feature List",
                            ),
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='categorical-bivariate-feat-list-y-select',
                                                 multi=False, options=[])
                                ],
                                title="Y-Axis Feature List",
                            ),
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='categorical-bivariate-plot-list-select',
                                                 multi=False, options=['Scatter'])
                                ],
                                title="Graph Categories Select",
                            ),
                        ], always_open=True
                    )
                )
            ]),
            dbc.Row([
                html.Div(
                    [
                        dbc.Button("Generate Graph",
                                   id="generate-categorical-bivariate-graph-btn"),
                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle("Graph")),
                                dbc.ModalBody(
                                    dcc.Graph(id='categorical-bivariate-graph-modal-body', figure={})),
                            ],
                            id="categorical-bivariate-graph-modal", fullscreen=True,

                        ),
                    ]
                )
            ])
        ], width=5)
    ])
])


numerical_tab = html.Div([
    dbc.Row([
        dbc.Col([dbc.Label(['Univariate Analysis:'])], width=5),
        dbc.Col(width=2),
        dbc.Col([dbc.Label(['Bi-variate Analysis:'])], width=5)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Div(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='numerical-univariate-feat-list-select',
                                                 multi=False, options=[])
                                ],
                                title="Feature List",
                            ),
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='numerical-univariate-plot-list-select',
                                                 multi=False, options=['Histogram', 'Box', 'Violin', 'Ecdf'])
                                ],
                                title="Graph Categories Select",
                            ),
                        ], always_open=True
                    )
                )
            ]),
            dbc.Row([
                html.Div(
                    [
                        dbc.Button("Generate Graph",
                                   id="generate-numerical-univariate-graph-btn"),
                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle("Graph")),
                                dbc.ModalBody(
                                    dcc.Graph(id='categorical-univariate-graph-modal-body', figure={})),
                            ],
                            id="categorical-univariate-graph-modal", fullscreen=True,

                        ),
                    ]
                )
            ])
        ], width=5),

        dbc.Col(width=2),

        dbc.Col([
            dbc.Row([
                html.Div(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='numerical-bivariate-feat-list-x-select',
                                                 multi=False, options=[])
                                ],
                                title="X-Axis Feature List",
                            ),
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='numerical-bivariate-feat-list-y-select',
                                                 multi=False, options=[])
                                ],
                                title="Y-Axis Feature List",
                            ),
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='numerical-bivariate-plot-list-select',
                                                 multi=False, options=['Line', 'Scatter', 'Histogram', 'Density heat map'])
                                ],
                                title="Graph Categories Select",
                            ),
                        ], always_open=True
                    )
                )
            ]),
            dbc.Row([
                html.Div(
                    [
                        dbc.Button("Generate Graph",
                                   id="generate-numerical-bivariate-graph-btn"),
                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle("Graph")),
                                dbc.ModalBody(
                                    dcc.Graph(id='categorical-bivariate-graph-modal-body', figure={})),
                            ],
                            id="categorical-bivariate-graph-modal", fullscreen=True,

                        ),
                    ]
                )
            ])
        ], width=5)
    ]),

    dbc.Row([
        dbc.Col([dbc.Label(['Multi-variate Analysis:'])], width=5),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Div(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='numerical-multi-variate-feat-list-select',
                                                 multi=True, options=[])
                                ],
                                title="Select Multiple Features",
                            ),
                            dbc.AccordionItem(
                                [
                                    dcc.Dropdown(id='numerical-multi-variate-plot-list-select',
                                                 multi=False, options=['Scatter'])
                                ],
                                title="Graph Categories Select",
                            ),
                        ],
                    )
                )
            ]),
            dbc.Row([
                html.Div(
                    [
                        dbc.Button("Generate Graph",
                                   id="generate-numerical-multi-variate-graph-btn"),
                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle(
                                    "Multi Variate Graph")),
                                dbc.ModalBody(
                                    dcc.Graph(id='numerical-multi-variate-graph-modal-body', figure={})),
                            ],
                            id="numerical-multi-variate-graph-modal", fullscreen=True,

                        ),
                    ]
                )
            ])
        ], width=5),
    ]),
])


EDA_layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [dbc.Label(['Upload Raw Data File:']), eda_upload_layout], width=4),
                dbc.Col([
                    html.Div(
                        [
                            html.Span(['File Name: ', 'NA'],
                                      id='eda-uploaded-file-name'),
                            html.Br(),
                            html.Span(
                                ['Date Uploaded: ', 'NA'])
                        ], id='eda-uploaded-file-meta-data', style={'margin-left': '5px'}
                    )
                ], width=8, style={'border-left': '1px solid black'}),
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            [
                                html.I(
                                    className='fa-regular fa-file-code'),
                                html.Span(
                                    [' Show Feature Info.'])
                            ],
                            id='eda-show-feat-list-btn',
                            color='primary',
                            style={
                                'margin-top': '15px', 'margin-bottom': '15px', 'width': '100%', 'fontSize': '0.8rem', 'height': 'auto'
                            }
                        ),

                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle(
                                    "Feature Infomation")),
                                dbc.ModalBody(id='eda-show-current-conf'),
                                dbc.ModalFooter(
                                    dbc.Button(
                                        "Close Feature List",
                                        id="eda-close-feature-list-modal",
                                        className="ms-auto",
                                        n_clicks=0,
                                    )
                                ),
                            ],
                            id="feature-list-modal",
                            scrollable=True,
                            is_open=False,
                        ),
                    ]
                ),

                dbc.Col(
                    [modal_graph_null]
                ),

                dbc.Col(
                    [modal_graph_corr]
                ),
            ]),

        dbc.Row(
            [
                html.Div(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab([numerical_tab], label="EDA for Numerical Features",
                                        id="numerical-tab"),
                                dbc.Tab([categorical_tab], label="EDA for Categorical Features",
                                        id="categorical-tab"),
                            ],
                            id="feature-dtype-tab-select", active_tab="numerical-tab",
                        ),
                        html.Div(id="eda-tab-content"),
                    ]
                )
            ]
        )
    ]
)

# -------------------------------------------------------------- CALLBACK METHODS --------------------------------------------------------------


@callback(
    Output("eda-tab-content", "children"),
    [
        Input("feature-dtype-tab-select", "active_tab")
    ]
)
def switch_current_tab(active_tab):
    if active_tab == "numerical-tab":
        return numerical_tab

    elif active_tab == "categorical-tab":
        return categorical_tab


@callback(
    Output('eda-uploaded-file-meta-data', 'children'),
    [
        Input('eda-raw-data-upload', 'contents')
    ],
    [
        State('eda-raw-data-upload', 'filename'),
    ]
)
def eda_upload_raw_data(file_content, file_name):
    if file_content is not None:
        return UtilityTools.parse_contents(file_content, file_name, 'eda')

    default_response_page = html.Div(
        [
            html.Span(['File Name: ', 'NA'],
                      id='eda-uploaded-file-name'),
            html.Br(),
            html.Span(['Date Uploaded: ', 'NA'])
        ], id='eda-uploaded-file-meta-data', style={'margin-top': '10px'}
    )

    return default_response_page


@callback(
    Output('eda-show-current-conf', 'children'),
    [
        Input('eda-uploaded-file-name', 'children')
    ]
)
def eda_show_feature_info_list(file_name_children):
    file_name = file_name_children[-1]

    if not (file_name == 'NA'):
        raw_data_uploaded = pd.read_csv(f'./data/raw/{file_name}')
        eda_conf.raw_data = raw_data_uploaded.copy()

        data_dtypes = raw_data_uploaded.dtypes
        data_dtypes = pd.DataFrame(
            {'Feature Name': data_dtypes.index, 'Feature Data Type': list(map(str, data_dtypes.values))})

        feat_info_table = dbc.Table.from_dataframe(
            data_dtypes, striped=True, bordered=True, hover=True)

        return feat_info_table

    return html.Div([html.Span([html.Em(['None File Uploaded!'])])])


@callback(
    Output("feature-list-modal", "is_open"),
    [
        Input("eda-show-feat-list-btn", "n_clicks"),
        Input("eda-close-feature-list-modal", "n_clicks"),
    ],
    [
        State("feature-list-modal", "is_open")
    ],
)
def toggle_modal(open_modal, close_modal, modal_is_open_state):
    if open_modal or close_modal:
        return not modal_is_open_state
    return modal_is_open_state


@callback(
    [
        Output("categorical-univariate-feat-list-select", "options"),
        Output("categorical-bivariate-feat-list-x-select", "options"),
        Output("categorical-bivariate-feat-list-y-select", "options")
    ],
    [
        Input('eda-uploaded-file-name', 'children')
    ]
)
def get_categorical_feature_list(file_name_children):
    file_name = file_name_children[-1]

    if not (file_name == 'NA'):
        raw_data_uploaded = eda_conf.raw_data.copy()

        cat_feat_list = raw_data_uploaded.select_dtypes(
            ['object', 'category']).columns.tolist()

        if len(cat_feat_list) > 0:
            return (cat_feat_list, cat_feat_list, cat_feat_list)
        else:
            return ([], [], [])
    else:
        return ([], [], [])


@callback(
    [
        Output("numerical-univariate-feat-list-select", "options"),
        Output("numerical-bivariate-feat-list-x-select", "options"),
        Output("numerical-bivariate-feat-list-y-select", "options"),
        Output('numerical-multi-variate-feat-list-select', "options")
    ],
    [
        Input('eda-uploaded-file-name', 'children')
    ]
)
def get_numerical_feature_list(file_name_children):
    file_name = file_name_children[-1]

    if not (file_name == 'NA'):
        raw_data_uploaded = eda_conf.raw_data.copy()

        num_feat_list = raw_data_uploaded.select_dtypes(
            exclude=['object', 'category']).columns.tolist()

        if len(num_feat_list) > 0:
            return (num_feat_list, num_feat_list, num_feat_list, num_feat_list)
        else:
            return ([], [], [], [])
    else:
        return ([], [], [], [])


@callback(
    Output("categorical-univariate-graph-modal", "is_open"),
    [
        Input("generate-categorical-univariate-graph-btn", "n_clicks")
    ],
    [
        State("categorical-univariate-graph-modal", "is_open")
    ],
)
def toggle_categorical_univariate_modal(open_modal, is_open_state):
    if open_modal:
        return not is_open_state
    return is_open_state


@callback(
    Output("categorical-univariate-graph-modal-body", "figure"),
    [
        Input('eda-uploaded-file-name', 'children'),
        Input('categorical-univariate-feat-list-select', 'value'),
        Input('categorical-univariate-plot-list-select', 'value')
    ]
)
def plot_categorical_univariate_graph(file_name_children, categorical_feat, plot_select):
    file_name = file_name_children[-1]

    if not (file_name == 'NA'):
        raw_data_uploaded = eda_conf.raw_data.copy()

        if plot_select == 'Histogram':
            return px.histogram(raw_data_uploaded, x=categorical_feat)

        # elif cat_univ_plot == 'Box':
        #     fig = px.box(raw_data_uploaded, y=cat_univ_ft)
        #     return fig

        # elif cat_univ_plot == 'Violin':
        #     fig = px.violin(raw_data_uploaded, y=cat_univ_ft, box=True,  # draw box plot inside the violin
        #                     points='all',  # can be 'outliers', or False
        #                     )
        #     return fig

        else:
            return {}
    else:
        return {}


# @callback(
#     Output("modal-fs-cat-Bi", "is_open"),
#     Input("open-fs-cat-Bi", "n_clicks"),
#     State("modal-fs-cat-Bi", "is_open"),
# )
# def toggle_modal(n, is_open):
#     if n:
#         return not is_open
#     return is_open


# @callback(
#     Output("modal-graph-cat-Bi", "figure"),
#     [
#         Input('eda-uploaded-file-name', 'children'),
#         Input('cat-Bi-X', 'value'),
#         Input('cat-Bi-y', 'value'),
#         Input('cat-Bi-plot', 'value')
#     ]
# )
# def plot_graph(file_name_children, x_ft, y_ft, cat_Bi_plot):
#     file_name = file_name_children[-1]
#     fig = {}
#     if not (file_name == 'NA'):
#         raw_data_uploaded = pd.read_csv(f'./data/raw/{file_name}')
#         if cat_Bi_plot == 'Scatter':
#             fig = px.scatter(raw_data_uploaded, x=x_ft, y=y_ft)
#             return fig
#         else:
#             return fig
#     else:
#         return fig

# # MODAL - 3  ------> Num-Uni


# @callback(
#     Output("modal-fs-num-uni", "is_open"),
#     Input("open-fs-num-uni", "n_clicks"),
#     State("modal-fs-num-uni", "is_open"),
# )
# def toggle_modal(n, is_open):
#     if n:
#         return not is_open
#     return is_open


# @callback(
#     Output("modal-graph-num-uni", "figure"),
#     [
#         Input('eda-uploaded-file-name', 'children'),
#         Input('num-uni-ft', 'value'),
#         Input('num-uni-plot', 'value')
#     ]
# )
# def plot_graph(file_name_children, num_univ_ft, num_univ_plot):
#     file_name = file_name_children[-1]
#     fig = {}
#     if not (file_name == 'NA'):
#         raw_data_uploaded = pd.read_csv(f'./data/raw/{file_name}')
#         raw_data_uploaded = raw_data_uploaded.dropna().reset_index(drop=True)
#         if num_univ_plot == 'Histogram':
#             fig = px.histogram(raw_data_uploaded, x=num_univ_ft)
#             return fig

#         elif num_univ_plot == 'Box':
#             fig = px.box(raw_data_uploaded, y=num_univ_ft)
#             return fig

#         elif num_univ_plot == 'Violin':
#             fig = px.violin(raw_data_uploaded, y=num_univ_ft, box=True,  # draw box plot inside the violin
#                             points='all',  # can be 'outliers', or False
#                             )
#             return fig

#         elif num_univ_plot == 'Ecdf':
#             fig = px.ecdf(raw_data_uploaded, x=num_univ_ft)

#         else:
#             return fig
#     else:
#         return fig

# # MODAL - 4  ----------> Num-Bi


# @callback(
#     Output("modal-fs-num-Bi", "is_open"),
#     Input("open-fs-num-Bi", "n_clicks"),
#     State("modal-fs-num-Bi", "is_open"),
# )
# def toggle_modal(n, is_open):
#     if n:
#         return not is_open
#     return is_open


# @callback(
#     Output("modal-graph-num-Bi", "figure"),
#     [
#         Input('eda-uploaded-file-name', 'children'),
#         Input('num-Bi-X', 'value'),
#         Input('num-Bi-y', 'value'),
#         Input('num-Bi-plot', 'value')
#     ]
# )
# def plot_graph(file_name_children, x_ft, y_ft, num_Bi_plot):
#     file_name = file_name_children[-1]
#     fig = {}
#     if not (file_name == 'NA'):
#         raw_data_uploaded = pd.read_csv(f'./data/raw/{file_name}')
#         if num_Bi_plot == 'Histogram':
#             fig = px.histogram(raw_data_uploaded, x=x_ft, y=y_ft)
#             return fig

#         elif num_Bi_plot == 'Line':
#             fig = px.line(raw_data_uploaded, x=x_ft, y=y_ft)
#             return fig

#         elif num_Bi_plot == 'Scatter':
#             fig = px.scatter(raw_data_uploaded, x=x_ft, y=y_ft)
#             return fig

#         elif num_Bi_plot == 'Density heat map':
#             fig = px.density_heatmap(
#                 raw_data_uploaded, x=x_ft, y=y_ft, text_auto=True)
#             return fig

#         else:
#             return fig
#     else:
#         return fig

# # MODAL - 5  ---> NUM Multi


# @callback(
#     Output("modal-fs-num-multi", "is_open"),
#     Input("open-fs-num-multi", "n_clicks"),
#     State("modal-fs-num-multi", "is_open"),
# )
# def toggle3_modal(n, is_open):
#     if n:
#         return not is_open
#     return is_open


# @callback(
#     Output("modal-graph-num-multi", "figure"),
#     [
#         Input('eda-uploaded-file-name', 'children'),
#         Input('num-multi-ft', 'value'),
#         Input('num-multi-plot', 'value')
#     ]
# )
# def plot_multi_graph(file_name_children, ft_li, num_multi_plot):
#     file_name = file_name_children[-1]
#     fig = {}
#     if not (file_name == 'NA'):
#         raw_data_uploaded = pd.read_csv(f'./data/raw/{file_name}')
#         if num_multi_plot == 'Scatter':
#             fig = px.scatter_matrix(
#                 raw_data_uploaded, dimensions=ft_li)

#         #     fig.update_layout(
#         #     title_y=0.3,
#         #     title_x=0.3,
#         # )
#             return fig
#         else:
#             return fig
#     else:
#         return fig


# # NULL MODAL
# @callback(
#     Output("modal-fs-null", "is_open"),
#     Input("open-fs-null", "n_clicks"),
#     State("modal-fs-null", "is_open"),
# )
# def toggle2_modal(n, is_open):
#     if n:
#         return not is_open
#     return is_open


# @callback(
#     Output("modal-graph-null", "figure"),
#     [
#         Input('eda-uploaded-file-name', 'children')
#     ]
# )
# def plot_null_1(file_name_children):
#     file_name = file_name_children[-1]
#     fig = {}
#     if not (file_name == 'NA'):
#         raw_data_uploaded = pd.read_csv(f'./data/raw/{file_name}')
#         fig = px.bar(raw_data_uploaded.isna().sum().reset_index(
#             name="count"), x='index', y='count')
#         return fig
#     else:
#         return fig


# # Corr MODAL
# @callback(
#     Output("modal-fs-corr", "is_open"),
#     Input("open-fs-corr", "n_clicks"),
#     State("modal-fs-corr", "is_open"),
# )
# def eda_toggle_modal(n, is_open):
#     if n:
#         return not is_open
#     return is_open


# @callback(
#     Output("modal-graph-corr", "figure"),
#     [
#         Input('eda-uploaded-file-name', 'children')
#     ]
# )
# def plot_null_2(file_name_children):
#     file_name = file_name_children[-1]
#     fig = {}
#     if not (file_name == 'NA'):
#         raw_data_uploaded = pd.read_csv(f'./data/raw/{file_name}')
#         # Correlation
#         df_corr = raw_data_uploaded.corr().round(1)
#         # Mask to matrix
#         mask = np.zeros_like(df_corr, dtype=bool)
#         mask[np.triu_indices_from(mask)] = True
#         # Viz
#         df_corr_viz = df_corr.mask(mask).dropna(
#             how='all').dropna('columns', how='all')
#         fig = px.imshow(df_corr_viz, text_auto=True)

#         fig.update_layout(
#             title_text='Heatmap',
#             title_x=0.5,
#             width=1000,
#             height=700,
#             xaxis_showgrid=False,
#             yaxis_showgrid=False,
#             xaxis_zeroline=False,
#             yaxis_zeroline=False,
#             yaxis_autorange='reversed',
#             template='plotly_white'
#         )
#         return fig
#     else:
#         return fig
