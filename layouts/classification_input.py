from dash import Input, Output, State, html, dcc

from .sub_components.data_transform.dt_layout import data_transform_layout
from .sub_components.feature_selection.fs_layout import feature_selection_layout
from .sub_components.baseline_modeling.bm_layout import baseline_modeling_layout

from src.dash_utils.config_input import ml_config, bm_conf

from server import app

import json

import dash_bootstrap_components as dbc


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

# --------------------------------------------- ## HYPER-PARAMETER TUNING CALLBACK ## --------------------------------------------- #
