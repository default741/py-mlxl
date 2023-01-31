import dash_bootstrap_components as dbc
from dash import html, Input, Output, State

from server import app

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "2rem",
    "margin-right": "2rem",
    "margin-top": "2rem",
}

footer = html.Div(
    dbc.Row(
        dbc.Col(
            html.P(
                "Copyright © 2021 Your Company Name. All rights reserved.",
                className="text-center"
            )
        )
    ),
    style={"position": "fixed", "bottom": "0", 'font-size': '0.75rem'}
)

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button(
                "Search", color="primary", className="ms-2", n_clicks=0
            ),
            width="auto",
        ),
    ],
    className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

service_accordion_layout = html.Div(
    dbc.Accordion([
        dbc.AccordionItem([
            dbc.NavLink([
                html.I(
                    className="fas fa-calendar-alt me-2"), html.Span("Classification"),
            ], href="/classification-details", active="exact"),

            dbc.NavLink([
                html.I(
                    className="fas fa-calendar-alt me-2"), html.Span("Regression"),
            ], href="/regression-details", active="exact"),

            dbc.NavLink([
                html.I(
                    className="fas fa-calendar-alt me-2"), html.Span("Deep Learning"),
            ], href="/deep-learning-details", active="exact"),
        ], title="Service Category")
    ], flush=True, start_collapsed=True),
)

sidebar = html.Div([
    html.Div([
        html.H2("MLXL", style={"color": "white"}),
    ], className="sidebar-header"),
    html.Hr(),
    dbc.Nav([
        dbc.NavLink([
            html.I(className="fas fa-home me-2"), html.Span("Dashboard")
        ], href="/", active="exact"),

        dbc.NavLink([
            html.I(
                className="fas fa-calendar-alt me-2"), html.Span(service_accordion_layout)
        ], href="/", active="exact"),

        dbc.NavLink([
            html.I(className="fas fa-envelope-open-text me-2"),
            html.Span("Datasets"),
        ], href="/datasets", active="exact")

    ], vertical=True, pills=True),
], className="sidebar",)


offcanvas = html.Div(
    [
        dbc.Button(
            [
                html.I(className="fas fa-home me-2"), html.Span("Menu")
            ],
            id="open-offcanvas-scrollable",
            n_clicks=0,
        ),
        dbc.Offcanvas(
            [sidebar, footer],
            id="offcanvas-scrollable",
            scrollable=True,
            title="Scrollable Offcanvas",
            is_open=False,
            placement='end'
        ),
    ]
)


navbar = dbc.NavbarSimple(
    [
        dbc.NavItem(dbc.NavLink(
                    [html.I(className='fa-solid fa-user-tie'), html.Span(' About This Tool')], href="#")),
        dbc.NavItem(dbc.NavLink(
                    [html.I(className='fa-solid fa-user-tie'), html.Span(' Profile')], href="#")),
        dbc.NavItem(dbc.NavLink(
                    [offcanvas], href="#"))

    ],
    color="white",
    dark=False,
    brand='MLXL',
    style={'margin-left': '10px', 'margin-right': '25px'}
)


@app.callback(
    Output("offcanvas-scrollable", "is_open"),
    Input("open-offcanvas-scrollable", "n_clicks"),
    State("offcanvas-scrollable", "is_open"),
)
def toggle_offcanvas_scrollable(n1, is_open):
    if n1:
        return not is_open
    return is_open


content = html.Div(id="page-content", style=CONTENT_STYLE)
