import dash_bootstrap_components as dbc

from dash import html, Input, Output, State
from assets.dash_css.side_menu_css import CONTENT_STYLE, SIDE_MENU_FOOTER, SIDE_MENU_NAVLINK_SPAN_STYLE

from server import app


SideMenuFooter = html.Div(
    children=[
        dbc.Row(
            children=[
                dbc.Col([
                    html.P(
                        'Copyright © 2023 HowToBeBoring.com. All rights reserved.',
                        className='text-center'
                    )
                ])
            ], justify='center'
        ),
    ], style=SIDE_MENU_FOOTER
)

SideMenuContent = html.Div([
    html.Div(
        children=[
            html.H1('PY-MLXL', style={'color': 'dark'}),
            html.H6('Machine Learning Accelerator to Automated and '
                    'Speed Up Model Building and Visualizing Process.')
        ], className='sidebar-header'
    ),

    html.Hr(),

    dbc.Nav(
        [
            dbc.NavLink([
                html.Div([
                    html.I(className='fa-solid fa-house'),
                    html.Span('Home Page (Dashboard)',
                              style=SIDE_MENU_NAVLINK_SPAN_STYLE)
                ])
            ], href='/'),

            html.Hr(),

            html.Div([
                html.Span('ML Sub-Category (Supervised Learning)')
            ], style={'margin-bottom': '10px'}),

            html.Div([
                html.Ul(
                    children=[
                        html.Li(
                            dbc.NavLink([
                                html.Div([
                                    html.I(className='fa-solid fa-table'),
                                    html.Span('Classification',
                                        style=SIDE_MENU_NAVLINK_SPAN_STYLE)
                                ])
                            ], href='/classification-details')
                        ),
                        html.Li(
                            dbc.NavLink([
                                html.Div([
                                    html.I(className='fa-solid fa-chart-area'),
                                    html.Span('Regression',
                                        style=SIDE_MENU_NAVLINK_SPAN_STYLE)
                                ])
                            ], href='/regression-details')
                        ),
                        html.Li(
                            dbc.NavLink([
                                html.Div([
                                    html.I(
                                        className='fa-solid fa-network-wired'),
                                    html.Span('Deep Learning',
                                              style=SIDE_MENU_NAVLINK_SPAN_STYLE)
                                ])
                            ], href='/deep-learning-details')
                        )
                    ]
                )
            ]),

            html.Hr(),

            html.Div([
                html.I(className='fa-solid fa-envelope'),
                html.Span('Current Status Table',
                          style=SIDE_MENU_NAVLINK_SPAN_STYLE)
            ])
        ], vertical=True
    ),
], className='sidebar',)


SideMenuButton = html.Div(
    [
        dbc.Button(
            children=[
                html.I(className='fa-solid fa-bars'),
                html.Span('Open Menu', style=SIDE_MENU_NAVLINK_SPAN_STYLE)
            ],
            id='open-side-menu-btn-navbar', n_clicks=0, size='sm', color='secondary'
        ),

        dbc.Offcanvas(
            children=[
                SideMenuContent, SideMenuFooter
            ], id='side-menu-offcanvas-content', scrollable=True, is_open=False, placement='end',
        ),
    ]
)


NavBar_Layout = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink(
                children=[html.Span('Home')], href='/', style={'margin-top': '3px'})),

        dbc.NavItem(
            dbc.NavLink(
                children=[html.Span('About This Tool')], href='/about-page', style={'margin-top': '3px'})),

        dbc.NavItem(dbc.NavLink(
                    children=[html.Span('EDA Tool')], href='/eda-tool', style={'margin-top': '3px'})),

        dbc.NavItem(dbc.NavLink(
                    children=[html.Span('KPI Visualization')], href='#', style={'margin-top': '3px'}), style={'margin-right': '20px'}),

        dbc.NavItem(dbc.NavLink(
                    children=[SideMenuButton]))

    ],

    brand=[
        html.Span(
            'PY-MLXL', style={'font-size': '1.2rem', 'margin-right': '10px'}),
        html.Span('Powered by: HowToBeBoring.com',
                  style={'font-size': '0.7rem'})
    ],

    color='dark', dark=True, brand_href='/', brand_style={'font-weight': 'bold', 'margin-left': '20px'}, fluid=True, sticky=True,

)

UI_Content_Layout = html.Div(id='page-content', style=CONTENT_STYLE)

# --------------------------------------------- ## CALLBACK METHODS ## --------------------------------------------- #


@app.callback(
    Output('side-menu-offcanvas-content', 'is_open'),
    Input('open-side-menu-btn-navbar', 'n_clicks'),
    State('side-menu-offcanvas-content', 'is_open'),
)
def toggle_side_menu_state(open_btn: int, side_menu_is_open_bool: bool) -> bool:
    """Toggles the Opening and Closing of the Side Menu through the click of the button on the Navbar.

    Args:
        open_btn (int): Button Click Event.
        side_menu_is_open_bool (bool): Side Menu Current State.

    Returns:
        bool: Reverse the Side Menu Current State.
    """

    if open_btn:
        return not side_menu_is_open_bool
    return side_menu_is_open_bool
