from dash import html
import dash_bootstrap_components as dbc

from assets.dash_css.dashboard_css import DASHBOARD_CARD_STYLE


card_object_classification = dbc.Card(
    children=[
        dbc.CardImg(
            src='/assets/images/classification-logo-dashboard.jpeg', top=True),
        dbc.CardBody([
            dbc.CardHeader(
                children=[
                    html.H4(
                        'Classification', id='card-title-classification'),
                    html.P(
                        'Classification is the process of predicting the class of given data points.', id='card-desc-classification')
                ], id='dashboard-card-header-classification'
            ),

            dbc.CardFooter(
                children=[
                    html.A(dbc.Button(
                        'More Infomation...', color='secondary'), href='/classification-details', id='card-btn-classification')
                ], id='dashboard-card-footer-classification'
            )
        ])
    ], style=DASHBOARD_CARD_STYLE
)

card_object_regression = dbc.Card(
    children=[
        dbc.CardImg(
            src='/assets/images/classification-logo-dashboard.jpeg', top=True),
        dbc.CardBody([
            dbc.CardHeader(
                children=[
                    html.H4(
                        'Regression', id='card-title-regression'),
                    html.P(
                        'Classification is the process of predicting the class of given data points.', id='card-desc-regression')
                ], id='dashboard-card-header-regression'
            ),

            dbc.CardFooter(
                children=[
                    html.A(dbc.Button(
                        'More Infomation...', color='secondary'), href='/regression-details', id='card-btn-regression')
                ], id='dashboard-card-footer-regression'
            )
        ])
    ], style=DASHBOARD_CARD_STYLE
)

card_object_deep_learning = dbc.Card(
    children=[
        dbc.CardImg(
            src='/assets/images/classification-logo-dashboard.jpeg', top=True),
        dbc.CardBody([
            dbc.CardHeader(
                children=[
                    html.H4(
                        'Deep Learning', id='card-title-deep-learning'),
                    html.P(
                        'Classification is the process of predicting the class of given data points.', id='card-desc-deep-learning')
                ], id='dashboard-card-header-deep-learning'
            ),

            dbc.CardFooter(
                children=[
                    html.A(dbc.Button(
                        'More Infomation...', color='secondary'), href='/deep-learning-details', id='card-btn-deep-learning')
                ], id='dashboard-card-footer-deep-learning'
            )
        ])
    ], style=DASHBOARD_CARD_STYLE
)


ml_categories_card_layout = html.Div(
    dbc.CardGroup(
        [card_object_classification, card_object_regression, card_object_deep_learning]
    )
)


app_dashboard_title_layout = html.Div(
    dbc.Container(
        [
            html.H1("Jumbotron", className="display-3"),
            html.P(
                "Use Containers to create a jumbotron to call attention to "
                "featured content or information.",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(
                "Use utility classes for typography and spacing to suit the "
                "larger container."
            ),
            html.P(
                dbc.Button("Learn more", color="primary"), className="lead"
            ),
        ],
        fluid=True,
        className="py-3",
    ),
    className="p-3 bg-light rounded-3",
)


Dashboard_Layout = html.Div([
    app_dashboard_title_layout,
    ml_categories_card_layout
])
