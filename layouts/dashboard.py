from dash import html
import dash_bootstrap_components as dbc

from src.dash_utils.utils import generate_card_object


card_object_classification = generate_card_object(
    card_img_src='/assets/images/classification-logo-dashboard.jpeg',
    card_header=('dashboard-card-header',
                 'dashboard-card-header-classification'),
    card_footer=('dashboard-card-footer',
                 'dashboard-card-footer-classification'),
    card_title=('Classification', 'card-title', 'card-title-classification'),
    card_desc=('Classification is the process of predicting the class of given data points.',
               'card-desc', 'card-desc-classification'),
    card_btn=('Details', 'primary', '/classification-details',
              'card-btn', 'card-btn-classification'),
    card_style={"width": "18rem", 'margin': '2rem', 'border': '5px'}
)

card_object_regression = generate_card_object(
    card_img_src='/assets/images/classification-logo-dashboard.jpeg',
    card_header=('dashboard-card-header',
                 'dashboard-card-header-regression'),
    card_footer=('dashboard-card-footer',
                 'dashboard-card-footer-regression'),
    card_title=('Regression', 'card-title', 'card-title-regression'),
    card_desc=('Classification is the process of predicting the class of given data points.',
               'card-desc', 'card-desc-regression'),
    card_btn=('Details', 'primary', 'https://google.com',
              'card-btn', 'card-btn-regression'),
    card_style={"width": "18rem", 'margin': '2rem', 'border': '5px'}
)

card_object_deep_learning = generate_card_object(
    card_img_src='/assets/images/classification-logo-dashboard.jpeg',
    card_header=('dashboard-card-header',
                 'dashboard-card-header-deep-learning'),
    card_footer=('dashboard-card-footer',
                 'dashboard-card-footer-deep-learning'),
    card_title=('Deep Learning', 'card-title', 'card-title-deep-learning'),
    card_desc=('Classification is the process of predicting the class of given data points.',
               'card-desc', 'card-desc-deep-learning'),
    card_btn=('Details', 'primary', 'https://google.com',
              'card-btn', 'card-btn-deep-learning'),
    card_style={"width": "18rem", 'margin': '2rem', 'border': '5px'}
)


card_object = html.Div(
    dbc.CardGroup(
        [card_object_classification, card_object_regression, card_object_deep_learning]
    )
)

jumbotron = html.Div(
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


dashboard_layout = html.Div([
    jumbotron,
    card_object
])
