import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from layouts.dashboard import dashboard_layout
from layouts.side_menu import sidebar, content, offcanvas, navbar
from layouts.classification_input import classification_layout, accordion

from server import app

dcc.Store(id='session', storage_type='session')
app.title = "MLXL Powered by HowToBeBoring.com"

app.layout = html.Div(
    [
        navbar,
        dbc.Container(
            className="p-5 dashboard-main-wrapper",
            fluid=True,
            children=[
                dcc.Location(id="url"),
                content,
            ]
        )
    ]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return dashboard_layout
    elif pathname == "/about-page":
        return html.P("This is the content of about-page. Yay!")
    elif pathname == "/contact-page":
        return html.P("Oh cool, this is contact-page!")

    elif pathname == "/classification-details":
        return accordion

    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    # app.run_server(debug=True, dev_tools_ui=False,
    #    host='0.0.0.0', dev_tools_props_check=False)

    app.run_server(debug=True)
