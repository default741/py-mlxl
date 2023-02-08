from dash import html, dcc
import dash_bootstrap_components as dbc


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
