from dash import html
import dash_bootstrap_components as dbc

import base64
import datetime
import io

import pandas as pd


def generate_card_object(
    card_img_src: str, card_header: tuple[str], card_footer: tuple[str], card_title: tuple[str], card_desc: tuple[str],
    card_btn: tuple[str], card_style: dict
) -> dbc.Card:
    """Generates a Dash Bootstrap Component Card with image, title, description and button link to got to the next page.

    Args:
        card_img_src (str): Card image Source
        card_header (tuple[str]): Card Header, Classname (idx: 0) and id (idx: 1)
        card_footer (tuple[str]): Card Footer, Classname (idx: 0) and id (idx: 1)
        card_title (tuple[str]): Card Title for HTML H4 tag, Content (idx: 0), Classname (idx: 1) and id (idx: 2)
        card_desc (tuple[str]): Card Description for HTML P tag, Content (idx: 0), Classname (idx: 1) and id (idx: 2)
        card_btn (tuple[str]): Card Button wrapped inside HTML A tag, Button Text (idx: 0), Color (idx: 1), HREF (idx: 2), Classname (idx: 3) and id (idx: 4)
        card_style (dict): CSS Style for the Card

    Returns:
        dbc.Card: Card Object
    """

    return dbc.Card(
        [
            dbc.CardImg(src=card_img_src, top=True),
            dbc.CardBody([
                dbc.CardHeader(
                    [
                        html.H4(
                            card_title[0], className=card_title[1], id=card_title[2]),
                        html.P(
                            card_desc[0], className=card_desc[1], id=card_desc[2])
                    ],
                    className=card_header[0], id=card_header[1]
                ),

                dbc.CardFooter(
                    [
                        html.A(dbc.Button(
                            card_btn[0], color=card_btn[1]), href=card_btn[2], className=card_btn[3], id=card_btn[4])
                    ],
                    className=card_footer[0], id=card_footer[1]
                )
            ])
        ],
        style=card_style,
    )


def parse_contents(upload_contents, file_name, process_name):
    _, content_string = upload_contents.split(',')
    decoded = base64.b64decode(content_string)

    raw_data_uploaded = pd.DataFrame()

    try:
        if 'csv' in file_name:
            raw_data_uploaded = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

        raw_data_uploaded.to_csv(f'./data/raw/{file_name}', index=False)

    except Exception as E:
        return html.Div([f'There was an error processing this file {E}.'])

    date_modified = datetime.datetime.now().strftime('%d %B, %Y %H:%M:%S')

    file_meta_data_page = html.Div(
        [
            html.Span(['File Name: ', file_name],
                      id=f'{process_name}-uploaded-file-name'),
            html.Br(),
            html.Span(
                ['Date Uploaded: ', date_modified])
        ], id=f'{process_name}-uploaded-file-meta-data', style={'margin-top': '10px'}
    )

    return file_meta_data_page
