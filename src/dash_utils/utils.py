from dash import html

import base64
import datetime
import io
import joblib

import pandas as pd


class UtilityTools:

    @staticmethod
    def parse_contents(upload_contents: str, file_name: str, process_name: str, file_category: str = '') -> object:
        """Parses Contents from the Uploaded File to show file details.

        Args:
            upload_contents (str): FIle Contents.
            file_name (str): Uploaded File Name.
            process_name (str): Name of the Sub Module.

        Returns:
            object: File Decription.
        """

        _, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)

        raw_data_uploaded = pd.DataFrame()

        try:
            if 'csv' in file_name.lower():
                raw_data_uploaded = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))

            raw_data_uploaded.to_csv(f'./data/raw/{file_name}', index=False)

        except Exception as E:
            return html.Div([f'There was an error processing this file {E}.'])

        date_uploaded = datetime.datetime.now().strftime('%d %B, %Y %H:%M:%S')

        file_meta_data_layout = html.Div([
            html.Span(['File Name: ', file_name],
                      id=f'{process_name}-uploaded-file-name{file_category}'),
            html.Br(),
            html.Span(
                ['Date Uploaded: ', date_uploaded])
        ])

        return file_meta_data_layout
