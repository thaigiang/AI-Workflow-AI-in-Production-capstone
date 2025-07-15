import dash_core_components as dcc
import dash_html_components as html

from dashboards.utils.utils import Header, make_dash_table, country_names, country_mapping

def create_layout(app):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]),
            html.Div([
                html.H3("Prediction"),
                html.Div([
                    html.Div([
                        html.Div(
                            "Select country",
                            className="three columns"
                        ),
                        #html.Div(id="pred-country-message", className="three columns"),
                    ], className="row"),

                    dcc.Dropdown(
                        options=country_names,
                        value='all',
                        id="country-dropdown",
                        clearable=False
                    ),
                ]),

                html.Br([]),
                html.Div([
                    html.Div(html.H5("45-days prediction", id="pred-country-message")),
                    dcc.Graph(id="pred-plot")
                ])
            ]),
            html.Div(id='intermediate-value', style={'display': 'none'})
        ]
    )

