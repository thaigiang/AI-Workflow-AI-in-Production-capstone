import os
from ast import literal_eval
import pandas as pd

import dash_html_components as html
import dash_core_components as dcc

from application.utils.ingestion import get_country_names
from application.model import model_load, model_train

# Layout utils
def Header(app):
    return html.Div([get_header(app), get_menu()], className="row")


def get_header(app):
    header = html.Div(html.H1("AAVAIL"), className="six columns")
    return header


def get_menu():
    menu = html.Div(
        [
            html.Div(className="five columns"),
            html.Div(html.H4(dcc.Link(
                "Prediction",
                href="/",
            )), className="two columns"),
            html.Div(html.H4(dcc.Link(
                "Monitor",
                href="/monitor/",
            )), className="two columns")
        ],
        className="row"
    )
    return menu


def make_dash_table(df, header=False):
    """ Return a dash definition of an HTML table for a Pandas dataframe """
    table = []
    if header:
        html_row = []
        for col in df.columns:
            html_row.append(html.Td([col]))
        table.append(html.Tr(html_row))
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table


def read_metrics_from_log(logdir=None):
    if not logdir:
        logdir = os.path.join(".", "logs")

    # Find 
    logfiles = [file for file in os.listdir(os.path.join(".", "logs")) if file.endswith('.log') and\
                                    file.startswith('train') and 'test' not in file]
    years = [int(s.split("-")[1]) for s in logfiles]
    last_year = sorted(years)[-1]
    months = [int(s.split(".")[0].split("-")[-1]) for s in logfiles if str(last_year) in s.split("-")]
    last_month = sorted(months)[-1]

    filename = "train-{}-{}.log".format(last_year, last_month)

    df = pd.read_csv(os.path.join(logdir, filename), delimiter=",").\
            sort_values(by="timestamp", ascending=True).\
            groupby("country").tail(1)[['country', 'eval_metric']]
    df['RMSE'] = df['eval_metric'].apply(lambda x : literal_eval(x)['rmse'])
    df.drop(columns=['eval_metric'], inplace=True)
    df.rename(columns={'country':'Country'},inplace=True)

    return df

# General data structures
try:
    data, models = model_load()
except:
    model_train()
    data, models = model_load()
    
# Country to identifier mappings
country_mapping = {'all':'All', **get_country_names()}
country_names = []

for key,country in country_mapping.items():
    if key in data.keys():
        country_names.append({'label':country,'value':key})

# Monitoring metrics
metrics = read_metrics_from_log()
metrics['Country'] = metrics['Country'].apply(lambda x : country_mapping[x]) 