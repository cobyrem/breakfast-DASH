import sqlite3
import webbrowser
from tabulate import tabulate
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

con = sqlite3.connect('reids.db')
cursor = con.cursor()

# Query 1 take the proportion of each item
df1 = pd.read_sql_query("""
SELECT
ITEMS.NAME,
COUNT(*) * 1.0 / (SELECT COUNT(*) FROM ITEMS) AS PERCENTAGE
FROM ITEMS
GROUP BY ITEMS.NAME 
""",con)
print(df1)

# Query 2 Get daily gross revenue for each date.
df2 = pd.read_sql_query("""SELECT
FULL_DATE,
COUNT(*) as "DAILY_ORDER_COUNT", 
SUM("TOTAL") as "DAILY_GROSS_REVENUE"
FROM CHARGES
GROUP BY "FULL_DATE"
""",con)
print(df2)

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Reids Breafast-DASH(board)', style={'textAlign': 'center', 'color': '#008080'}),
    html.Div([
        dcc.Checklist(options=[{'label': UP_CHARGE, 'value': UP_CHARGE} for UP_CHARGE in df1.NAME.unique()],inline = True,value=df1.NAME.unique(), id='dropdown-selection1'),
        dcc.Graph(id='graph-content1'),html.P('Future feature might include dynamically updating proportions depending on check marks. Not a very effective bar chart without the this feature. Dashboarding presents new issues compared to more static charts. Added to issues on GitHub')]
        )
])

@callback(
    Output('graph-content1', 'figure'),
    Input('dropdown-selection1', 'value'),
)
def update_graph(upcharges):
    dff1 = df1[df1.NAME.isin(upcharges)]
    fig1 = px.bar(dff1, x='NAME', y='PERCENTAGE',title='Proportion of item sales')
    fig1.update_layout(title_x=0.5)
    return fig1

if __name__ == '__main__':

    app.run(debug= False)
    webbrowser.open_new("http://localhost:8050")  # Change the port if needed
