import sqlite3
import webbrowser
from dash import Dash, html, dcc, callback, Output, Input, dash_table
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


app = Dash(__name__)
app.scripts.config.serve_locally = True
application = app.server

app.layout = html.Div([
    html.H1(children='Reids Breakfast-DASH(board)', style={'textAlign': 'center', 'color': '#008080'}),
    html.Div([dash_table.DataTable(df1.to_dict('records'), [{"name": i, "id": i} for i in df1.columns])]),
    html.Div([
        dcc.Checklist(options=[{'label': UP_CHARGE, 'value': UP_CHARGE} for UP_CHARGE in df1.NAME.unique()],inline = True,value=df1.NAME.unique(), id='dropdown-selection1'),
        dcc.Graph(id='graph-content1'),html.P('Future feature might include dynamically updating proportions depending on check marks. Not a very effective bar chart without the this feature. Dashboarding presents new issues compared to more static charts. Added to issues on GitHub')]
        ),
    html.Div([dcc.Graph(id='graph-content2')])
])

@callback(
    Output('graph-content1', 'figure'),
    Output('graph-content2', 'figure'),
    Input('dropdown-selection1', 'value'),
)
def update_graph(upcharges):
    dff1 = df1[df1.NAME.isin(upcharges)]
    fig1 = px.bar(dff1, x='NAME', y='PERCENTAGE',title='Proportion of item sales')
    fig1.update_layout(xaxis_title = 'Food and Drink Category', yaxis_title = 'Proportion of Sales', title_x=0.5)
    fig2 = px.line(df2,x = 'FULL_DATE', y = 'DAILY_GROSS_REVENUE', title = 'Daily Gross Revenue')
    fig2.update_layout(title_x=0.5, yaxis_title = 'DAILY_GROSS_REVENUE ($)', xaxis_title = "FULL_DATE")
    return fig1, fig2

if __name__ == '__main__':
    application.run(debug=False, port=8080)