import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc


# Connecting to the main app.py file
from app import app
from app import server


# Connecting to different app pages from the
from apps import playerdetails, playersearch



# Putting in the main layout


app.layout = dbc.Container([

                            dbc.Row([
                            dbc.Col([html.H1("Era Adjustment Project",className = "text-center mb-4"),
                                    html.Hr(),

                                    # Adding the links here
                                    dbc.Row([
                                    dcc.Link('Player Search |', href = '/apps/playerSearch', style = {'font-size':'20px'}),
                                    dcc.Link(' Player Details', href = '/apps/playerDetails', style = {'font-size':'20px'})
                                    ],justify = "center"),
                                    html.Br(),
                                    dbc.Row([
                                     html.Label(["Baseball Player Search",
                                                dcc.Dropdown(id = "player_name_det",
                                                             options = playersearch.dedup_player_dict_list,
                                                             value = 'Nap Lajoie',
                                                             persistence = True,
                                                             persistence_type = 'session',
                                                             clearable = True)
                                                ], style = {'width':'80%', 'font-size': '15px'})
                                    ],justify = "center"),

                                    dcc.Location(id='url', refresh = False),
                                    html.Div(id='content',children = [])
                                    ])
                                ])
                            ])


@app.callback(Output('content','children'),
              [Input('url','pathname')])
def display_page(pathname):
    if pathname == '/apps/playerSearch':
        return playersearch.layout
    if pathname == '/apps/playerDetails':
        return playerdetails.layout
    else:
        return "404 Page Error! Please Choose a link"


if __name__ == '__main__':
    app.run_server(debug=True)
