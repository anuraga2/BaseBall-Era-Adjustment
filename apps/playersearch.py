import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import base64
import dash_table
import pathlib
from dash.exceptions import PreventUpdate
from app import app


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()


## Reading the data required for players bio and summary statistics
player_bio = pd.read_csv(DATA_PATH.joinpath("bio_data.csv"))
player_summary = pd.read_csv(DATA_PATH.joinpath("summary_data.csv"))

# Reading the individual csv Files
batters_df = pd.read_csv(DATA_PATH.joinpath("raw_batter.csv"))
pitchers_df = pd.read_csv(DATA_PATH.joinpath("raw_pitcher.csv"))

# Getting the bio details
temp_bio = player_bio.loc[(player_bio['player_name'] == 'Nap Lajoie') & -pd.isnull(player_bio['bio_details']),['bio_header','bio_details']]

## Getting the player search options in place
player_search_options = [{"label":player, "value":player} for player in player_bio['player_name'].unique()]

# Finding out the distinct entries of players name and their IDs
batters_name_id = batters_df.loc[:,['Name','playerID']].drop_duplicates()
# Removing null values (batters)
batters_name_id = batters_name_id.loc[-batters_name_id['Name'].isnull(),]

# Finding out the distinct entries of players name and their IDs
pitchers_name_id = pitchers_df.loc[:,['Name','playerID']].drop_duplicates()
# Removing null values (pitchers)
pitchers_name_id = pitchers_name_id.loc[-pitchers_name_id['Name'].isnull(),]


# Creating pitchers and batters dict and then deduping it
pitchers_dict = [{'label': row['Name'], 'value' : row['Name']} for index, row in pitchers_name_id.iterrows()]
batters_dict = [{'label': row['Name'], 'value' : row['Name']} for index, row in batters_name_id.iterrows()]

dedup_player_dict_list = [dict(t) for t in {tuple(d.items()) for d in (pitchers_dict + batters_dict)}]

### Functions that are used across the dashboard
# Function to encode an image file
def encode_image(image_file):
    encoded = base64.b64encode(open(image_file,'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())


def transpose_data(player_name):
    summ = player_summary.loc[(player_summary['player_name'] == player_name) & -pd.isnull(player_summary['player_stats_value']),['player_stats','player_stats_value']]
    summ.columns = ['Summary','Career']
    summ = summ.set_index('Summary')
    summ = summ.T
    for item in list(summ.columns):
        summ[item]=summ[item].map("{:,.3f}".format)
    return(summ)



layout = html.Div(id = "player_search_content",
        children = [
            dbc.Row([

            html.P("Baseball Player Search:", style = {'font-size':'20px'}),
            html.Br(),
            html.Label([dcc.Dropdown(id = 'player_det_search',
                                     value = 'Nap Lajoie',
                                     persistence = True,
                                     persistence_type = 'session')], style = {'width':'100%'}),
            html.Br(),
            html.Br(),
            # This section contains the code for player images, summary and bio
            html.Div([
                            dbc.Row([

                                    # This column contains the player Image
                                    dbc.Col([
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),
                                    dbc.Row([html.Img(id = "player_search_image",src='children',height=250)],justify = "center"),
                                    html.Br(),
                                    dbc.Row([html.P(id = "player_img_txt",children ="Nap Lajoie", style = {'font-weight':'bold'})], justify = "center")
                                    ], width = 4),

                                    # This column contains the Player bio details and summary
                                    dbc.Col([
                                            dbc.Row([html.P("Bio: ", style = {'font-size':'20px'})]),
                                            dbc.Row([
                                            dash_table.DataTable(
                                            id = 'player_bio_table',
                                            columns = [{"name": "Attributes", "id": "bio_header"},
                                                       {"name": "Details", "id": "bio_details"}],
                                            data = temp_bio.to_dict('records'),
                                            style_cell={'textAlign': 'left'},
                                            style_data = {'width':'100px'}
                                            )
                                            ],justify = "left"),
                                            html.Br(),
                                            dbc.Row([html.P("Career Summary:", style = {'font-size':'20px'})]),
                                            dbc.Row([
                                            dash_table.DataTable(
                                            id = 'player_summary_table',
                                            columns = [{'name':'Summary', 'id':'Summary'},
                                                       {'name': 'WAR', 'id': 'WAR'},
                                                       {'name': 'AB', 'id': 'AB'},
                                                       {'name': 'H', 'id': 'H'},
                                                       {'name': 'HR', 'id': 'HR'},
                                                       {'name': 'BA', 'id': 'BA'}],
                                            data = transpose_data('Nap Lajoie').to_dict('records'),
                                            style_cell={'textAlign': 'left'},
                                            style_data = {'width':'100px'}
                                            )
                                            ], justify = "left")
                                    ], width = 8)

                            ])
            ])

            ]),
            html.Br(),
            dcc.Link(
            dbc.Button("Click here to visit the player stats page", color="success", id="player_det_tab_switch"),
            href = '/apps/playerDetails'
            ),
        ]
    )



#################### Call Backs for Drop downs player Search Tab (Starts here) #####################################
@app.callback(
    dash.dependencies.Output("player_det_search", "options"),
    [dash.dependencies.Input("player_det_search", "search_value")],
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    return [o for o in player_search_options if search_value in o["label"]]

# Callback for the players images
@app.callback(Output('player_search_image','src'),[Input("player_det_search","value")])
def callback_image(player_name):
    path = '../img_new/'+player_name+'.png'
    return encode_image(path)

# Callback for the players image names
@app.callback(Output('player_img_txt','children'),[Input("player_det_search","value")])
def image_footer_text(player_name):
    return(player_name)

# Callback for the player bio table
@app.callback(Output('player_bio_table','data'),[Input('player_det_search','value')])
def update_player_bio(player_name):
    df = player_bio.loc[(player_bio['player_name'] == player_name) & -pd.isnull(player_bio['bio_details']),['bio_header','bio_details']]
    return(df.to_dict('records'))

# Callback for the player summary table is below. This callback is broken down into two parts. The table header and the table body
## Header callback
@app.callback(Output('player_summary_table','columns'),[Input('player_det_search','value')])
def update_player_summ_header(player_name):
    summ = transpose_data(player_name)
    cols = [{'name':'Summary', 'id':'Summary'}]+[{'name':item, 'id':item} for item in list(summ.columns)]
    return(cols)

## player summary table callback
@app.callback(Output('player_summary_table','data'),[Input('player_det_search','value')])
def update_player_summ_header(player_name):
    summ = transpose_data(player_name)
    rec = summ.to_dict('records')[0]
    rec['Summary'] = 'Career'
    return([rec])


#################### Call Backs for Drop downs player Search Tab (Ends here) #####################################
