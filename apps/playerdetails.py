import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import base64
import dash_table
import pathlib
from dash.exceptions import PreventUpdate
from app import app
#import tooltipdata

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

# Dictionary for time selection
time_dict = [{'label': i, 'value' : i} for i in range(1871,2020)]

# Description for batting tooltip
batting_tooltip = {
                    'yearID':['Year of Play'],
                    'lgID':['Baseball League Id'],
                    'Name':['Player Name'],
                    'Team':['Team Name'],
                    'PA':['Plate Appearances'],
                    'AB':['At Bats'],
                    'HR':['Home Runs'],
                    'H':['Hits'],
                    'X2B':['Second Bases'],
                    'X3B':['Third Bases'],
                    'RBI':['Runs Batted In'],
                    'SB':['Stolen Bases'],
                    'ISO':['Isolated Power'],
                    'BABIP':['Batting Average on Balls in Play'],
                    'AVG':['Number of Hits divided by At Bats'],
                    'OBP': ['On Base Percentage'],
                    'SLG': ['Slugging Percentage'],
                    'wOBA': ['Weighted On Base Average'],
                    'wRC.': ['Weighted Runs Created Plus'],
                    'fWAR': ['Wins Above Replacement'],
                    'CS': ['Caught Stealing'],
                    'BB.': ['Base on Balls percentage'],
                    'K.': ['Strikeout percentage'],
                    'Off': ['Offense'],
                    'Def':['Defense']
}


############################################################################## Helper Functions (Start) ##############################################################################

# Function to encode an image file
def encode_image(image_file):
    encoded = base64.b64encode(open(image_file,'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())


# This function helps in getting the dynamic date ranges
def year_range(player_name):
    # Finding out the min year for each player
    min_year_bat = batters_df.loc[batters_df['Name'] == player_name,['yearID']].min()['yearID']
    min_year_pit = pitchers_df.loc[pitchers_df['Name'] == player_name,['yearID']].min()['yearID']

    # Finding out the maximum year for each player
    max_year_bat = batters_df.loc[batters_df['Name'] == player_name,['yearID']].max()['yearID']
    max_year_pit = pitchers_df.loc[pitchers_df['Name'] == player_name,['yearID']].max()['yearID']

    # based on whether a player is pitcher or batter deciding upon the min year
    if min_year_pit!=min_year_pit:
        min_year = min_year_bat
    elif min_year_bat!=min_year_bat:
        min_year = min_year_pit
    else:
        min_year = min(min_year_bat, min_year_pit)

    # based on whether a player is pitcher or batter deciding upon the min year
    if max_year_pit!=max_year_pit:
        max_year = max_year_bat
    elif max_year_bat!=max_year_bat:
        max_year = max_year_pit
    else:
        max_year = max(max_year_bat, max_year_pit)

    return(min_year, max_year)



# Function to remove categorical colunmns
def remove_categorical_columns(lst):
    if 'yearID' in lst:
        lst.remove('yearID')

    if 'Team' in lst:
        lst.remove('Team')

    if 'lgID' in lst:
        lst.remove('lgID')

    return(lst)

# Function to format colunmns
def format_cols(df,met, format_columns):
    bool_vec = [item in met for item in format_columns]
    check = any(bool_vec)
    if check:
        i = 0
        for item in bool_vec:
            if item:
                df[format_columns[i]] = df[format_columns[i]].round(3)
            i += 1
    return df


# Helper function for populating top batting/pitching metric
def top_metric(lst):
    col_list = []
    for item in lst:
        if item == 'Team':
            continue
        elif item == 'yearID':
            continue
        elif item == 'lgID':
            continue
        else:
            col_list.append(item)
    return col_list


# Helper function to select the top N rows by a particular metric
def top_n_metric(df,k,metric):
    """
    -- This function takes in three parameters. These are:
        * df: data frame
        * k: Window width of the duration in which the best performance is to be found out
        * metric: Metric on which the best performance is being calculated
    -- This function returns the top 'n' years for a player for any selected metric
    -- If the user doest not select the year column then this function will raise an exception
    -- and it will coerce the year field into the resultant dataset

    """
    try:
        # The if clause checks whether a correct value of k has been selected. If K is more than the number of rows in the
        # data set then it return the entire data set as it is. Same goes for K = 0
        if k == None:
            return df
        if k >= np.shape(df)[0] or k == 0:
            return df
        else:

            # For any other value of K this code chunk is executed. You can see that this code chunk requires the yearID
            # column to be present in the dataset
            # if the year id column is not there then this chunk will throw a key error
            # we are catching that error in the code chunk below

            df_new = df.loc[:,['yearID',metric]].groupby(['yearID']).sum().reset_index()
            start_new = df_new[metric].rolling(k).mean().dropna().idxmax()
            end_new = start_new + (k-1)
            window_start = df_new.loc[start_new,['yearID']]['yearID']
            window_end = df_new.loc[end_new,['yearID']]['yearID']
            return(df.loc[(df['yearID']>=window_start) &  (df['yearID']<= window_end),])

    except (KeyError,ValueError) as e :
        # If the user skips the yearID column from his/her selection then we coerece the yearID column after pullling
        # it from batters_df data set (batters_df) is a global variable

        year = batters_df.loc[batters_df['Name'] == df['Name'].unique()[0],['yearID']]
        print(year)
        # deleting the index column since we wont be needing it
        df['yearID'] = year['yearID']
        del year
        return top_n_metric(df,k,metric)

############################################################################## Helper Functions (End) ##############################################################################

############################################################################## Layout (Start) ######################################################################################

layout = html.Div(id = "player_details_content",children = [

                                 dbc.Row([

                                 # Content for column 1 will go here
                                 dbc.Col([
                                         html.P("Filters",style = {'font-size':'20px'}),
                                         html.Br(),

                                         # Adding the player category dropdown
                                         html.Label(["Player Name",
                                                    dcc.Dropdown(id = "player_name_det",
                                                                 options = dedup_player_dict_list,
                                                                 value = 'Nap Lajoie',
                                                                 clearable = True)
                                                    ], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["Start Year",
                                                    dcc.Dropdown(
                                                        id = 'start_year',
                                                        options = time_dict,
                                                        value = 1871,
                                                        clearable = False
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["End Year",
                                                    dcc.Dropdown(
                                                    id = 'end_year',
                                                    options = time_dict,
                                                    value = 2019,
                                                    clearable = False
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["Batting Metrics",
                                                    dcc.Dropdown(
                                                    id = 'batting_metrics',
                                                    multi = True,
                                                    value = ['yearID','Team','PA','AB','AVG','HR','H','X2B','X3B','RBI','SB','ISO','BABIP','fWAR']
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["Select Top N years",
                                                    dbc.Input(
                                                    id = 'batting_top_n',
                                                    type = "number",
                                                    placeholder = "type in a number"
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["By (Batting Metric):",
                                                    dcc.Dropdown(
                                                    id = 'top_batting_metric',
                                                    clearable = True
                                                    )
                                                   ], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["Pitching Metrics",
                                                    dcc.Dropdown(
                                                    id = 'pitching_metrics',
                                                    multi = True,
                                                    value = ['yearID','Team','IP','K.9','BB.9','HR.9','BABIP','ERA','FIP','WHIP','fWAR']
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["Select Top N years",
                                                    dbc.Input(
                                                    id = 'pitching_top_n',
                                                    type = "number",
                                                    placeholder = "type in a number"
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["By (Pitching Metric):",
                                                    dcc.Dropdown(
                                                    id = 'top_pitching_metric',
                                                    clearable = True
                                                    )
                                                   ], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Br(),
                                        dbc.Button("Submit Metrics to Update Table",color = "success",n_clicks = 0, id = 'submit_player_id')

                                 ], width = 4),


                                 # Content for column 2 will go here
                                 dbc.Col([
                                 html.Br(),
                                 html.Br(),
                                 html.Br(),
                                 html.Br(),
                                 dbc.Row([html.Img(id = "player_det_image",src="children",height=250)], justify="left"),
                                 html.Br(),
                                 dbc.Row([html.P("Nap Lajoie", id="player_img_name", style = {'font-weight':'bold'})]),
                                 html.Br(),
                                 ##dbc.Row([html.P(id="player_det_txt", style = {'font-weight':'bold'})],justify="left"),
                                 dbc.Row([
                                     html.Div(id="batting_det",
                                              children = [html.P("Batting",id="batting_txt", style = {'font-weight':'bold'}),
                                                          dash_table.DataTable(
                                                          id = 'bat_table',
                                                          style_cell={'textAlign': 'left',
                                                                      'whiteSpace':'normal',
                                                                      'height':'auto'},
                                                          style_data = {'width':'120px'},
                                                          #style_data_conditional=style_row_by_top_values(df),
                                                          sort_action="native",
                                                          sort_mode="multi",
                                                          style_table={'overflowX': 'auto'},
                                                          filter_action="native"
                                                                            ),
                                                          html.Br(),
                                                          html.P("Batting Averages",id="batting_average_txt", style = {'font-weight':'bold'})], style= {'display': 'block'}),
                                                          html.Br(),
                                                          dash_table.DataTable(
                                                          id = 'bat_avg_table',
                                                          style_cell = {
                                                          'textAlign':'left',
                                                          'whiteSpace':'normal',
                                                          'height':'auto'},
                                                          style_data = {'width':'120px'}
                                                          )
                                  ],justify="left"),
                                 html.Br(),
                                 dbc.Row([
                                     html.Div(id="pitching_det",
                                              children = [html.P("Pitching",id="pitching_txt", style = {'font-weight':'bold'}),
                                                          dash_table.DataTable(
                                                          id = 'pit_table',
                                                          style_cell={'textAlign': 'left'},
                                                          style_data = {'width':'120px'},
                                                          sort_action="native",
                                                          sort_mode="multi",
                                                          filter_action="native"
                                                          ),
                                                          html.Br(),
                                                          html.P("Pitching Averages",id="pitching_average_txt", style = {'font-weight':'bold'}),
                                                          html.Br(),
                                                          dash_table.DataTable(
                                                          id = 'pit_avg_table',
                                                          style_cell = {
                                                          'textAlign':'left',
                                                          'whiteSpace':'normal',
                                                          'height':'auto'},
                                                          style_data = {'width':'120px'}
                                                          )
                                                          ], style= {'display': 'block'})
                                  ],justify="left"),
                                 ], width = 8),

                                 ])
                    ])

############################################################################## Layout (End) ######################################################################################


#################### Player Details call backs start ###############################################

# Setting up the call back for updating the end year drop down start
@app.callback(Output("start_year","options"),[Input("player_name_det","value")])
def update_year_start(selection):
    min_year, max_year = year_range(selection)
    time_dict = [{'label': i, 'value' : i} for i in range(min_year,max_year+1)]
    return time_dict

# Setting up the call back for updating the end year drop down start
@app.callback(Output("start_year","value"),[Input("player_name_det","value")])
def update_year_start_def(selection):
    min_year, max_year = year_range(selection)
    return min_year

# Setting up the call back for updating the end year drop down start
@app.callback(Output("end_year","options"),[Input("start_year","value"),Input("player_name_det","value")])
def update_end_year_menu(year, player_name):
    min_year, max_year = year_range(player_name)
    menu_start = int(year)
    time_dict = [{'label': i, 'value' : i} for i in range(menu_start,max_year+1)]
    return time_dict

@app.callback(Output("end_year","value"),[Input("start_year","value"),Input("player_name_det","value")])
def update_end_year_menu(year, player_name):
    min_year, max_year = year_range(player_name)
    return max_year

@app.callback(Output("batting_metrics","options"),[Input("player_name_det","value")])
def update_metrics_menu_items(selection):
    if selection in list(batters_df['Name'].unique()):
        col_list = ['yearID','lgID']+list(batters_df.columns[4:len(batters_df.columns)-2])
        bat_col_dict = [{'label':cols, 'value':cols} for cols in col_list]
    else:
        bat_col_dict = []

    return(bat_col_dict)

# Setting up the callback for batting metric dropdown
@app.callback(Output('top_batting_metric','options'),[Input('batting_metrics','value')])
def update_top_batting_metric(selection):
    return [{'label':item, 'value':item} for item in top_metric(selection)]

@app.callback(Output("pitching_metrics","options"),[Input("player_name_det","value")])
def update_metrics_menu_items(selection):
    if selection in list(pitchers_df['Name'].unique()):
        col_list = ['yearID','lgID']+list(pitchers_df.columns[4:len(pitchers_df.columns)-2])
        pit_col_dict = [{'label':cols, 'value':cols} for cols in col_list]
    else:
        pit_col_dict = []

    return(pit_col_dict)

# Setting up the callback for pitching metric dropdown
@app.callback(Output('top_pitching_metric','options'),[Input('pitching_metrics','value'),Input("player_name_det","value")])
def update_top_pitching_metric(lst,selection):
    if selection in list(pitchers_df['Name'].unique()):
        return [{'label':item, 'value':item} for item in top_metric(lst)]
    else:
        return []


# Callback for the players images
@app.callback(Output('player_det_image','src'),[Input("player_name_det","value")])
def callback_image(player_name):
    path = '../img_new/'+player_name+'.png'
    return encode_image(path)

# Setting up the callback for player name below the image
@app.callback(Output('player_img_name','children'),[Input('player_name_det','value')])
def update_player_name(selection):
    return str(selection)


#############Batting table header and tooltip (Start)##################
# Setting up a call back for the batting table header
@app.callback(Output('bat_table','columns'),
              [Input('submit_player_id','n_clicks')],
              [State('batting_metrics','value')])
def update_table_header(n_clicks,value_list):
    return [{"name": i, "id": i} for i in value_list]

# Setting up a call back for the batting table header tooltip
@app.callback(Output('bat_table','tooltip_header'),
              [Input('submit_player_id','n_clicks')],
              [State('batting_metrics','value')])
def table_header_tooltip(n_clicks,value_list):
    return {val:batting_tooltip[val] for val in value_list}
#############Batting table header and tooltip (End)##################

#############Batting Average table header and tooltip (Start)##################
# Setting up a callback for the batting average table headers
@app.callback(Output('bat_avg_table','columns'),
             [Input('submit_player_id','n_clicks')],
             [State('batting_metrics','value')])
def update_avg_batting_table_header(n_clicks, value_list):
    ## Excluding the three categorical columns from the average table
    lst = remove_categorical_columns(value_list)
    return [{"name": i, "id": i} for i in lst]

@app.callback(Output('bat_avg_table','tooltip_header'),
             [Input('submit_player_id','n_clicks')],
             [State('batting_metrics','value')])
def update_avg_batting_table_header(n_clicks, value_list):
    ## Excluding the three categorical columns from the average table
    lst = remove_categorical_columns(value_list)
    return {val:batting_tooltip[val] for val in lst}

#############Batting Average table header and tooltip (End)##################

# setting up a callback for the pitching table header
@app.callback(Output('pit_table','columns'),
              [Input('submit_player_id','n_clicks')],
              [State('pitching_metrics','value')])
def update_table_header(n_clicks,value_list):
    return [{"name": i, "id": i} for i in value_list]

# Setting up a callback for the batting average table headers
@app.callback(Output('pit_avg_table','columns'),
             [Input('submit_player_id','n_clicks')],
             [State('pitching_metrics','value')])
def update_avg_pitching_table_header(n_clicks, value_list):
    ## Excluding the three categorical columns from the average table
    lst = remove_categorical_columns(value_list)
    return [{"name": i, "id": i} for i in lst]

# updating table data (For both batters)
@app.callback(Output('bat_table','data'),
             [Input('submit_player_id','n_clicks')],
             [State('start_year','value'),
             State('end_year','value'),
             State('batting_metrics','value'),
             State('player_name_det','value'),
             State('batting_top_n','value'),
             State('top_batting_metric','value')])
def update_batter_table(n_clicks,start_year,end_year,bat_met,player_name,k,metric):

    df = batters_df.loc[(batters_df['yearID'] >= int(start_year)) & (batters_df['yearID'] <= int(end_year)) & (batters_df['Name'] == player_name),bat_met+['Name']]
    df = df.fillna(0)
    format_columns = ['ISO','BABIP','AVG', 'OBP', 'SLG','wOBA']
    df = format_cols(df,bat_met, format_columns)
    df = top_n_metric(df,k,metric)
    df = df.loc[:,df.columns != 'Name']
    return(df.to_dict('records'))


# Updating and populating the average table
@app.callback(Output('bat_avg_table','data'),
             [Input('submit_player_id','n_clicks')],
             [State('start_year','value'),
             State('end_year','value'),
             State('batting_metrics','value'),
             State('player_name_det','value')])
def update_batting_average(n_clicks,start_year,end_year,bat_met,player_name):
    lst = remove_categorical_columns(bat_met)
    df = batters_df.loc[(batters_df['yearID'] >= int(start_year)) & (batters_df['yearID'] <= int(end_year)) & (batters_df['Name'] == player_name),lst]
    data_dict = df.mean(axis = 0).to_dict()
    if bat_met in ['BABIP','ISO','AVG']:
        BABIP = data_dict['BABIP']
        ISO = data_dict['ISO']
        AVG = data_dict['AVG']
    else:
        pass

    res = {key : round(data_dict[key], 2) for key in data_dict}
    if bat_met in ['BABIP','ISO','AVG']:
        res["BABIP"] = round(BABIP, 4)
        res["ISO"] = round(ISO, 4)
        res["AVG"] = round(AVG, 4)
    return [res]


# updating table data (For Pitchers)
@app.callback(Output('pit_table','data'),
             [Input('submit_player_id','n_clicks')],
             [State('start_year','value'),
             State('end_year','value'),
             State('pitching_metrics','value'),
             State('player_name_det','value'),
             State('pitching_top_n','value'),
             State('top_pitching_metric','value')])
def update_pitcher_table(n_clicks,start_year,end_year,pit_met,player_name,k,metric):
    df = pitchers_df.loc[(pitchers_df['yearID'] >= int(start_year)) & (pitchers_df['yearID'] <= int(end_year)) & (pitchers_df['Name'] == player_name),pit_met+['Name']]
    df = df.fillna(0)
    format_columns = ['BABIP']
    df = format_cols(df,pit_met, format_columns)
    df = top_n_metric(df,k,metric)
    df = df.loc[:,df.columns != 'Name']
    return(df.to_dict('records'))


# Updating and populating the pitching average table
@app.callback(Output('pit_avg_table','data'),
             [Input('submit_player_id','n_clicks')],
             [State('start_year','value'),
             State('end_year','value'),
             State('pitching_metrics','value'),
             State('player_name_det','value')])
def update_pitching_average(n_clicks,start_year,end_year,pit_met,player_name):
    lst = remove_categorical_columns(pit_met)
    df = pitchers_df.loc[(pitchers_df['yearID'] >= int(start_year)) & (pitchers_df['yearID'] <= int(end_year)) & (pitchers_df['Name'] == player_name),lst]
    data_dict = df.mean(axis = 0).to_dict()
    if pit_met in ['BABIP','ISO','AVG']:
        BABIP = data_dict['BABIP']
        ISO = data_dict['ISO']
        AVG = data_dict['AVG']
    else:
        pass
    res = {key : round(data_dict[key], 2) for key in data_dict}
    if pit_met in ['BABIP','ISO','AVG']:
        res["BABIP"] = round(BABIP, 3)
        res["ISO"] = round(ISO, 3)
        res["AVG"] = round(AVG, 3)
    else:
        pass
    return [res]


# Depnding on whether the player selected is a batter or pitcher, the correspoing table will appear based on the two callbacks below
@app.callback(
   Output('batting_det','style'),
   [Input('player_name_det', 'value')])
def show_hide_element(selection):
    if selection in list(batters_df['Name'].unique()):
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
   Output('pitching_det','style'),
   [Input('player_name_det', 'value')])
def show_hide_element(selection):
    if selection in list(pitchers_df['Name'].unique()):
        return {'display': 'block'}
    else:
        return {'display': 'none'}

#################### Player Details call backs end ###############################################
