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
from app import tooltip
from dash_table.Format import Format, Scheme
from scipy.stats import pareto, beta
import collections
import pickle
import math
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

# Dataset load for detrending
WAR_batter_best = pd.read_csv(DATA_PATH.joinpath('WAR_batter_best.csv'))

## Reading the pickled Batting file for Batting fWAR
with open(DATA_PATH.joinpath('Batter_Name.pkl'),'rb') as batter_pickle:
    batter_dict = pickle.load(batter_pickle)


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



# New Function to format columns
def format_cols(met,format_columns_sp,format_columns_dp, format_columns_tp):
    """
    This helper function is primarily used for defining the formatting of the numbers displayed in the data tables
    Parameters:
    -- met: The list of all the metrics that the user has selected to be displayed in the table
    -- format_columns_sp: List of all the columns that will be displayed with single digit of precision
    -- format_columns_dp: List of all the columns that will be displayed with double digits of precision
    -- format_columns_tp: List of all the columns that will be displayed with triple digits of precision
    """
    emp_list = []
    #format_columns = ['ISO','BABIP','AVG', 'OBP', 'SLG','wOBA']
    for item in met:
        if item in format_columns_sp or item in format_columns_dp or item in format_columns_tp:
            emp_list.append(item)

    if not emp_list:
        return [{"name": i, "id": i} for i in met]
    else:
        ret_list = []
        for item in met:
            if item in format_columns_sp:
                ret_list.append({"name": item, "id": item, "type":"numeric", "format":Format(precision=1, scheme=Scheme.fixed)})
            elif item in format_columns_dp:
                ret_list.append({"name": item, "id": item, "type":"numeric", "format":Format(precision=2, scheme=Scheme.fixed)})
            elif item in format_columns_tp:
                ret_list.append({"name": item, "id": item, "type":"numeric", "format":Format(precision=3, scheme=Scheme.fixed)})
            else:
                ret_list.append({"name": item, "id": item})

        return ret_list

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

################################################################# Helper Function for batting averages and totals #################################################################
def remove_columns(lst):
    """
    This helper functions removes certain columns from the data frame if they exist in it
    """
    cols_rem = ['yearID','Team','lgID','Name','X','playerID','pops']

    for item in cols_rem:
        if item in lst:
            lst.remove(item)

    return(lst)

def check_base_fields(df,base_fields):
    """
    This function check whether the entries in the base fields list is present in the data frame or not.
    If the item is not present in the data frame it appends it to an empty list and returns it
    """
    emp_list = []
    for item in base_fields:
        if item not in list(df.columns):
            emp_list.append(item)

    return emp_list


def original_dataframe(start_year,end_year,bat_met,player_name):
    """
    This function returns a dataframe subsetted according to the bat_met columns and other parameters
    """
    return batters_df.loc[(batters_df['yearID'] >= int(start_year)) & (batters_df['yearID'] <= int(end_year)) & (batters_df['Name'] == player_name),bat_met]


def slg_average(df,start_year,end_year,bat_met,player_name):
    """
    Helper function to calculate the slogging average value
    """
    base_fields = ['AB','HR','X3B','X2B','SLG']
    emp_list = check_base_fields(df,base_fields)

    if not emp_list:
        df['X1B'] = round(df['SLG']*df['AB'] - (4*df['HR'] + 3*df['X3B'] + 2*df['X2B']),0)
        return round((df['X1B'].sum(axis = 0) + df['X2B'].sum(axis = 0) * 2 + df['X3B'].sum(axis = 0) * 3 + df['HR'].sum(axis = 0) * 4) / df['AB'].sum(axis = 0),3)

    else:
        df = original_dataframe(start_year,end_year,bat_met+emp_list,player_name)
        df['X1B'] = round(df['SLG']*df['AB'] - (4*df['HR'] + 3*df['X3B'] + 2*df['X2B']),0)
        SLG = round((df['X1B'].sum(axis = 0) + df['X2B'].sum(axis = 0) * 2 + df['X3B'].sum(axis = 0) * 3 + df['HR'].sum(axis = 0) * 4) / df['AB'].sum(axis = 0),3)
        del df['X1B']
        return SLG

def batting_average(df,start_year,end_year,bat_met,player_name):

    """
    Helper function to calculate the batting average
    """

    base_fields = ['H','AB']
    emp_list = check_base_fields(df,base_fields)

    if not emp_list:
        return round(df['H'].sum(axis = 0) / df['AB'].sum(axis = 0),3)

    else:
        df = original_dataframe(start_year,end_year,bat_met+emp_list,player_name)
        return round(df['H'].sum(axis = 0) / df['AB'].sum(axis = 0),3)


def strikeout_percentage_average(df,start_year, end_year,bat_met, player_name):
    """
    Helper function to calculate the strikeout percentage
    """

    base_fields = ['PA']
    emp_list = check_base_fields(df,base_fields)

    if not emp_list:
        k_val = round((pd.to_numeric(df['K.'].str.split('%').str[0])/100)*df['PA'],0).sum()
        pa_total = df['PA'].fillna(0).sum()
        return "{:.2%}".format(k_val / pa_total)
    else:
        df = original_dataframe(start_year,end_year,bat_met+emp_list,player_name)
        return strikeout_percentage_average(df,start_year, end_year,bat_met, player_name)


def walkout_percentage_average(df,start_year, end_year,bat_met, player_name):
    """
    Helper function to calculate the walkout percentage
    """
    base_fields = ['PA']
    emp_list = check_base_fields(df,base_fields)

    if not emp_list:
        bb_val = round((pd.to_numeric(df['BB.'].str.split('%').str[0])/100)*df['PA'],0).sum()
        pa_total = df['PA'].fillna(0).sum()
        return "{:.2%}".format(bb_val / pa_total)
    else:
        df = original_dataframe(start_year,end_year,bat_met+emp_list,player_name)
        return walkout_percentage_average(df,start_year, end_year,bat_met, player_name)


## The function below ties all the helper functions together to do that aggregation
def update_batting_average(start_year,end_year,bat_met,player_name):
    # Removing the columns we don't want in our aggregation
    lst = remove_columns(bat_met)

    # Removing the columns that are not required in our data frame for our aggregation
    df = original_dataframe(start_year,end_year,lst,player_name)

    # Declaring two empty dictionaries. One for Average and other for total
    emp_dict_avg = {}
    emp_dict_sum = {}

    # Making a list of metrics where a simple summation of columns works
    simp_agg = ['PA','AB','HR','H','X2B','X3B','RBI','SB','CS','Off','Def']
    for item in simp_agg:
        if item in lst:
            emp_dict_avg[item] = round(df[item].mean(axis = 0),2)
            emp_dict_sum[item] = round(df[item].sum(axis = 0),2)

    # Code chunk for calculating Slogging Percentage
    if 'SLG' in lst:
        emp_dict_avg['SLG'] = slg_average(df,start_year,end_year,lst,player_name)


    # Checking for batting averages
    if 'AVG' in lst:
        emp_dict_avg['AVG'] = batting_average(df,start_year,end_year,lst,player_name)

    # Checking for Isolated power
    if 'ISO' in lst:
        agg_slg = slg_average(df,start_year,end_year,lst,player_name)
        agg_avg = batting_average(df,start_year,end_year,lst,player_name)
        emp_dict_avg['ISO'] = round(agg_slg - agg_avg,3)

    # Checking for strikeout percentage
    if 'K.' in lst:
        emp_dict_avg['K.'] = strikeout_percentage_average(df,start_year,end_year,lst,player_name)

    # Checking for walkout percentage
    if 'BB.' in lst:
        emp_dict_avg['BB.'] = walkout_percentage_average(df,start_year,end_year,lst,player_name)

    # last three metrics are sabermetrics. Since we don't have the underlying data, we are doing a simple aggregation (i.e. mean)
    if 'fWAR' in lst:
        emp_dict_avg['fWAR'] = round(df['fWAR'].mean(axis = 0),2)

    if 'wOBA' in lst:
        emp_dict_avg['wOBA'] = round(df['wOBA'].mean(axis = 0),2)

    if 'wRC.' in lst:
        emp_dict_avg['wRC.'] = round(df['wRC.'].mean(axis = 0),2)

    if 'BABIP' in lst:
        emp_dict_avg['BABIP'] = round(df['BABIP'].mean(axis = 0),2)

    if 'OBP' in lst:
        emp_dict_avg['OBP'] = round(df['OBP'].mean(axis = 0),2)

    # returning both the
    return [emp_dict_avg,emp_dict_sum]

################################################################################ Helper Function for batting averages and totals ################################################################################
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
            end_new = df_new[metric].rolling(k).mean().dropna().idxmax()
            start_new = end_new - (k-1)
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

############################################################################## Helper Functions for Detrending (Start) #############################################################

## Over a thousand iteration, this code chunk ~150 seconds in total and 0.15 per run
def player_threshold(metric,threshold):

    met_thresh_dict = {}
    for k,d in batter_dict.items():
        ini_dict = list(map(lambda x: {metric:x[metric]},d))
        counter = collections.Counter()
        for j in ini_dict:
            counter.update(j)
        if metric in dict(counter) and dict(counter)[metric] >= threshold:
            met_thresh_dict[k] = d

    return met_thresh_dict



def find_interval(num, lst):

    """
    Function to return index of the largest element smaller than the given number
    """
    if num <= lst[0]:
        return -1

    if num >= lst[len(lst)-1]:
        return len(lst) - 1


    for i in range(0,len(lst)-1):
        if lst[i] < num and lst[i+1] > num:
            return i


def map_Y(u, ytilde):
    n = len(ytilde) - 1
    sequence = np.arange(0, 1, 1/n).tolist() + [1]
    pos = find_interval(u, sequence)
    out = (n*u - (pos + 1) + 1) * (ytilde[(pos+1)] - ytilde[pos]) + ytilde[pos]
    return out


def map_pareto_vals_vec(war_talent, npop, alpha = 1.16):
    n = len(war_talent)
    if len(npop) == 1:
        npop = [npop] * n
    lst = []
    for i in range(n):
        lst.append(beta.cdf(pareto.cdf(war_talent[i], b=alpha, scale = 1),a = (i+1) + npop[i] - n, b = n+1 - (i+1)))

    return lst

def order_qempirical(vec, ytilde):
    n = len(vec)
    a = []
    for i in range(n):
        a.append(beta.ppf(vec[i], i+1, n-i))

    out = []
    for i in range(n):
        out.append(map_Y(a[i], ytilde))

    return out



## Central Function for detrending which calls all the above helper function that are listed

def career_talent(filtered_dict,WAR_batter_best, player_name, year_proj):

    """
    This function is the parent function that projects the talent of a particular player in a particular year

    Parameters
    ########################################################################################################

    filtered_dict: dictionary of players which cross the threshold (for example: players having AB > 4000)
    WAR_Batter_best: Dataframe that contains the scaled data set
    player_name: Player namefor which you are running the projection
    year_proj: Year for which you are running the projection
    """

    # Subsetting the dictionary for a particular player
    filtered_dict_new = filtered_dict[player_name]

    # Converting it to a dataframe
    df = pd.DataFrame(filtered_dict_new).sort_values(by=['yearID']).reset_index(drop = True)

    # Changing the player ID
    df['playerID'] = df['playerID'].astype(str) + '_proj'

    # Putting a counter for the number of iteration (we can use this counter later to determine the runtime of each loop)
    # j = 0

    # Creating an empty dataframe where we will be merging the rows
    emp_df = pd.DataFrame()

    # Iterating over all the years where the selected player has played baseball
    for idx, year in df['yearID'].iteritems():
        #j+=1

        # subsetting the Normalized WAR Talent Dataset for the year in which we want to find the projection
        batters_int = WAR_batter_best.loc[WAR_batter_best['yearID'] == year_proj]

        # creating a vector with all the WAR values in the projection year
        yy = batters_int.loc[:,'scale_WAR'].sort_values().reset_index(drop = True)

        # Finding the length of the vector created above
        n = len(yy)

        # Creating a zero vector with one more cell than the vector with all the scaled WAR values
        ytilde = [0] * (n+1)

        # Changing the first value in ytilde (Extrapolation logic developed by Shen and Daniel)
        ytilde[0] = yy[0] - 1/(yy[n-1] - yy[0])

        # Changing the last value in ytilde
        ytilde[n] = yy[n-1] + 1/(yy[n-1] - yy[n-4])

        # Iterating over all the cells of ytilde and updating their values
        # The updated ytilde is the new empirical distribution function of the WAR Talent
        for i in range(1,n):
            ytilde[i] = (yy[i]+yy[i-1])/2

        batters_int = pd.concat([batters_int,df.loc[df['yearID'] == year]]).reset_index(drop = True)
        batters_int['pops'][batters_int.shape[0]-1] = batters_int['pops'][0]
        batters_int = batters_int.sort_values(by = ['WAR_talent']).reset_index(drop = True)
        batters_int['foo'] = map_pareto_vals_vec(batters_int['WAR_talent'], batters_int['pops'])
        batters_int['foo'] = batters_int['foo'].reset_index(drop = True)
        batters_int['adj_WAR'] = order_qempirical(batters_int['foo'], ytilde)
        del batters_int['foo']
        batters_int = batters_int.loc[batters_int['playerID'] == df['playerID'].unique()[0]]
        emp_df = pd.concat([emp_df, batters_int])
        emp_df['target_year'] = year_proj

    return emp_df

############################################################################## Helper Functions for Detrending (End) #############################################################
############################################################################## Layout (Start) ######################################################################################

layout = html.Div(id = "player_details_content",children = [

                                 dbc.Row([
                                 html.Br(),
                                 # Content for column 1 will go here
                                 dbc.Col([
                                         html.P("Filters",style = {'font-size':'20px'}),
                                        html.Label(["Start Year",
                                                    dcc.Dropdown(
                                                        id = 'start_year',
                                                        options = time_dict,
                                                        value = 1871,
                                                        persistence = True,
                                                        persistence_type = 'session',
                                                        clearable = False
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["End Year",
                                                    dcc.Dropdown(
                                                    id = 'end_year',
                                                    options = time_dict,
                                                    value = 2019,
                                                    persistence = True,
                                                    persistence_type = 'session',
                                                    clearable = False
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["Batting Metrics",
                                                    dcc.Dropdown(
                                                    id = 'batting_metrics',
                                                    multi = True,
                                                    value = ['yearID','Team','PA','AB','AVG','HR','H','X2B','X3B','RBI','SB','ISO','BABIP','fWAR'],
                                                    persistence = True,
                                                    persistence_type = 'session'
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["Select Top N years",
                                                    dbc.Input(
                                                    id = 'batting_top_n',
                                                    type = "number",
                                                    placeholder = "type in a number",
                                                    persistence = True,
                                                    persistence_type = 'session'
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["By (Batting Metric):",
                                                    dcc.Dropdown(
                                                    id = 'top_batting_metric',
                                                    clearable = True,
                                                    persistence = True,
                                                    persistence_type = 'session'
                                                    )
                                                   ], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["Pitching Metrics",
                                                    dcc.Dropdown(
                                                    id = 'pitching_metrics',
                                                    multi = True,
                                                    value = ['yearID','Team','IP','K.9','BB.9','HR.9','BABIP','ERA','FIP','WHIP','fWAR'],
                                                    persistence = True,
                                                    persistence_type = 'session'
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["Select Top N years",
                                                    dbc.Input(
                                                    id = 'pitching_top_n',
                                                    type = "number",
                                                    placeholder = "type in a number",
                                                    persistence = True,
                                                    persistence_type = 'session'
                                                    )], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Label(["By (Pitching Metric):",
                                                    dcc.Dropdown(
                                                    id = 'top_pitching_metric',
                                                    clearable = True,
                                                    persistence = True,
                                                    persistence_type = 'session'
                                                    )
                                                   ], style = {'width':'80%'}),

                                        html.Br(),
                                        html.Br(),
                                        dbc.FormGroup(
                                            [
                                                dbc.Label("Detrend Section"),
                                                dbc.Checklist(
                                                    options = [
                                                        {"label":"Detrend Batting", "value":1},
                                                        {"label":"Detrend Pitching", "value":2},
                                                    ],
                                                    value = [],
                                                    id = "detrending_toggles",
                                                    inline = False,
                                                    switch = True,
                                                    persistence = True,
                                                    persistence_type = 'session'
                                                )
                                            ]
                                        ),
                                        html.Br(),
                                        html.Label(["Projection Year",
                                                    dcc.Dropdown(
                                                    id = 'projection_year',
                                                    options = time_dict,
                                                    value = 2019,
                                                    persistence = True,
                                                    persistence_type = 'session',
                                                    clearable = False
                                                    )], style = {'width':'80%'}),
                                        html.Br(),
                                        html.Br(),
                                        dbc.Button("Submit Metrics to Update Table",color = "success",n_clicks = 0, id = 'submit_player_id'),

                                        html.Br(),
                                        html.Br(),
                                        dcc.Markdown("""
                                        ##### __Note:__
                                        * The data tables (for both batting/pitching) can be filtered/sorted based on any column
                                        * Sorting can be done by clicking on any column header
                                        * Filteration can be done by typing in the search bar present below the column header, the following text:
                                            * Greater than some Value ( > Value)
                                            * <=Value
                                            * =Value
                                            * To restore the table to its original form clear the text and hit 'enter'
                                        """)

                                 ], width = 4),


                                 # Content for column 2 will go here
                                 dbc.Col([
                                 html.Br(),
                                 html.Br(),
                                 html.Br(),
                                 html.Br(),
                                 dbc.Row([html.P("Nap Lajoie", id="player_img_name", style = {'font-weight':'bold'})]),
                                 html.Br(),
                                 dbc.Row([
                                    dbc.Col([html.Img(id = "player_det_image",src="children",height=250)], width = 3),
                                    dbc.Col([dash_table.DataTable(
                                                id = "player_bio_table_new",
                                                columns = [{"name": "Attributes", "id": "bio_header"},
                                                           {"name": "Details", "id": "bio_details"}],
                                                style_cell={'textAlign': 'left'},
                                                style_data = {'width':'50px'}
                                    )], width = 5)
                                 ],justify="left"),
                                 html.Br(),
                                 ##dbc.Row([html.P(id="player_det_txt", style = {'font-weight':'bold'})],justify="left"),
                                 dbc.Row([
                                 dbc.Spinner(children = [
                                 html.Div(id = "detrend_batting",
                                         children = [html.P("Detrended Batting stats", id = "detrend_bat_title", style = {'font-weight':'bold'}),
                                                     html.Br(),
                                                     dash_table.DataTable(
                                                     id = "bat_detrend_table",
                                                     columns = [{'name': 'yearID', 'id': 'yearID'}, {'name': 'Proj. fWAR',
                                                                                                     'id': 'Proj. fWAR',
                                                                                                     "type":"numeric",
                                                                                                     "format":Format(precision=2, scheme=Scheme.fixed)}],
                                                     style_cell = {

                                                         'textAlign':'left',
                                                         'whiteSpace':'normal',
                                                         'height':'auto',
                                                         'maxWidth':'100px', 'minWidth':'100px'
                                                     },
                                                     style_data = {'width':'120px'},
                                                     sort_action = "native",
                                                     sort_mode = "multi",
                                                     style_table = {'overflowX':'auto'},
                                                     filter_action = "native"
                                                     )], style= {'display': 'block'})
                                 ], fullscreen = True)

                                 ], justify = "left"),
                                 html.Br(),
                                 dbc.Row([
                                     html.Div(id="batting_det",
                                              children = [html.P("Batting",id="batting_txt", style = {'font-weight':'bold'}),

                                                          dash_table.DataTable(
                                                          id = 'bat_table',
                                                          style_cell={'textAlign': 'left',
                                                                      'whiteSpace':'normal',
                                                                      'height':'auto',
                                                                      'maxWidth': '50px','minWidth': '50px'},
                                                          style_data = {'width':'120px'},
                                                          #style_data_conditional=style_row_by_top_values(df),
                                                          sort_action="native",
                                                          sort_mode="multi",
                                                          style_table={'overflowX': 'auto'},
                                                          filter_action="native"
                                                                            ),
                                                          html.Br(),

                                                          html.P("Batting Totals",id="batting_tot_txt", style = {'font-weight':'bold'}),

                                                          html.Br(),

                                                          dash_table.DataTable(
                                                          id = 'bat_tot_table',
                                                          style_cell = {
                                                          'textAlign':'left',
                                                          'whiteSpace':'normal',
                                                          'height':'auto'},
                                                          style_data = {'width':'120px'}),

                                                          html.Br(),

                                                          html.P("Batting Averages",id="batting_average_txt", style = {'font-weight':'bold'}),

                                                          dash_table.DataTable(
                                                          id = 'bat_avg_table',
                                                          style_cell = {
                                                          'textAlign':'left',
                                                          'whiteSpace':'normal',
                                                          'height':'auto',
                                                          'maxWidth': '50px','minWidth': '50px'},
                                                           style_data = {'width':'120px'}
                                                            )], style= {'display': 'block'}),
                                                          html.Br()
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

# Callback for the player bio table
@app.callback(Output('player_bio_table_new','data'),[Input('player_name_det','value')])
def update_player_bio(player_name):
    df = player_bio.loc[(player_bio['player_name'] == player_name) & -pd.isnull(player_bio['bio_details']),['bio_header','bio_details']]
    return(df.to_dict('records'))

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
    format_columns_tp = ['ISO','BABIP','AVG', 'OBP', 'SLG','wOBA']
    format_columns_dp = ['Def','Off','fWAR']
    format_columns_sp = []
    ret_lst = format_cols(value_list,format_columns_sp,format_columns_dp,format_columns_tp)
    return ret_lst

# Setting up a call back for the batting table header tooltip
@app.callback(Output('bat_table','tooltip_header'),
              [Input('submit_player_id','n_clicks')],
              [State('batting_metrics','value')])
def table_header_tooltip(n_clicks,value_list):
    return {val:tooltip['batting_tooltip'][val] for val in value_list}
#############Batting table header and tooltip (End)##################

#############Batting Average table header and tooltip (Start)##################
# Setting up a callback for the batting average table headers
@app.callback(Output('bat_avg_table','columns'),
             [Input('submit_player_id','n_clicks')],
             [State('batting_metrics','value')])
def update_avg_batting_table_header(n_clicks, value_list):
    ## Excluding the three categorical columns from the average table
    lst = remove_categorical_columns(value_list)
    format_columns_tp = ['ISO','BABIP','AVG', 'OBP', 'SLG','wOBA']
    format_columns_dp = ['Def','Off','fWAR']
    format_columns_sp = []
    ret_lst = format_cols(lst,format_columns_sp,format_columns_dp, format_columns_tp)
    return ret_lst

@app.callback(Output('bat_avg_table','tooltip_header'),
             [Input('submit_player_id','n_clicks')],
             [State('batting_metrics','value')])
def update_avg_batting_table_header_tooltip(n_clicks, value_list):
    ## Excluding the three categorical columns from the average table
    lst = remove_categorical_columns(value_list)
    return {val:tooltip['batting_tooltip'][val] for val in lst}

#############Batting Average table header and tooltip (End)##################

#############Batting Average table header and tooltip (Start)##################
@app.callback(Output('bat_tot_table','columns'),
              [Input('submit_player_id','n_clicks')],
              [State('batting_metrics','value')])
def update_total_batting_table_header(n_clicks, value_list):
    simp_agg = ['PA','AB','HR','H','X2B','X3B','RBI','SB','CS','Off','Def']
    emp_list = []
    for item in simp_agg:
        if item in value_list:
            emp_list.append(item)

    format_columns_tp = ['ISO','BABIP','AVG', 'OBP', 'SLG','wOBA']
    format_columns_dp = ['Def','Off','fWAR']
    format_columns_sp = []
    res_lst = format_cols(emp_list,format_columns_sp,format_columns_dp, format_columns_tp)
    return res_lst

#############Batting Average table header and tooltip (End)##################

############# Pitching table header and tooltip (Start)##################
# setting up a callback for the pitching table header
@app.callback(Output('pit_table','columns'),
              [Input('submit_player_id','n_clicks')],
              [State('pitching_metrics','value')])
def update_table_header(n_clicks,value_list):
    format_columns_tp = ['BABIP']
    format_columns_dp = ['K.9','BB.9','HR.9','ERA','FIP','WHIP','fWAR']
    format_columns_sp = ['IP']
    ret_lst = format_cols(value_list,format_columns_sp,format_columns_dp, format_columns_tp)
    return ret_lst

# Setting up a call back for the batting table header tooltip
@app.callback(Output('pit_table','tooltip_header'),
              [Input('submit_player_id','n_clicks')],
              [State('pitching_metrics','value')])
def table_header_tooltip(n_clicks,value_list):
    return {val:tooltip['pitching_tooltip'][val] for val in value_list}

############# Pitching table header and tooltip (End)##################

############# Pitching Average table header and tooltip (Start)##################
# Setting up a callback for the batting average table headers
@app.callback(Output('pit_avg_table','columns'),
             [Input('submit_player_id','n_clicks')],
             [State('pitching_metrics','value')])
def update_avg_pitching_table_header(n_clicks, value_list):
    ## Excluding the three categorical columns from the average table
    lst = remove_categorical_columns(value_list)
    format_columns_tp = ['BABIP']
    format_columns_dp = ['K.9','BB.9','HR.9','ERA','FIP','WHIP','fWAR']
    format_columns_sp = ['IP']
    ret_lst = format_cols(lst,format_columns_sp,format_columns_dp, format_columns_tp)
    return ret_lst


@app.callback(Output('pit_avg_table','tooltip_header'),
             [Input('submit_player_id','n_clicks')],
             [State('pitching_metrics','value')])
def update_avg_batting_table_header(n_clicks, value_list):
    ## Excluding the three categorical columns from the average table
    lst = remove_categorical_columns(value_list)
    return {val:tooltip['pitching_tooltip'][val] for val in lst}


############# Pitching Average table header and tooltip (End)##################
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
    df = top_n_metric(df,k,metric)
    df = df.loc[:,df.columns != 'Name']
    return(df.to_dict('records'))

# updating and populating the total table
@app.callback(Output('bat_tot_table','data'),
             [Input('submit_player_id','n_clicks')],
             [State('start_year','value'),
             State('end_year','value'),
             State('batting_metrics','value'),
             State('player_name_det','value')])
def batting_average_val(n_clicks,start_year,end_year,bat_met,player_name):
    res = update_batting_average(start_year,end_year,bat_met,player_name)[1]
    return [res]

# Updating and populating the average table
@app.callback(Output('bat_avg_table','data'),
             [Input('submit_player_id','n_clicks')],
             [State('start_year','value'),
             State('end_year','value'),
             State('batting_metrics','value'),
             State('player_name_det','value')])
def batting_average_val(n_clicks,start_year,end_year,bat_met,player_name):
    res = update_batting_average(start_year,end_year,bat_met,player_name)[0]
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



# Detrending Batting Metrics Table (Body)
@app.callback(Output('bat_detrend_table','data'),
             [Input('detrending_toggles','value'),
             Input('player_name_det','value'),
             Input('start_year','value'),
             Input('end_year','value'),
             Input('projection_year','value')])
def update_detrended_batting_table(selection, player_name, start_year, end_year, proj_year):
    if 1 in selection:
        filtered_dict = player_threshold('AB',0)
        example = career_talent(filtered_dict, WAR_batter_best, player_name, int(proj_year))
        example.rename(columns = {'adj_WAR':'Proj. fWAR'}, inplace = True)
        example = example.loc[(example['yearID'] >= int(start_year)) & (example['yearID'] <= int(end_year)),['yearID','Proj. fWAR']]
        example['Proj. fWAR'] = example['Proj. fWAR'].round(decimals = 2)
        example['Proj. fWAR_str'] = example['Proj. fWAR'].apply(lambda x: str(x))
        return(example.to_dict('records'))
    else:
        return None


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

# The call back below is for adjusting the detrending of batting statistic
@app.callback(
    Output('detrend_batting','style'),
    [Input('detrending_toggles','value')])
def show_hide_batting_detrending(selection):
    if 1 in selection:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


#################### Player Details call backs end ###############################################
