import pandas as pd
import numpy as np
import math
import time
import pickle
import collections
from scipy.stats import pareto, beta, norm, binom
import random
import warnings
import multiprocessing
import concurrent.futures
import functools
import pathlib
warnings.filterwarnings('ignore')


PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

## For Bating Averages
AVG_batter = pd.read_csv(DATA_PATH.joinpath("batters_talent_AVG.csv"))

## Reading batters average file to establish threshold
batters = pd.read_csv(DATA_PATH.joinpath("batters_average.csv"))

## creating a cutoff for each year
cut_off_df = pd.DataFrame()
for year in list(batters['yearID'].unique()):
    df = batters.loc[(batters['yearID'] == year) & (batters['AB'] >= 75),].copy()
    cut_off_df = pd.concat([cut_off_df, pd.DataFrame(list(zip([df['AB'].median()], [year])), columns = ['cutoff','yearID'])])


# Reading the pickled file (For Calculating WAR)
with open(DATA_PATH.joinpath('Batter_Name.pkl'),'rb') as batter_pickle:
    batter_dict = pickle.load(batter_pickle)



# Reading the pickled file (For Calculating Batting Averages)
with open(DATA_PATH.joinpath('Batter_Average.pkl'),'rb') as batter_average_pickle:
    batter_avg_dict = pickle.load(batter_average_pickle)



## writing all the helper funnctions below to detrend the Batting Averages

## Over a thousand iteration, this code chunk ~150 seconds in total and 0.15 per run
def player_threshold(metric,threshold, player_dict):
    
    met_thresh_dict = {}
    for k,d in player_dict.items():
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


def map_pareto_vals_vec(talent, npop, alpha = 1.16):
    n = len(talent)
    if len(npop) == 1:
        npop = [npop] * n
    lst = []
    for i in range(n):
        lst.append(beta.cdf(pareto.cdf(talent[i], b=alpha, scale = 1),a = (i+1) + npop[i] - n, b = n+1 - (i+1)))
    
    return lst

def order_qnorm(vec, mean, sd):
    n = len(vec)
    a = []
    for i in range(n):
        a.append(beta.ppf(vec[i], i+1, n-i))
    
    # converting the beta values to normal values
    out = norm.ppf(a, loc = mean, scale = sd)
    
    return out

def order_qempirical(vec, ytilde):
    n = len(vec)
    a = []
    for i in range(n):
        a.append(beta.ppf(vec[i], i+1, n-i))
    
    out = []
    for i in range(n):
        out.append(map_Y(a[i], ytilde))
    
    return out


def career_talent_average(filtered_dict, metric_df, player_id, year_proj):
    
    # Subsetting the dictionary for a particular player
    filtered_dict_new = filtered_dict[player_id]
    
    # converting the dictionary to a dataframe and then sorting it according to the year id
    df = pd.DataFrame(filtered_dict_new).sort_values(by=['yearID']).reset_index(drop = True)
    
    # changing the player id
    df['playerID'] = df['playerID'].astype(str) + '_proj'
    
    # creating an empty data frame
    emp_df = pd.DataFrame()
    
    # iterating over all the years where the selected player has played baseball
    for idx, year in df['yearID'].iteritems():
        
        # Finding out the threshold for the projection year
        AB_thresh = cut_off_df.loc[cut_off_df['yearID'] == year].cutoff.iloc[0]
        
        # subsetting the Normalized AVG Talent Dataset for the year in which we want to find the projection
        batters_int = metric_df.loc[(metric_df['yearID'] == year_proj) & (metric_df['AB'] >= AB_thresh)].copy()
        
        # Finding out the min, mean and standard deviation for adjustment
        min_int = batters_int['AVG'].min()
        mean_int = batters_int['AVG'].mean()
        sd_int = batters_int['AVG'].std()
        
        # combining the Normalised AVG Talent Dataset with the player data set extracted outside the loop
        batters_int = pd.concat([batters_int,df.loc[df['yearID'] == year]]).reset_index(drop = True)
        
        # matching the value of population for every row
        batters_int['pops'][batters_int.shape[0]-1] = batters_int['pops'][0]
        
        # sorting the batters_int data frame by AVG_talent and then resetting the index values
        batters_int = batters_int.sort_values(by = ['AVG_talent']).reset_index(drop = True)
        
        # Mapping the talent score (for batting average) to population percentiles
        batters_int['foo'] = map_pareto_vals_vec(batters_int['AVG_talent'], batters_int['pops'])
        batters_int['foo'] = batters_int['foo'].reset_index(drop = True)
        
        # Mapping the population percentiles back to normal distribution
        batters_int['adj_AVG'] = order_qnorm(batters_int['foo'], mean_int, sd_int)
        
        # deleting the intermediate column
        del batters_int['foo']
        
        # changing the population value, so that it is similar to other rows
        batters_int = batters_int.loc[batters_int['playerID'] == df['playerID'].unique()[0]]
        
        # capping the batting average at a minimum value
        batters_int['adj_AVG'] = batters_int['adj_AVG'].apply(lambda x: min_int if x < min_int else x)
        emp_df = pd.concat([emp_df, batters_int])
        emp_df['target_year'] = year_proj
    
    return emp_df

def year_based_calculation(year_proj, cut_off_df, df, year):
    
    # Finding out the threshold for the projection year
    AB_thresh = cut_off_df.loc[cut_off_df['yearID'] == year].cutoff.iloc[0]

    # subsetting the Normalized WAR Talent Dataset for the year in which we want to find the projection
    batters_int = AVG_batter.loc[(AVG_batter['yearID'] == year_proj) & (AVG_batter['AB'] >= AB_thresh)].copy()

    min_int = batters_int['AVG'].min()
    mean_int = batters_int['AVG'].mean()
    sd_int = batters_int['AVG'].std()

    # combining the Normalised AVG Talent Dataset with the player data set extracted outside the loop
    batters_int = pd.concat([batters_int,df.loc[df['yearID'] == year]]).reset_index(drop = True)

    # matching the value of population for every row
    batters_int['pops'][batters_int.shape[0]-1] = batters_int['pops'][0]

    # sorting the batters_int data frame by AVG_talent and then resetting the index values
    batters_int = batters_int.sort_values(by = ['AVG_talent']).reset_index(drop = True)


    batters_int['foo'] = map_pareto_vals_vec(batters_int['AVG_talent'], batters_int['pops'])
    batters_int['foo'] = map_pareto_vals_vec(batters_int['AVG_talent'], batters_int['pops'])
    batters_int['foo'] = batters_int['foo'].reset_index(drop = True)
    batters_int['adj_AVG'] = order_qnorm(batters_int['foo'], mean_int, sd_int)


    del batters_int['foo']
    batters_int = batters_int.loc[batters_int['playerID'] == df['playerID'].unique()[0]]

    batters_int['adj_AVG'] = batters_int['adj_AVG'].apply(lambda x: min_int if x < min_int else x)

    batters_int['target_year'] = year_proj
    
    return batters_int


## Main Guard

if __name__ == '__main__':


    # Filtering the dictionary for a particular player
    filtered_dict_new = batter_avg_dict['lajoina01']
    df = pd.DataFrame(filtered_dict_new).sort_values(by=['yearID']).reset_index(drop = True)

    # changing the player id
    df['playerID'] = df['playerID'].astype(str) + '_proj'

    year = list(df['yearID'])
    time_lst1 = []
    time_lst2 = []
    
    for year_dum in np.arange(2010,2020):
        year_proj = year_dum
        co_df = cut_off_df
        ndf = df

        ## Code chunk within the for loop for parallelization (start)
        
        start1 = time.time()
        year = list(df['yearID'].unique())
        partial_year_based_calculation = functools.partial(year_based_calculation, year_proj, co_df, ndf)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(partial_year_based_calculation, year)


        emp_df = pd.DataFrame()
        for result in results:
            emp_df = pd.concat([emp_df, result])
        
        #print(emp_df)
        end1 = time.time()

        ## Code chunk within the for loop for parallelization (end)
        
        # print(end1 - start1)
        time_lst1.append(end1 - start1)

        ## Code chunk within the for loop without the parallelization
        # start2 = time.time()
        # samp_df = career_talent_average(batter_avg_dict, AVG_batter, player_id = 'lajoina01', year_proj = year_dum)
        # end2 = time.time()

        # time_lst2.append(end2-start2)
    
    print('Parallel Time: ',  np.mean(time_lst1))
    #print('Non Parallel Time: ',  np.mean(time_lst2))







