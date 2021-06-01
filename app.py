import dash
import dash_bootstrap_components as dbc

# Defining the app default stylesheet
app = dash.Dash(external_stylesheets = [dbc.themes.MATERIA],
                meta_tags = [
                    {
                    "name":"viewport",
                    "content":'width=device-width,initial-scale=1.0'
                    }
                ],
                suppress_callback_exceptions = True)


tooltip = {

'batting_tooltip': {
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
},

'pitching_tooltip': {
                    'yearID':['Year of Play'],
                    'lgID':['Baseball League Id'],
                    'Name':['Player Name'],
                    'Team':['Team Name'],
                    'IP':['Innings Pitched'],
                    'K.9':['Number of Strikeouts a pithcer averages every nine innings Pitched'],
                    'BB.9':['Average number of bases on balls (or walks) given up by a pitcher per nine innigs pitched'],
                    'HR.9':['Average number of home runs allowed by a pitcher on a nine inning scale'],
                    'BABIP':["""Batting Average on Balls In Play (BABIP) measures how often a ball in play goes for a hit.
                                A ball is “in play” when the plate appearance ends in something other than a strikeout, walk, hit batter,
                                catcher’s interference, sacrifice bunt, or home run. In other words, the batter put the ball in play and it didn’t clear the outfield fence.
                                Typically around 30% of all balls in play fall for hits, but there are several variables that can affect BABIP rates for individual players,
                                such as defense, luck, and talent level"""],
                    'ERA':['Earned Runs Average: Average of earned runs given up by a pitcher per nine innings pitched'],
                    'FIP':["""Fielding Independent Pitching: This metric converts a pitcher's three outcomes into an earned run average like Number
                              The formula is (13*HR+3*(HBP+BB)-2*K)/IP, plus a constant (usually around 3.2) to put it on the same scale as earned run average"""],
                    'WHIP':["""Walks plus hits per inning pitched: WHIP is a sabermetric measurement
                               of the number of baserunners a pitcher has allowed per innings pitched. WHIP is calculated by adding the number of walks and hits allowed
                               and dividing this sum by the number of innings pitched"""],
                    'fWAR':['Wins above replacements']
}

}



server = app.server
