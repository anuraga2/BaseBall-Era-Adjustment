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

}



server = app.server
