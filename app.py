import dash
import dash_bootstrap_components as dbc

# Defining the app default stylesheet
app = dash.Dash(external_stylesheets = [dbc.themes.MATERIA],
                meta_tags = [
                    {
                    "name":"viewport",
                    "content":'width=device-width,initial-scale=1.0'
                    }
                ])

server = app.server
