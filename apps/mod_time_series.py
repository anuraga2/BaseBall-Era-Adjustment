### Writing a class for the eda of the data frame

class Time_Series_Eda:
    
    ## Constructor of the class
    def __init__(self,df):
        self.df = df
    
    ## function to print the name of the data set
    def print_shape(self):
        print("The dimension of the data is: ", self.df.shape)
        
    
    ## Function to bucket all the columns of the dataset into their datatypes
    def col_classification(self):
        emp_dict = {}
        
        ## Initializing type dictionary
        for types in self.df.dtypes.unique():
            if types in emp_dict:
                continue
            else:
                emp_dict[types] = []
        
        # Appending column name to each data type 
        for col in self.df.columns:
            emp_dict[self.df[col].dtypes].append(col)
        
        return emp_dict
    
    ## Function to plot Univariate Time Series Data  
    def univariate_time_series_plot(self, name, xaxis, yaxis, column):

        """

        -- This function takes in the object and spits out Univariate plots for the specified sets of columns
        -- The parameter list is given below
            * self: The object
            * name: This variable contains the filteration criteria
            * xaxis: The vairable that will make the x axis of the univariate time series plot (mostly time)
            * yaxis: Variable that will make the y axis of the plot 

        """

        try:
            samp_df = self.df.loc[self.df[column] == name,:]
            if samp_df.shape[0] == 0:
                print("No rows left in the data frame after filteration. The function will return None")
                return None
            else:
                title_text = "Trend of " + yaxis + " for " + name
                fig = px.line(samp_df, x=xaxis, y=yaxis, title = title_text)
                return fig

        except ValueError:
            print("The value x/y column does not exist in the data frame")

        except NameError:
            print("The data frame that you are passing does not exist. Please review the input")

        # Except Clause 
        except KeyError:
            print("The column containing the filteration criteria does not exist in the data frame")
    
    
    ## Function to plot multiple trend lines in a single plot
        
    
    
    # print the type of column
    def column_type(self):
        dict_type = {}
        
        for col in self.df.columns:
            if self.df[col].dtypes not in dict_type.keys():
                dict_type[self.df[col].dtypes] = 1
            else:
                dict_type[self.df[col].dtypes] += 1
        
        col_class = self.col_classification()
        print('\n**********************************\n')
        print(col_class)
        print('\n**********************************\n')
        print(dict_type)