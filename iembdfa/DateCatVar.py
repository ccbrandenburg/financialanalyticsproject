#H.2) Human assisted Data preprocessing and transformation for modelling -
#Text processing and Dates processing into variables that can be used in modelling.
#Create a list of possible actions that could be taken and create an user interface for a human to decide what to do.
#Once the action is chosen, perform the action.
import pandas as pd
from sklearn import preprocessing
import numpy as np
from datetime import datetime

def get_column_names(y):
    #create a list of the column names
    col_li = list(y)
    return(col_li)

#this function identfies categorical values (str) and converts them into integers to be used for modelling
def convert_categorical(y):
    #this function will encode categorical values
    def encode_categorical(x):
        enc = preprocessing.LabelEncoder()
        y[x] = enc.fit_transform(y[x])
    # function to each apply
    #for loop to check next values
    for item in get_column_names(y):
        if type(y[item][1]) == str:
            encode_categorical(item)
        if type(y[item][1]) == pd.tslib.Timestamp:
            pass
        if type(y[item][1]) != str:
            pass

#converting date string into a timestamps
#date to year
def convert_date_year(x):
    df[x] = pd.to_datetime(df[x])
    df[x+'_year'] = df[x].dt.year


#date to month
def convert_date_month(y, x):
    y[x] = pd.to_datetime(y[x])
    y[x+'_month'] = y[x].dt.month

#date to day
def convert_date_day(y, x):
    y[x] = pd.to_datetime(y[x])
    y[x+'_day'] = y[x].dt.day

#date to weekday
def convert_date_weekday(y, x):
    y[x] = pd.to_datetime(y[x])
    y[x+'_weekday'] = y[x].dt.weekday

#date to weeknum
def convert_date_weeknum(y, x):
    y[x] = pd.to_datetime(y[x])
    y[x+'_week'] = y[x].dt.week


def convert_dateflag(x):
    y[x+'_daydif'] = y['date'].apply(compare_dates)
    if y[x+'_daydif'] <= 30:
        y[x+'_daydif'] = 1
    if y[x+'_daydif'].apply(compare_dates) > 30:
        y[x+'_daydif'] = 0
    else:
        y[x+'_daydif'] = 0

def value_transformation(x):
    df = x
    date_var = []
    print("This program is going to help you with \n 1.Converting categorical values to numerical values \n 2.Converting dates into categorical values \n 2.1 Turn dates into year, month, week, day")
    #path = input("Please point to the csv file > ")
    #think about fixing this and list content of cwd
    #path = "/Users/cbrandenburg/Documents/IE/Courses/Term3/Financial_Analytics/Final Project/MBD_FA2.csv"
    #path = "/Users/cbrandenburg/Documents/IE/Courses/Term3/Financial_Analytics/Final Project/output.csv"
    #df = pd.read_csv(path)
    print("STATUS: csv loaded as dataframe.")
    print("These are the column names of your dataframe", get_column_names(df))
    #identify date columns
    while True:
        answer_date = str(input("Do you want to add one or multiple date columns? y/n > "))
        if answer_date == "n":
            break
        if answer_date == "y":
            date = input("Please indicate one date column (case sensitive) > ")
            if date in get_column_names(df):
                print(date, "is a valid column")
                date_var.append(date)
            if date not in get_column_names(df):
                print(date, " does not exist in the dataframe, please try again")
        while answer_date not in ["y", "n"]:
            print("Invalid input")
            break
    print("The following date transformations are possible:\n 1. Extract the year of the date as new variable \n 2. Extract the month of the date as new variable \n 3. Extract the day of the date as new variable \n 4. Extract the weekday of the date as new variable \n 5. Extract the weeknumber of the date as new variable \n 6. End transformations and export new .csv")
    #date transformations
    while True:
        transform_answer = input("Please pick one of the options (1,2,3,4,5,6) each can only be executed once > ")
        if transform_answer == "1":
            df[x+'_year'] = df.apply(lambda row: convert_date_year(row["date"]), axis=1)
        if transform_answer == "2":
            for i in date_var:
                convert_date_month(df, i)
        if transform_answer == "3":
            for i in date_var:
                convert_date_day(df, i)
        if transform_answer == "4":
            for i in date_var:
                convert_date_weekday(df, i)
        if transform_answer == "5":
            for i in date_var:
                convert_date_weeknum(df, i)
        if transform_answer == "6":
            print("STATUS: Ending date transformations")
            print("STATUS: Starting categorical values transformations")
            break
        while transform_answer not in ["1","2","3","4","5","6"]:
            print("Invalid input")
            break
    #categorical transformations
    while True:
        cat_answer = input("Do you want to turn categorical values (strings) into numerical values? Identification happens automatically. y/n > ")
        if cat_answer == "y":
            convert_categorical(df)
            print("STATUS: Transformation complete")
            break
        if cat_answer == "n":
            break
        while cat_answer not in ["y","n"]:
            print("Invalid input")
            break
    df.to_csv("output_test.csv")
    print("STATUS: The dataframe has beeen saved as output_test.csv")
    print(df.head(5))

