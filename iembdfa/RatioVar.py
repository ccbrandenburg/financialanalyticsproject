import pandas as pd
import itertools


def load_training(dataset):
    training = pd.read_csv(dataset,sep=";", decimal=".", thousands =",")
    return training


def load_test(dataset):
    test = pd.read_csv(dataset)
    return test

def Variable_create(dataset):
    training = load_training(dataset)
    print(list(training))
    print(training)
    print(training.columns)

    column1 = input("Please select the columns you want separated by spaces:").split(" ")
    column_f = list(itertools.combinations(column1,2))

    k = list(column_f)
    print(k)

    operation = input("Please select the operations between the columns:")
    operators = ['+','-','*','/']
    if operation not in operators:
        print("operation not supported")
    print("Do you want to do this operation for all the combinations of columns?")
    confirmation = input("Y/N: ")

    if confirmation == 'Y':
        if operation == "+":
            i = 0
            for columns1,columns2 in k:
                training["Sum_column_"+str(len(training.columns)+i)] = training[columns1] + training[columns2]
                i = i+1

        elif operation == "-":
            i = 0
            for columns1,columns2 in k:
                training["Diff_column_"+str(len(training.columns)+i)] = training[training[columns1]] - training[training.columns[columns2]]
                i = i+1

        elif operation == "*":
            i = 0
            for columns1,columns2 in k:
                training["Product_column_"+str(len(training.columns)+i)] = training[training[columns1]] * training[training[columns2]]
                i = i+1

        elif operation == "/":
            i = 0
            print("I was here")
            for columns1,columns2 in k:
                training[columns1].fillna(training[columns1].median(),inplace = True)
                training[columns2].fillna(training[columns2].median(),inplace = True)
                training["Division_column_"+str(i)] = training[columns1] / training[columns2]
                i = i+1

    print(training.head())
    return(training)
