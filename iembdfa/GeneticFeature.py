print("IMPORTING LIBRARIES...")
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re
from deap import creator, base, tools, algorithms #GENETIC ALGORITHM LIBRARY - requirement: pip install deap
import random
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import (LinearRegression, Ridge,
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from minepy import MINE
import json


#print("DOWNLOADING DATASETS...")
#df = pd.read_csv("https://dl.dropboxusercontent.com/u/28535341/dev.csv") #DEV-SAMPLE
#dfo = pd.read_csv("https://dl.dropboxusercontent.com/u/28535341/oot0.csv")#OUT-OF-TIME SAMPLE
#target_column = "ob_target"

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))

def feature_selection(df,target_column):
    print("IDENTIFYING TYPES...")
    in_model = []
    list_ib = set()  #input binary
    list_icn = set() #input categorical nominal
    list_ico = set() #input categorical ordinal
    list_if = set()  #input numerical continuos (input float)
    list_inputs = set()
    output_var = target_column



    for var_name in df.columns:
        if re.search('^ib_',var_name):
            list_inputs.add(var_name)
            list_ib.add(var_name)
            print (var_name,"is input binary")
        elif re.search('^icn_',var_name):
            list_inputs.add(var_name)
            list_icn.add(var_name)
            print (var_name,"is input categorical nominal")
        elif re.search('^ico_',var_name):
            list_inputs.add(var_name)
            list_ico.add(var_name)
            print (var_name,"is input categorical ordinal")
        elif re.search('^if_',var_name):
            #list_inputs.add(var_name)
            list_if.add(var_name)
            print (var_name,"is input numerical continuos (input float)")
        elif re.search('^ob_',var_name):
            output_var = var_name
        else:
            print ("ERROR: unable to identify the type of:", var_name)


    if (df[output_var].isin([0,1]).all()):
        method_type = 'categorical'
    else:
        method_type = 'numerical'

    print(method_type)

    if method_type == "categorical":
        methods = ["SVM","Decision Trees","KNNs","Logistic Regression","Naive Bayes"]
    elif method_type == "numerical":
        methods = ["SVM","Ridge","Lasso"]


    if method_type == "categorical":
        print ("GENETIC ALGORITHM FOR FEATURE SELECTION (CLASSIFICATION):")

        #####
        #SETING UP THE GENETIC ALGORITHM and CALCULATING STARTING POOL (STARTING CANDIDATE POPULATION)
        #####
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(list_inputs))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        def evalOneMax(individual):
            return sum(individual),

        toolbox.register("evaluate", evalOneMax)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        NPOPSIZE = 50 #RANDOM STARTING POOL SIZE
        population = toolbox.population(n=NPOPSIZE)


    #####
    #ASSESSING GINI ON THE STARTING POOL
    #####
    dic_gini={}
    for i in range(np.shape(population)[0]):

        # TRASLATING DNA INTO LIST OF VARIABLES (1-81)
        var_model = []
        for j in range(np.shape(population)[0]):
            if (population[i])[j]==1:
                var_model.append(list(list_inputs)[j])

        # ASSESSING GINI INDEX FOR EACH INVIVIDUAL IN THE INITIAL POOL

        X_train=df[var_model]
        Y_train=df[output_var]

        ######
        # CHANGE_HERE - START: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
        #####
        if "Logistic Regression" in methods:
            lr = sm.Logit(Y_train, X_train)
            model=lr.fit()
            Y_predict=model.predict(X_train)
        ######
        # CHANGE_HERE - END: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
        #####


        ######
        # CHANGE_HERE - START: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
        #####
            fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_predict)
            auc = metrics.auc(fpr, tpr)
            gini_power = abs(2*auc-1)
        ######
        # CHANGE_HERE - END: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
        #####

            gini=str(gini_power)+";"+str(population[j]).replace('[','').replace(', ','').replace(']','')
            dic_gini[gini]=population[j]
        list_gini=sorted(dic_gini.keys(),reverse=True)


    ####
    # ASSESSING RMSE ON THE STARTING POOL
    ####
    if method_type == "numerical":
        X_train=df[var_model]
        Y_train=df["if_var_73"]

        names = list(X_train)
        ranks = {}

        lr = LinearRegression(normalize=True)
        lr.fit(X_train, Y_train)
        ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)

        ridge = Ridge(alpha=7)
        ridge.fit(X_train, Y_train)
        ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)


        lasso = Lasso(alpha=.05)
        lasso.fit(X_train, Y_train)
        ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)


        rlasso = RandomizedLasso(alpha=0.04)
        rlasso.fit(X_train, Y_train)
        ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)

        rf = RandomForestRegressor()
        rf.fit(X_train,Y_train)
        ranks["RF"] = rank_to_dict(rf.feature_importances_, names)

        f, pval  = f_regression(X_train, Y_train, center=True)
        ranks["Corr."] = rank_to_dict(f, names)

        r = {}
        for name in names:
            r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)



        methods = sorted(ranks.keys())
        ranks["Mean"] = r
        methods.append("Mean")
        print(ranks["Mean"])

        print("\t\t%s" % "\t".join(methods))
        for name in names:
            print ("%s\t%s" % (name, "\t".join(map(str,
                [ranks[method][name] for method in methods]))))

        ranks_f = pd.DataFrame(ranks)
        ranks_f.sort_values("RF",0,0,inplace = True)

        print(ranks_f)

        featureset = ranks_f.index.values[0:5]

        print(featureset)

    if method_type == "categorical":
        #GENETIC ALGORITHM MAIN LOOP - START
        # - ITERATING MANY TIMES UNTIL NO IMPROVMENT HAPPENS IN ORDER TO FIND THE OPTIMAL SET OF CHARACTERISTICS (VARIABLES)
        #####
        sum_current_gini=0.0
        sum_current_gini_1=0.0
        sum_current_gini_2=0.0
        first=0
        OK = 1
        a=0
        while OK:  #REPEAT UNTIL IT DO NOT IMPROVE, AT LEAST A LITLE, THE GINI IN 2 GENERATIONS
            a=a+1
            print('loop ', a)
            OK=0

            ####
            # GENERATING OFFSPRING - START
            ####
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1) #CROSS-X PROBABILITY = 50%, MUTATION PROBABILITY=10%
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population =toolbox.select(offspring, k=len(population))
            ####
            # GENERATING OFFSPRING - END
            ####

            sum_current_gini_2=sum_current_gini_1
            sum_current_gini_1=sum_current_gini
            sum_current_gini=0.0

            #####
            #ASSESSING GINI ON THE OFFSPRING - START
            #####
            for j in range(np.shape(population)[0]):
                if population[j] not in dic_gini.values():
                    var_model = []
                    for i in range(np.shape(population)[0]):
                        if (population[j])[i]==1:
                            var_model.append(list(list_inputs)[i])

                    X_train=df[var_model]
                    Y_train=df[output_var]

                    ######
                    # CHANGE_HERE - START: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
                    #####
                    lr = sm.Logit(Y_train, X_train)
                    model=lr.fit()
                    Y_predict=model.predict(X_train)
                    ######
                    # CHANGE_HERE - END: YOU ARE VERY LIKELY USING A DIFFERENT TECHNIQUE BY NOW. SO CHANGE TO YOURS.
                    #####


                    ######
                    # CHANGE_HERE - START: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
                    #####
                    fpr, tpr, thresholds = metrics.roc_curve(Y_train, Y_predict)
                    auc = metrics.auc(fpr, tpr)
                    gini_power = abs(2*auc-1)
                    ######
                    # CHANGE_HERE - END: HERE IT USES THE DEVELOPMENT GINI TO SELECT VARIABLES, YOU SHOULD A DIFFERENT GINI. EITHER THE OOT GINI OR THE SQRT(DEV_GINI*OOT_GINI)
                    #####

                    gini=str(gini_power)+";"+str(population[j]).replace('[','').replace(', ','').replace(']','')
                    dic_gini[gini]=population[j]
            #####
            #ASSESSING GINI ON THE OFFSPRING - END
            #####

            #####
            #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - START
            #####
            list_gini=sorted(dic_gini.keys(),reverse=True)
            population=[]
            for i in list_gini[:NPOPSIZE]:
                population.append(dic_gini[i])
                gini=float(i.split(';')[0])
                sum_current_gini+=gini
            #####
            #SELECTING THE BEST FITTED AMONG ALL EVER CREATED POPULATION AND CURRENT OFFSPRING - END
            #####

            #HAS IT IMPROVED AT LEAST A LITLE THE GINI IN THE LAST 2 GENERATIONS
            print ('sum_current_gini=', sum_current_gini, 'sum_current_gini_1=', sum_current_gini_1, 'sum_current_gini_2=', sum_current_gini_2)
            if(sum_current_gini>sum_current_gini_1+0.0001 or sum_current_gini>sum_current_gini_2+0.0001):
                OK=1
        #####
        #GENETIC ALGORITHM MAIN LOOP - END
        #####

    if method_type == "categorical":

        gini_max=list_gini[0]
        gini=float(gini_max.split(';')[0])
        features=gini_max.split(';')[1]


        ####
        # PRINTING OUT THE LIST OF FEATURES
        #####
        f=0
        for i in range(len(features)):
            if features[i]=='1':
                f+=1
                print('feature ', f, ':', list(list_inputs)[i])
        print ('gini: ', gini)

        featureset = features

    return(featureset)
