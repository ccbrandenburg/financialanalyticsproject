===============================
iembdfa
===============================

Module Names

DataCleaning.py

AutoInterpolation.py

DateCatVar.py

GeneticFeature.py

RatioVar.py

Included in this module are 5 libraries that will help you during your data science adventures and help you save some of that valuable time you would rather spend on modelling rather than on data cleaning.

1. DataCleaning - A.1) Automated Data Cleaning; identify invalid values and/or rows and automatically solve the problem- NAN, missing, outliers, unreliable values, out of the range, automated data input.

import DataCleaning

Description: 

The autoclean function includes one parameter i.e. pandas dataframe. It automatically detects the numeric and string variables in your dataset and fills na based on the type of columns. For 'int64' and 'float64' type variables, it impute the NaN values with median value of that column and for the string objects, it imputes the NaN values with mode of that particular column.

Once the missing values in your dataset are treated, the function treats the outliers in your dataset using the basic rule of statistics. The rule states that if a particular value is 2.5 standard deviations away from the mean, the value will be treated as an outlier. The same rule is applied by the autoclean function. All the values detected as an outlier are imputed with the median value of the column in which the outlier exists.

Next to treating outliers, the function looks for any unreliable values in your dataset. It mainly works with numeric columns and detect for the percentage of negative values in a particular column. If the percentage of negative values in a columns in your dataset is less than equal to 0.1 percent of the total values in the column, then the negative value will be converted in positive values.

In the last part of the function, the function converts all the categorical variables into numeric.

2. AutoInterpolation - A.4.3) Automated Interpolation transformation.

import AutoInterpolation

Description: 

The AutoInterpolation package automatically detects for date variables in the dataset. It then divides the dataset into two lists. One list containing the columns having null values and the other containing the columns having non null values. The columns having null values are further divided into rows with null values and rows not containing null values. Once the above steps are performed, the non null values are used to interpolate the null values.

3. GeneticFeature - A.7) Characteristics/Feature selection - Stepwise and Genetic Algorithm

import GeneticFeature


4. DateCatVar - H.2) Human assisted Data preprocessing and transformation for modelling - Text processing and Dates processing into variables that can be used in modelling.

import DateCatVar

Description: 
![Alt text](http://i.imgur.com/Dq38eb0.png "Initial Diagram")

DateCatVar program consist of a series of human assisted functions to perform Data Preprocessing and transformation for modelling. The program takes one parameter i.e. pandas dataframe and performs a series of human assited fucntions which includes identification of date columns and adding multiple date columns to perform further transformation.

Once the date columns on which transformation are to be performed selected, value from 1 to 6 can be used to get the parts of date, for example: year, month, day, weekday and week number.

The final part of the program converts the categorical variables into numerical. 

5. RatioVar - H.4) Human assisted variables and ratios creation. Create a list of possible actions that could be taken and create an user interface for a human to decide what to do.

import RatioVar

Description: RatioVar program consists of a series of human assisted steps which asks to select the columns for which different type of operations are to be performed. Once the columns are selected, the type of operation canbe selected to be performed.


* Free software: MIT license
* Documentation: https://iembdfa.readthedocs.io.


Features
--------

* TODO

Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
