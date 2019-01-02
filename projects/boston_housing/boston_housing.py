#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Model Evaluation & Validation
# ## Project: Predicting Boston Housing Prices
# 
# Welcome to the first project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# In this project, you will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a *good fit* could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/). The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. For the purposes of this project, the following preprocessing steps have been made to the dataset:
# - 16 data points have an `'MEDV'` value of 50.0. These data points likely contain **missing or censored values** and have been removed.
# - 1 data point has an `'RM'` value of 8.78. This data point can be considered an **outlier** and has been removed.
# - The features `'RM'`, `'LSTAT'`, `'PTRATIO'`, and `'MEDV'` are essential. The remaining **non-relevant features** have been excluded.
# - The feature `'MEDV'` has been **multiplicatively scaled** to account for 35 years of market inflation.
# 
# Run the code cell below to load the Boston housing dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[2]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
#from sklearn.cross_validation import ShuffleSplit  ### doesn't work, use below instead
from sklearn.model_selection import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))


# ## Data Exploration
# In this first section of this project, you will make a cursory investigation about the Boston housing data and provide your observations. Familiarizing yourself with the data through an explorative process is a fundamental practice to help you better understand and justify your results.
# 
# Since the main goal of this project is to construct a working model which has the capability of predicting the value of houses, we will need to separate the dataset into **features** and the **target variable**. The **features**, `'RM'`, `'LSTAT'`, and `'PTRATIO'`, give us quantitative information about each data point. The **target variable**, `'MEDV'`, will be the variable we seek to predict. These are stored in `features` and `prices`, respectively.

# ### Implementation: Calculate Statistics
# For your very first coding implementation, you will calculate descriptive statistics about the Boston housing prices. Since `numpy` has already been imported for you, use this library to perform the necessary calculations. These statistics will be extremely important later on to analyze various prediction results from the constructed model.
# 
# In the code cell below, you will need to implement the following:
# - Calculate the minimum, maximum, mean, median, and standard deviation of `'MEDV'`, which is stored in `prices`.
#   - Store each calculation in their respective variable.

# In[3]:


# TODO: Minimum price of the data
minimum_price = prices.min()

# TODO: Maximum price of the data
maximum_price = prices.max()

# TODO: Mean price of the data
mean_price = prices.mean()

# TODO: Median price of the data
median_price = prices.median()

# TODO: Standard deviation of prices of the data
std_price = prices.std()

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))


# ### Question 1 - Feature Observation
# As a reminder, we are using three features from the Boston housing dataset: `'RM'`, `'LSTAT'`, and `'PTRATIO'`. For each data point (neighborhood):
# - `'RM'` is the average number of rooms among homes in the neighborhood.
# - `'LSTAT'` is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
# - `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.
# 
# 
# ** Using your intuition, for each of the three features above, do you think that an increase in the value of that feature would lead to an **increase** in the value of `'MEDV'` or a **decrease** in the value of `'MEDV'`? Justify your answer for each.**
# 
# **Hint:** This problem can phrased using examples like below.  
# * Would you expect a home that has an `'RM'` value(number of rooms) of 6 be worth more or less than a home that has an `'RM'` value of 7?
# * Would you expect a neighborhood that has an `'LSTAT'` value(percent of lower class workers) of 15 have home prices be worth more or less than a neighborhood that has an `'LSTAT'` value of 20?
# * Would you expect a neighborhood that has an `'PTRATIO'` value(ratio of students to teachers) of 10 have home prices be worth more or less than a neighborhood that has an `'PTRATIO'` value of 15?

# **Answer:**
# from matplotlib import pyplot as plt 
# 
# def rm_price_line():
#     rm = data['RM']
#     medv = data['MEDV']
#     plt.title("RM-Price Line")
#     plt.xlabel("RM")
#     plt.ylabel("Price")
#     plt.plot(rm, medv, "ob")
#     plt.show()
# 
# def lstat_price_line():
#     lstat = data['LSTAT']
#     medv = data['MEDV']
#     plt.title("LSTAT-Price Line")
#     plt.xlabel("LSTAT")
#     plt.ylabel("Price")
#     plt.plot(lstat, medv, "ob")
#     plt.show()
# 
# def tratio_price_line():
#     ptratio= data['PTRATIO']
#     medv = data['MEDV']
#     plt.title("Ptratio-Price Line")
#     plt.xlabel("PTRATIO")
#     plt.ylabel("Price")
#     plt.plot(ptratio, medv, "ob")
#     plt.show()
# 
# 
# rm_price_line()
# lstat_price_line()
# tratio_price_line()
# 
# '''
# I have tried to plot each of feature-to-price lines by using the above three functions. I was hoping it will somehow show the results. It relation between input and output is kind of linear we can tell the answer right away. However, it's not easy to tell the answer easily by simply looking the plotted picture, and those X-Y dots scattered randomly. I think that's why we are here to explore more machine learning technics to find a solution to this issue. 
# '''
# 

# ----
# 
# ## Developing a Model
# In this second section of the project, you will develop the tools and techniques necessary for a model to make a prediction. Being able to make accurate evaluations of each model's performance through the use of these tools and techniques helps to greatly reinforce the confidence in your predictions.

# ### Implementation: Define a Performance Metric
# It is difficult to measure the quality of a given model without quantifying its performance over training and testing. This is typically done using some type of performance metric, whether it is through calculating some type of error, the goodness of fit, or some other useful measurement. For this project, you will be calculating the [*coefficient of determination*](http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination), R<sup>2</sup>, to quantify your model's performance. The coefficient of determination for a model is a useful statistic in regression analysis, as it often describes how "good" that model is at making predictions. 
# 
# The values for R<sup>2</sup> range from 0 to 1, which captures the percentage of squared correlation between the predicted and actual values of the **target variable**. A model with an R<sup>2</sup> of 0 is no better than a model that always predicts the *mean* of the target variable, whereas a model with an R<sup>2</sup> of 1 perfectly predicts the target variable. Any value between 0 and 1 indicates what percentage of the target variable, using this model, can be explained by the **features**. _A model can be given a negative R<sup>2</sup> as well, which indicates that the model is **arbitrarily worse** than one that always predicts the mean of the target variable._
# 
# For the `performance_metric` function in the code cell below, you will need to implement the following:
# - Use `r2_score` from `sklearn.metrics` to perform a performance calculation between `y_true` and `y_predict`.
# - Assign the performance score to the `score` variable.

# In[4]:


# TODO: Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score

score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))


# ### Question 2 - Goodness of Fit
# Assume that a dataset contains five data points and a model made the following predictions for the target variable:
# 
# | True Value | Prediction |
# | :-------------: | :--------: |
# | 3.0 | 2.5 |
# | -0.5 | 0.0 |
# | 2.0 | 2.1 |
# | 7.0 | 7.8 |
# | 4.2 | 5.3 |
# 
# Run the code cell below to use the `performance_metric` function and calculate this model's coefficient of determination.

# In[5]:


# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))


# * Would you consider this model to have successfully captured the variation of the target variable? 
# * Why or why not?
# 
# ** Hint: **  The R2 score is the proportion of the variance in the dependent variable that is predictable from the independent variable. In other words:
# * R2 score of 0 means that the dependent variable cannot be predicted from the independent variable.
# * R2 score of 1 means the dependent variable can be predicted from the independent variable.
# * R2 score between 0 and 1 indicates the extent to which the dependent variable is predictable. An 
# * R2 score of 0.40 means that 40 percent of the variance in Y is predictable from X.

# **Answer:**
# Since R^2 is 0.923, 92.3% percent of the variance in Y is predictable from X, but not successfully or exactly predictable from X cause that requires R^2 to be 1. 

# ### Implementation: Shuffle and Split Data
# Your next implementation requires that you take the Boston housing dataset and split the data into training and testing subsets. Typically, the data is also shuffled into a random order when creating the training and testing subsets to remove any bias in the ordering of the dataset.
# 
# For the code cell below, you will need to implement the following:
# - Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the `features` and `prices` data into training and testing sets.
#   - Split the data into 80% training and 20% testing.
#   - Set the `random_state` for `train_test_split` to a value of your choice. This ensures results are consistent.
# - Assign the train and testing splits to `X_train`, `X_test`, `y_train`, and `y_test`.

# In[6]:


# TODO: Import 'train_test_split'
#from sklearn.cross_validation import train_test_split  ## depricated, use the following library instead 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

# TODO: Shuffle and split the data into training and testing subsets
# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state = 36)

# Success
print("Training and testing split was successful.")


# ### Question 3 - Training and Testing
# 
# * What is the benefit to splitting a dataset into some ratio of training and testing subsets for a learning algorithm?
# 
# **Hint:** Think about how overfitting or underfitting is contingent upon how splits on data is done.

# **Answer: **
# Our goal is to find a model which can better predicts the target based on the existent train data, but we don't want to overfit the model by using the all the train data to test our model which will get us a better training accuracy but worse test accuracy. To get an generalized training accuracy and test accuracy, a better solution is to split the traning data into two sets, one for training and the other for testing. Also, we don't want to underfit the model by acclocating only a small percentage of the data for traning, and a bigger ramain part for testing. I think to give 70% to 80% of the data for traning and remain 30% to 20% percent for testing is a good start to go. 

# ----
# 
# ## Analyzing Model Performance
# In this third section of the project, you'll take a look at several models' learning and testing performances on various subsets of training data. Additionally, you'll investigate one particular algorithm with an increasing `'max_depth'` parameter on the full training set to observe how model complexity affects performance. Graphing your model's performance based on varying criteria can be beneficial in the analysis process, such as visualizing behavior that may not have been apparent from the results alone.

# ### Learning Curves
# The following code cell produces four graphs for a decision tree model with different maximum depths. Each graph visualizes the learning curves of the model for both training and testing as the size of the training set is increased. Note that the shaded region of a learning curve denotes the uncertainty of that curve (measured as the standard deviation). The model is scored on both the training and testing sets using R<sup>2</sup>, the coefficient of determination.  
# 
# Run the code cell below and use these graphs to answer the following question.

# In[7]:


import numpy as np
import pandas as pd
import visuals as vs

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)


# ### Question 4 - Learning the Data
# * Choose one of the graphs above and state the maximum depth for the model. 
# * What happens to the score of the training curve as more training points are added? What about the testing curve? 
# * Would having more training points benefit the model? 
# 
# **Hint:** Are the learning curves converging to particular scores? Generally speaking, the more data you have, the better. But if your training and testing curves are converging with a score above your benchmark threshold, would this be necessary?
# Think about the pros and cons of adding more training points based on if the training and testing curves are converging.

# **Answer: **
# I would choose graph with max_depth = 3, and max_depth for the model is ???
# When more training points are added, the training curve tends to get closer to one stable valaue and the same trends happens for testing curve and testing curve is also getting closer to a stable value if the enough training points are added;
# Even though more than enough points are added, it does not benefit the model, for example the below 2 pictures shows that the training curve and testing curve will never meet or even getting more closer to each other even if more than enough trainning points are added.

# ### Complexity Curves
# The following code cell produces a graph for a decision tree model that has been trained and validated on the training data using different maximum depths. The graph produces two complexity curves — one for training and one for validation. Similar to the **learning curves**, the shaded regions of both the complexity curves denote the uncertainty in those curves, and the model is scored on both the training and validation sets using the `performance_metric` function.  
# 
# ** Run the code cell below and use this graph to answer the following two questions Q5 and Q6. **

# In[8]:


import numpy as np
import pandas as pd
import visuals as vs
from sklearn.model_selection import train_test_split 

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state = 36)

vs.ModelComplexity(X_train, y_train)


# ### Question 5 - Bias-Variance Tradeoff
# * When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance? 
# * How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?
# 
# **Hint:** High bias is a sign of underfitting(model is not complex enough to pick up the nuances in the data) and high variance is a sign of overfitting(model is by-hearting the data and cannot generalize well). Think about which model(depth 1 or 10) aligns with which part of the tradeoff.

# **Answer: **
# High bias causes underfitting and high variance causes overfitting. According to the graph above, when modle is  trained with a maximum depth of 1, the model gets lower score and it is underfitting, so it suffers from high bias. However, it the model is tranined with a maximum of depth of 10, the traning score gets higher and closer to 1, but the testing score or validation score gets further away from trainniing curve which causes overfitting, so it suffers from high variance.
# Visually speaking, i would rather choose some value between 2 and 4, maybe 3 or 4 and i would eventually choose 3.

# ### Question 6 - Best-Guess Optimal Model
# * Which maximum depth do you think results in a model that best generalizes to unseen data? 
# * What intuition lead you to this answer?
# 
# ** Hint: ** Look at the graph above Question 5 and see where the validation scores lie for the various depths that have been assigned to the model. Does it get better with increased depth? At what point do we get our best validation score without overcomplicating our model? And remember, Occams Razor states "Among competing hypotheses, the one with the fewest assumptions should be selected."

# **Answer: **
# As of answer above i would choose 3 as the maximum depth in a model that best generalizes to unseen data. 

# -----
# 
# ## Evaluating Model Performance
# In this final section of the project, you will construct a model and make a prediction on the client's feature set using an optimized model from `fit_model`.

# ### Question 7 - Grid Search
# * What is the grid search technique?
# * How it can be applied to optimize a learning algorithm?
# 
# ** Hint: ** When explaining the Grid Search technique, be sure to touch upon why it is used,  what the 'grid' entails and what the end goal of this method is. To solidify your answer, you can also give an example of a parameter in a model that can be optimized using this approach.

# **Answer: **
# Suppose we have a set of models and each model has different parameter values. We want to find a way to evaluate which (model,parameters) are the best solution for a problem. So the models with different parameters can make a grid, and then we use cross validation to figure out which (model, parameters) wins by having the best score. 
# Grid Search can be applied to improve a specific mode by providing a range of parameters and then fit them back into the model and then reevaluate until the best score is found. 

# ### Question 8 - Cross-Validation
# 
# * What is the k-fold cross-validation training technique? 
# 
# * What benefit does this technique provide for grid search when optimizing a model?
# 
# **Hint:** When explaining the k-fold cross validation technique, be sure to touch upon what 'k' is, how the dataset is split into different parts for training and testing and the number of times it is run based on the 'k' value.
# 
# When thinking about how k-fold cross validation helps grid search, think about the main drawbacks of grid search which are hinged upon **using a particular subset of data for training or testing** and how k-fold cv could help alleviate that. You can refer to the [docs](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) for your answer.

# **Answer: **
# One way of evaluating a model is separate the data into two parts, traning data and testing data, and usually 70-80% for traning data and the remain 20-30% for testing data. But this solution will cause variance. A better solution is to use K-fold cross-validation technique, and K means the number of groups the sub data sets. For instance, if we have 1000 data, and if K is 10, then each sub-data set will be 100. We can use K-1 to K-8 data sets for traning data and K-9 to K-10 for testing data for the first round. Then we use K-1 to K-7 and K10 for traning data, and K-8 to K-9 for testing data. We continue this way until K(10) times traning and testing are performed which will generate 10 scores for each round validation. And then we calculate the mean of the 10 scores to get score for the current evaluated model. 
# When comes to a optimize a model, we can use K-fold cross-validation to evaluate different parameters or parameter ranges, and find a best fit paramerter set for this model. Every time we change the parameters we have fit back into model and use k-fold CV to evaluate the score untill the best score is found. 

# ### Implementation: Fitting a Model
# Your final implementation requires that you bring everything together and train a model using the **decision tree algorithm**. To ensure that you are producing an optimized model, you will train the model using the grid search technique to optimize the `'max_depth'` parameter for the decision tree. The `'max_depth'` parameter can be thought of as how many questions the decision tree algorithm is allowed to ask about the data before making a prediction. Decision trees are part of a class of algorithms called *supervised learning algorithms*.
# 
# In addition, you will find your implementation is using `ShuffleSplit()` for an alternative form of cross-validation (see the `'cv_sets'` variable). While it is not the K-Fold cross-validation technique you describe in **Question 8**, this type of cross-validation technique is just as useful!. The `ShuffleSplit()` implementation below will create 10 (`'n_splits'`) shuffled sets, and for each shuffle, 20% (`'test_size'`) of the data will be used as the *validation set*. While you're working on your implementation, think about the contrasts and similarities it has to the K-fold cross-validation technique.
# 
# Please note that ShuffleSplit has different parameters in scikit-learn versions 0.17 and 0.18.
# For the `fit_model` function in the code cell below, you will need to implement the following:
# - Use [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from `sklearn.tree` to create a decision tree regressor object.
#   - Assign this object to the `'regressor'` variable.
# - Create a dictionary for `'max_depth'` with the values from 1 to 10, and assign this to the `'params'` variable.
# - Use [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) from `sklearn.metrics` to create a scoring function object.
#   - Pass the `performance_metric` function as a parameter to the object.
#   - Assign this scoring function to the `'scoring_fnc'` variable.
# - Use [`GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) from `sklearn.grid_search` to create a grid search object.
#   - Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and `'cv_sets'` as parameters to the object. 
#   - Assign the `GridSearchCV` object to the `'grid'` variable.

# In[9]:


# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    # cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':np.arange(1,11)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# ### Making Predictions
# Once a model has been trained on a given set of data, it can now be used to make predictions on new sets of input data. In the case of a *decision tree regressor*, the model has learned *what the best questions to ask about the input data are*, and can respond with a prediction for the **target variable**. You can use these predictions to gain information about data where the value of the target variable is unknown — such as data the model was not trained on.

# ### Question 9 - Optimal Model
# 
# * What maximum depth does the optimal model have? How does this result compare to your guess in **Question 6**?  
# 
# Run the code block below to fit the decision tree regressor to the training data and produce an optimal model.

# In[10]:


# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))


# ** Hint: ** The answer comes from the output of the code snipped above.
# 
# **Answer: **
# The max_depth is 4. From question 6, i was considering choosing either 3 or 4 and eventually i chose 3.  

# ### Question 10 - Predicting Selling Prices
# Imagine that you were a real estate agent in the Boston area looking to use this model to help price homes owned by your clients that they wish to sell. You have collected the following information from three of your clients:
# 
# | Feature | Client 1 | Client 2 | Client 3 |
# | :---: | :---: | :---: | :---: |
# | Total number of rooms in home | 5 rooms | 4 rooms | 8 rooms |
# | Neighborhood poverty level (as %) | 17% | 32% | 3% |
# | Student-teacher ratio of nearby schools | 15-to-1 | 22-to-1 | 12-to-1 |
# 
# * What price would you recommend each client sell his/her home at? 
# * Do these prices seem reasonable given the values for the respective features? 
# 
# **Hint:** Use the statistics you calculated in the **Data Exploration** section to help justify your response.  Of the three clients, client 3 has has the biggest house, in the best public school neighborhood with the lowest poverty level; while client 2 has the smallest house, in a neighborhood with a relatively high poverty rate and not the best public schools.
# 
# Run the code block below to have your optimized model make predictions for each client's home.

# In[11]:


# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))


# **Answer: **
# Predicted selling price for Client 1's home: $417,822.22
# Predicted selling price for Client 2's home: $231,118.87
# Predicted selling price for Client 3's home: $896,000.00
# To me those prices that were pridicted by the model it pretty resonable given the respective features.

# ### Sensitivity
# An optimal model is not necessarily a robust model. Sometimes, a model is either too complex or too simple to sufficiently generalize to new data. Sometimes, a model could use a learning algorithm that is not appropriate for the structure of the data given. Other times, the data itself could be too noisy or contain too few samples to allow a model to adequately capture the target variable — i.e., the model is underfitted. 
# 
# **Run the code cell below to run the `fit_model` function ten times with different training and testing sets to see how the prediction for a specific client changes with respect to the data it's trained on.**

# In[12]:


vs.PredictTrials(features, prices, fit_model, client_data)


# ### Question 11 - Applicability
# 
# * In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.  
# 
# **Hint:** Take a look at the range in prices as calculated in the code snippet above. Some questions to answering:
# - How relevant today is data that was collected from 1978? How important is inflation?
# - Are the features present in the data sufficient to describe a home? Do you think factors like quality of apppliances in the home, square feet of the plot area, presence of pool or not etc should factor in?
# - Is the model robust enough to make consistent predictions?
# - Would data collected in an urban city like Boston be applicable in a rural city?
# - Is it fair to judge the price of an individual home based on the characteristics of the entire neighborhood?

# **Answer: **
# 
# I will definitely not recommend this model to be used in real-world setting.
# 
# We are pretty sure that the house price as of today has changed dramatically compared to that of 1978. 
# 
# Those feature are important to describe a home but not sufficient enough, and there are more features like lot size, square footage, location and nearby by elementary school, middle school and high school, especially the sales price over time and market housing price over time, etc. The more feature we count in, the more accurate result we are going to get. 
# 
# This model seems to be robust to make predictions but not reliable for real-world setting.
# 
# Data collected from urban city like Boston is irrelevant to predict house prices in a rural city. 
# 
# No, it is not fair to judge the price of an individual home value based on the entire neighbourhood. We have to take into account of selling house's characteristics, like lot size, square footage, brand new house or very old house which needs to be refurbished before moving in. 
# 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
