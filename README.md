# NEURAL NETWORK CHARITY ANALYSIS
## SUMMARY
The purpose of this analysis is to create a model that will predict whether or not charity applicants will be succesful if funded by the Alphabet Soup foundation.  This analysis will explore the use of deep learning neural network machine learning to create this model.  

###  DATA EXPLORATION

The data provided includes over 34,000 organizations who have applied for funding from Alphabet Soup over the years.  The data fields include various classifiers and an indicator as to whether or not the charity was successful.  Success is measured by whether or not the charity used the investment funds effectively. Our target variable that we are trying to predict is therefore, "IS_SUCCESSFUL."

Possible Features provided in the data include:

*  EIN - business registration identifier - this is unique to each organization so is not predictive and will be dropped
*  NAME - Unique identifer of charity so will be dropped
*  APPLICATION_TYPE - We don't know what the values mean but there seem to be various types of applications so this might be a feature
*  AFFILIATION - Independent vs Company Sponsored - this might be predictive keep
*  Classification - we do not know what the values mean so need to see if they are meaningful or not
*  USE_CASE - several different types here, keep and test
*  ORGANIZATION  - type of organization may or may not be predictive
*  STATUS - Do not know what this is, but most values are 1
*  INCOME_AMT - May be predictive
*  SPECIATION CONSIDERATIONS - Do not know if this is relevant
*  ASK_AMOUNT - This may be relevent


###  Data Preprocessing:  

the following data fields have been removed in order to prepare for the machine learning model because they are unique and therefore redundant for a predictive model:   
1.  EIN 
2.  NAME of organization

A summary of the unique values within each of the remainder possible features is:
unique
![]()

To explore whether any values should be bucketed into classes within these possible features, I looked at the 2 fields that have more than 10 unique values. 

The APPLICATION_TYPE field has 17 unique classes.  
buckets
![]()

The preponderance of the data is in class T3.  With over 27,000 observations, this data point leaves a hump far out on the density function.
density
![]()
To reduce the variation in this class, I have bucketed all classes with counts less than 500 into an "other" bucket.

The other object variable with more than 10 unique instances is the CLASSIFICATION variable.  I have reduced this by bucketing classes with fewer than 1800 values into "Other".

Class
![]()

In order to prepare the remaining variables for the machine learning model, I have encoded the remaining categorical variables using OneHotEncoder.  This results in 41 feature columns.  The encoded categorical values are then merged with the remainder of the original dataset and the original redundant encoded variables are deleted.  This leaves us with 44 features in our dataset. 

After pre-processing, the data is split into a target array using the variable "IS_SUCCESSFUL" and all the other features are in the X dataset.  These are converted to value arrays and are then ready for running through the model after splitting into training and test datasets.  The final step is scaling the data.  

##  MODEL SET-UP

I have used deep neural net model with 2 hidden node layers.  The 1st hidden layer is assigned 8 neurons whereas the 2nd hidden layer has 4 neurons. I have started with 2 hidden layers because this is the generally accepted level of sufficiency for deep networks (Cybenko 1988).  In addition, this is a fairly simple dataset so there is not much need for an over complicated model. With too many nodes, the model will tend to overfit the training data and then would not perform well on the test data.  

To select the number of neurons in the hidden layer, I selected values that had been used before in the learning module.  8 for first hidden node and 4 for the 2nd hidden node.  Further research revealed a rule of thumb that is wide open, "the optimal size of the hidden layer is usually between the size of the input and size of the output layers."  So this is somewhere between 1 and 43.  To start, however, I wanted to choose a smaller amount so the model didn't take too long to run.  In addition, I chose to run it with only 25 epochs to save time.  While it is running, the model clearly hits the best accuracy very quickly so 25 epochs is reasonble for this model. 

I used 2 activation functions because these were given by the challenge code.  The first hidden node uses the "relu" activation model whereas the output activation model uses "sigmoid."  If these do not perform well, other activation models can be explored.  

This initial model achieves an accuracy rate of .729 which is not very good.  

results
![]()

So to try and improve the model, I have explored various adjustments.

## MODEL ADJUSTMENT TO TRY AND IMPROVE ACCURACY

I have run additional model variations described below:

1.  Bin additional variables.    
  a.  I changed the bin for CLASSIFCATION to put classes with less than 100 in Other (rather than 500)
  b.  USE_CASE - I added a bin for this variable and put classes with less than 5,000 observations into "other"
  c.  ORGANIZATION - I binned those less than 10,000 into other since there are really just two major categories that matter for this variable
  
This extra binning changed overall accuracy to .7259 with loss at .555.  This is slightly worse than the original model so am keeping the original model to make more adjustments to


  extrabinning
  ![]()

2.  Removed variables that don't have much differentiation
    a.  removed "SPECIAL_CONSIDERATIONS' since all are "N" except for 27 "Y" which is too small a percent to be important
    b.  removed "STATUS" because almost all except 5 observations are a "1"
    c.  removed "ASK_AMT" because this variable is almost all 5000 and then numerous other amounts that only have 1 entry
  
  Keeping the binning as the original and removing this two variables results in accuracy of:  .7296
  
  
  dropVariable
  ![]()
  
3.  Try different activation functions for the hidden layers

 Using the dropvariable model as a base since the best performance so far, try to run different activation functions. The original activation model was "relu".  Running it with

    *  relu - Applies the rectified linear unit activation function
    *  tanh  - hyperbolic tangent activation function.   - accuracy = .728  loss = .555  no improvement
    *  Sigmoid - accuracy = .7304  loss = .555  best performing so far

4.  Try adding neurons to a hidden layer  (starting with Sigmoid/drop 3 variable model)  
    * increase hidden layer 2 to 8 from 4 - accuracy = .73072  loss = .555 - tiny increase so try increasing layer 1
    *  increase hidden layer 1 from 8 to 12, keep hid 2 at 8  accuracy = .73026, loss = .552  no improvement to accuracy tiny decrease to loww5.
    *  try more neurons on 2nd layer than 1st layer - don't know if this is possible....
   
5.  Add additional hidden layer (keep new neurons on 1 and 2) neurons = 6  accuracy = .7300 so went down a bit  Loss = .554

6.  Increasing Epochs to 100 - don't expect improvement because it reaches the accuracy pretty fast.

Using original 2 hidden layers
increase layer 2 neurons to 8
removed 3 variables
Used Sigmoid activation on hidden layers


Accuracy = .72909

So all the changes that were explored barely moved the dial so this dataset is not reliably predictive of successful charity funding so I would not use this model for those purposes.  


TARGETED MODEL

MOVE APPLICATION TYPE DOWN TO <100 because there is a wide variation of success within this category

Leaves us with just APPLICATION_TYPE, AFFILIATION, CLASSIFICATION AND ORGANIZATION AS FEATURES



 









