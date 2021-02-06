## NEURAL NETWORK CHARITY ANALYSIS
### SUMMARY
The purpose of this analysis is to create a neural network model that will help predict whether or not charity applicants will be succesful if funded by the Alphabet Soup foundation.

###  DATA EXPLORATION

The data provided includes over 34,000 organizations who have applied for funding from Alphabet Soup over the years.  The data fields include various classifiers and an indicator as to whether or not the charity was successful.  Success is measured as whether or not the charity used the investment funds effectively.  

Our target variable that we are trying to predict is "IS_SUCCESSFUL."

Possible Features include:
EIN - business registration identifier - this is unique to each organization so is not predictive and will be dropped
NAME - Unique identifer of charity so will be dropped
APPLICATION_TYPE - We don't know what the values mean but there seem to be various types of applications so this might be a featuer
AFFILIATION - Independent vs Company Sponsored - this might be predictive keep
Classification - we do not know what the values mean so need to see if they are meaningful or not
USE_CASE - several different types here, keep and test
ORGANIZATION  - type of organization may or may not be predictive
STATUS - Do not know what this is
INCOME_AMT - KEEP
SPECIATION CONSIDERATIONS - Do not know if this is relevant
ASK_AMOUNT - this should be relavent


Data Preprocessing:  

Unnecessary data fields have been removed in order to prepare for machine learning model: 
1.  EIN 
2.  NAME of organization

The APPLICATION_TYPE field is examined to see if any values should be bucketed together.
buckets
![]()

The preponderance of the data is in class T3.  With over 27,000 observations, this data point leaves an hump far out on the density function.
density
![]()
I have bucketed all classes with counts less than 500 into an "other" bucket.

The other object variable with more than 10 unique instances is the CLASSIFICATION variable.  I have reduced this by bucketing classes with fewer than 1800 values into "Other".

Class
![]()

# MODEL ADJUSTMENT TO TRY AND IMPROVE ACCURACY

Original model accuracey .7290
Original model Loss  .55785

1.  Bin additional variables.  
  a.  changed bin for CLASSIFCATION to put classes with less than 100 in Other, otherwise keep their own
  b.  USE_CASE - binned classes with less than 5,000 observations into "other"
  c.  ORGANIZATION - binned those less than 10,000 into other since there are really just two major categories that matter for this variable
  
  This extra binning changed overall accuracy to .7259 with loss at .555
  extrabinning
  ![]()

2.  Removed variables that don't have much differentiation
  a.  removed "SPECIAL_CONSIDERATIONS' since all are "N" except for 27 "Y" which is too small a percent to be important
  b.  removed "STATUS" because almost all except 5 observations are a "1"
  C.  removed "ASK_AMT" because this variable is almost all 5000 and then numerous other amounts that only have 1 entry
  
  Keeping the binning as the original and removing this two variables results in accuracy of:  .7296
  
  
  dropVariable
  ![]()
  
3.  Try different activation functions.  









