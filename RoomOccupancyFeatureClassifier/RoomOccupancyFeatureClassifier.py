#!/usr/bin/env python
# coding: utf-8

# # **The goal of this homework is to practically study metrics of classifier accuracy.**
# 

# We will use UCI Machine Learning Repository for teh dataset used in this assignment. https://archive.ics.uci.edu/
# The following code will install a python package and download the dataset

# In[1]:


get_ipython().system('pip install ucimlrepo')


# In[2]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
occupancy_detection = fetch_ucirepo(id=357) 
  
# data (as pandas dataframes) 
X = occupancy_detection.data.features 
y = occupancy_detection.data.targets 
  
# metadata 
print(occupancy_detection.metadata) 
  
# variable information 
print(occupancy_detection.variables) 


# # Inspect the dataset in debugger (0 pts)

# In[3]:


#Example image



# In[4]:


import matplotlib.pyplot as plt

# Example: Plotting the Temperature column
plt.figure(figsize=(10, 5))
plt.plot(X['Temperature'], label='Temperature', color='blue')
plt.xlabel('Index')
plt.ylabel('Temperature')
plt.title('Temperature Plot')
plt.legend()
plt.grid(True)
plt.show()


# # Graduate/Undergraduate assignment (90 pts)
# 
# Study the “Occupancy Detection Data Set”. The class label in this dataset is 0/1, based on whether an office was occupied or not. Split the dataset into training and validation subsets. 
# 
# Develop a simple rule-based classifier, that will work similarly to the fish classifier discussed in lecture. Use one feature a time. 
# 
# Using the data in the training dataset and K-fold cross validation, find the optimal decision boundary that minimizes classification error. The K-fold cross validation should be implemented without using built-in functions from Python libraries. This means all code must be yours!
# 
# Implement this code as a function. Apply to all features and for each feature, compute and print the error measures: Error rate, Recall, Precision, Specificity, False alarm rate. Also compute and plot the ROC and (on the same graph) display the AUC.
# 
# 

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

"""
Takes in a string and determines if it numeric.

"""
def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

"""
Takes in a DataFrame with feature data, DataFrame of occupancy data, set of features to test, and a integer k for number of folds.
Returns the best feature, best threshold, and direction of comparison.

"""
    
def train_model(X, y, features, k):
    #Prepare the folds
    remainder = len(X) % k
    length = len(X) // k
    folds = []
    last = 0
    maxRoc = 0
    returnFeature = None
    returnThreshold = None
    rule = None
    for _ in range(0, k - remainder):
        folds.append([last, last + length])
        last += length
    for _ in range(k - remainder, k):
        folds.append([last, last + length + 1])
        last += length + 1
        
    #Iterate through all features and test individually
    
    for feature in features:
        
        # Ensure numeric conversion for the feature values and labels
        
        trainData = X[feature].values.astype(float)
        resultData = y['Occupancy'].values.astype(int)
        thresholds = sorted(list(set(trainData)))
        bestThreshold = None
        bestAvgError = float('inf')
        
        #Iterate through possible thresholds (unique values of feature)
        
        for threshold in thresholds:
            totalError = 0.0
            
            #Iterate through folds
            
            for i in range(0, k):
                m, n = folds[i][0], folds[i][1]
                currTrainData = np.concatenate((trainData[:m], trainData[n:]))
                currTrainResult = np.concatenate((resultData[:m], resultData[n:]))
                currValidateData = trainData[m:n]
                currValidateResult = resultData[m:n]
                
                #Train the model
                
                predictionsGEQ = (currTrainData >= threshold)
                errorGEQ = np.mean(predictionsGEQ != currTrainResult)
                predictionsLess = (currTrainData < threshold)
                errorLess = np.mean(predictionsLess != currTrainResult)
                
                #Evaluate threshold on validation data
                
                if errorGEQ < errorLess:
                    valPredictions = (currValidateData >= threshold)
                else:
                    valPredictions = (currValidateData < threshold)
                
                foldError = np.mean(valPredictions != currValidateResult)
                totalError += foldError
                
            avgError = totalError / k
            if avgError < bestAvgError:
                bestAvgError = avgError
                bestThreshold = threshold

        predictionsGEQ = (trainData >= bestThreshold)
        errorGEQ = np.mean(predictionsGEQ != resultData)
        predictionsLess = (trainData < bestThreshold)
        errorLess = np.mean(predictionsLess != resultData)
        if errorGEQ < errorLess:
            valPredictions = predictionsGEQ
            chosen_rule = 'GEQ'
        else:
            valPredictions = predictionsLess
            chosen_rule = 'LESS'
        
        #Compute and plot error rates for each feature
        
        tp = np.sum((valPredictions == 1) & (resultData == 1))
        fp = np.sum((valPredictions == 1) & (resultData == 0))
        tn = np.sum((valPredictions == 0) & (resultData == 0))
        fn = np.sum((valPredictions == 0) & (resultData == 1))
        
        error_rate = (fp + fn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"\nFeature: {feature}")
        print(f"Optimal Threshold: {bestThreshold}, Direction: {chosen_rule}")
        print(f"Error Rate: {error_rate}")
        print(f"Recall (Sensitivity): {recall}")
        print(f"Precision: {precision}")
        print(f"Specificity: {specificity}")
        print(f"False Alarm Rate: {false_alarm_rate}")
        
        
        if chosen_rule == 'LESS':
            scores = -trainData  
        else:
            scores = trainData
        
        
        fpr, tpr, roc_thresholds = roc_curve(resultData, scores)
        roc_auc = auc(fpr, tpr)
        if roc_auc > maxRoc:
            maxRoc = roc_auc
            returnFeature = feature
            returnThreshold = bestThreshold
            rule = chosen_rule
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.title(f'ROC Curve for Feature: {feature}')
        plt.legend(loc="lower right")
        plt.text(0.6, 0.2, f'AUC = {roc_auc}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig(f"{feature}_plot")
        plt.show()
    
    return returnFeature, returnThreshold, rule
    
dfX = pd.DataFrame(X)
attributes = set(["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
removed = []
for attribute in attributes:
    for index, value in dfX[attribute].items():
        if not is_numeric(value):
            removed.append(index)
            dfX = dfX.drop(index=index)
                
dfY = pd.DataFrame(y)
dfY = dfY.drop(index=removed)
    
feature, threshold, rule = train_model(dfX, dfY, attributes, 10)


# Determine the best classifier and apply this classiifer to the test data set. Assess classification errors. In the report, provide conclusions about the generalization capabilities of the developed classification model.

# In[6]:


trainData = dfX[feature].values.astype(float)
resultData = dfY['Occupancy'].values.astype(int)
predictions = (trainData >= threshold) if rule == 'GEQ' else (trainData < threshold)
tp = np.sum((predictions == 1) & (resultData == 1))
fp = np.sum((predictions == 1) & (resultData == 0))
tn = np.sum((predictions == 0) & (resultData == 0))
fn = np.sum((predictions == 0) & (resultData == 1))
        
error_rate = (fp + fn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
print(f"\nFeature: {feature}")
print(f"Optimal Threshold: {threshold}, Direction: GEQ")
print(f"Error Rate: {error_rate}")
print(f"Recall (Sensitivity): {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"False Alarm Rate: {false_alarm_rate}")
