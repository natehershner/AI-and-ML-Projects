import bnlearn as bn
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from tabulate import tabulate

def batch_predict(model, test_data, batch_size=1000):
    """Splits test data into smaller batches to speed up predictions."""
    predictions = []
    
    for i in range(0, len(test_data), batch_size):
        batch = test_data.iloc[i:i+batch_size]
        batch_pred = bn.predict(model, batch, variables=['DATE_DIED'])
        predictions.append(batch_pred)
    
    return pd.concat(predictions, axis=0)

def predict_covid_death_probability():
    df = pd.read_csv("Covid_Data.csv")
    df.replace([97, 99], np.nan, inplace=True)
    df["DATE_DIED"] = df["DATE_DIED"].apply(lambda x: 0 if x == "9999-99-99" else 1)
    BOOL_FEATURES = [
    "PNEUMONIA", "PREGNANT", "DIABETES", "COPD", "ASTHMA", "INMSUPR",
    "HIPERTENSION", "CARDIOVASCULAR", "RENAL_CHRONIC", "OTHER_DISEASE",
    "OBESITY", "TOBACCO", "INTUBED", "ICU", "SEX", "PATIENT_TYPE"
    ]

    df[BOOL_FEATURES] = df[BOOL_FEATURES].replace({2: 0})
    CATEGORICAL_FEATURES = ["SEX", "CLASIFFICATION_FINAL", "PATIENT_TYPE", "USMER", "MEDICAL_UNIT"]
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype("category")
    df.fillna(df.mode().iloc[0], inplace=True)
    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    model = bn.parameter_learning.fit(model, df, methodtype='bayes')
    k = 10  
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    accuracy_scores = []
    
    for train_index, test_index in kfold.split(df):
        
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        
        modelPartial = bn.structure_learning.fit(train_data)
        
        modelPartial = bn.parameter_learning.fit(modelPartial, train_data)
       
        predictions = bn.predict(modelPartial, test_data, variables=['DATE_DIED'])
        true_values = test_data["DATE_DIED"]
        
        tn, fp, fn, tp = confusion_matrix(true_values, predicted_values).ravel()
        
       
        print(f"Confusion Matrix for this fold:\n[[{tn} {fp}]\n [{fn} {tp}]]")
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Negatives (TN): {tn}")
      
        accuracy = accuracy_score(test_data["DATE_DIED"], predictions["DATE_DIED"])
     
        accuracy_scores.append(accuracy)
        print("Accuracy Score")
        print(accuracy)

    print(f"Accuracy Scores for {k}-fold cross-validation: {accuracy_scores}")
    print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"Standard Deviation: {np.std(accuracy_scores):.4f}")
  
    with open('nul', 'w') as f:
        with redirect_stdout(f):
            cpd = bn.print_CPD(model)
        
    for key in cpd.keys():
        print(tabulate(cpd[key], tablefmt="grid", headers="keys"))

    print("Predicting againts training data")
    predictions = batch_predict(model, df, 10000)
    true_values = df["DATE_DIED"]
    
    predicted_values = predictions["DATE_DIED"]
       
    tn, fp, fn, tp = confusion_matrix(true_values, predicted_values).ravel()
        
    print(f"Confusion Matrix for this fold:\n[[{tn} {fp}]\n [{fn} {tp}]]")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
        
    accuracy = accuracy_score(df["DATE_DIED"], predictions["DATE_DIED"])
    print("Accuracy Score")
    print(accuracy)
    dot = bn.plot_graphviz(model)
    dot.render(f"Covid19_Model1")