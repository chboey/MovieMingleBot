# PCR Classification and RFS Regression

## Running the file

1) Check if your interpreter is missing any libraries or packages, if so
   there is a requirements.txt file, you can use the command
   pip install -r requirements.txt

2a) 
We have provided you a joblib files for each of our models, please
ensure that the *TestDatasetExample.xls* file is present in the parent
folder *COMP3009_Coursework_2_Code_Group_12*, so that you would be able
to run *FinalTestPCR.py* and *FinalTestRFS.py*.

2b) Ensure that you do not change the path of the joblib files as well!
    [AdaBoost_Model.joblib] -> For PCR Classification
    [RandomForest_Regressor_Model] -> For RFS Regression

3) Once you have ensured that you have adhered to steps 1 to 2b, you can run FinalTestPCR.py and FinalTestRFS.py, Happy Testing!! :D

## Classification

1) If you would like to look at our implementation for PCR classification, it would be under a subfolder PCR_Classification,
which contains three file:
    * *GridSearch_PCR_Classification.py*
    * *PCRModelEvaluation.py*
    * *Preprocessing_PCR_Classification.py*

- *GridSearch_PCR_Classification.py*

Contains the implementation of GridSearch CV for Hyperparameter Tuning of Classification Models


- *PCRModelEvaluation.py*

Contains the analysis of the performance of models such as: Balanced Accuracy, Precision and the Accuracy for Negative and Positive Classes. Apart from that, there's visualisation to
capture the number of True Positives, True Negatives, False Positives, False Negatives with Confusion Matrices for each model.

This python file would generate the *AdaBoost_Model.joblib* file.

- *Preprocessing_PCR_Classification.py*

Contains the analysis of Feature Importance using ANNOVA, with dropping rows and without, and the best value of k.

## Regression

1) Implementation for our RFS Regression can be found in the subfolder
RFS_Regression, which contains three file:
    * *GridSearch_RFS_Regression.py*
    * *RFSModelEvaluation.py*
    * *Preprocessing_RFS_Regression.py*
    

- *GridSearch_RFS_Regression.py*

Please note that grid search will take some time. When grid search is completed for a model, the model's best parameters and its corresponding Mean Absolute Error will be printed.

- *RFSModelEvaluation.py*

When model evaluation for a model is complete, a plot for the model's predicted values against the actual values will be made, followed by another plot which shows the model's predicted values against the residual values. The Mean Absolute Error for the model will also be printed. After all model evaluation has been completed, a joblib file for the best model (RandomForestRegressor) will be created, with the filename of RandomForest_Regressor_Model.joblib

- *Preprocessing_RFS_Regression.py*

Two plots will be generated to show feature importance on the dataset using ANOVA f-scores before and after mean imputation.The plot for feature importance after mean imputation will then be used to decide the number of features to be used during feature selection with ANOVA.


## Thank you - Group 12












   

