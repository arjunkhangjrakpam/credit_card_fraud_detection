# credit_card_fraud_detection
Logistic regression was used to predict the label for the transactions. The data was highly imbalanced.Hence I tried 
oversampling techniques like random oversampling, SMOTE(Synthetic Minority Oversampling Technique) and 
SMOTEEN(Combination of SMOTE and Edited Nearest Neighbors Undersampling). The best Area under the ROC curve was
0.973780 which is from SMOTE 

I used predict.proba() funcition to get the probabilities of each transaction and binarized them choosing the threshold 
for binarization at 0.2(by default it is 0.5) along with SMOTE which gives the optimal sensitivity and specificity.

Accuracy of the model is 0.9982. The dataset is highly skewes with 99.827 % of labels as non-fraud and 0.172 % of labels
as fraudulent. But accuracy is not a good metric for measuring the performance of a model especially when the data is 
highly skewed or dataset is highly imbalanced as in this case. A better metric would be a high Recall and F1 score with 
high sensitivity along with high AUC score.
