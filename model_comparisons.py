from classifier_pipe import classifier_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def metrics(mod, X_t: pd.DataFrame, y_t: pd.DataFrame)->None:
    """
    produces metrics for given model

    Params:
    model: fitted model
    X_t: x testing data
    y_tL y training data

    Returns:
    None

    """
    from sklearn.metrics import classification_report
    from sklearn.metrics import brier_score_loss
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from sklearn.metrics import jaccard_score
    from sklearn.metrics import roc_auc_score

    preds = mod.predict(X_t)
    probs = mod.predict_proba(X_t)
    print(classification_report(preds, y_t)) 
    jacc = jaccard_score(y_t, preds, average = 'macro')
    print('The jaccard score is ' + str(jacc))
    roc = roc_auc_score(y_t, probs, multi_class = 'ovr')
    print('The ROC AUC is ' + str(roc))
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 5))
    cmp = ConfusionMatrixDisplay(
        confusion_matrix(y_t, preds),
        display_labels=["good", "neutral", "bad"]
    )
    
    cmp.plot(ax=ax)
    plt.show()


unengineered_dat = pd.read_csv("data/merged_dat.csv")
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
param_grid = {
                 'class__n_estimators': np.arange(30, 110, 5),
                 'class__max_depth': np.arange(1, 20),
                 'class__max_features': ['sqrt', 'log2', 'None'],
                 'class__criterion' : ['gini', 'entropy'],
             }
rf_model, X_test, y_test = classifier_pipeline(unengineered_dat, "alignment", classifier=rf_classifier, cv_grid=param_grid)
print(rf_model)
metrics(mod=rf_model, X_t=X_test, y_t=y_test) 

from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
lr_param_grid = {
                    "class__penalty": ["l2", "none"],
                    "class__max_iter": np.arange(100, 10000, 100),
                    "class__multi_class": ["ovr","auto"],
                    'class__solver' : ['liblinear','sag', 'saga']
                }
lr_model, X_test, y_test = classifier_pipeline(unengineered_dat, "alignment", classifier=lr_classifier, cv_grid=lr_param_grid)
metrics(mod=lr_model, X_t=X_test, y_t=y_test)

from sklearn.ensemble import ExtraTreesClassifier
et_classifier = ExtraTreesClassifier()
et_param_grid = {
                    "class__n_estimators": np.arange(50,500,5), 
                    'class__max_features' : ['log2','sqrt','auto'],
#                     'class__criterion' : ['gini', 'entropy'],
                    'class__max_depth': np.arange(1,20)
                }
et_model, X_test, y_test = classifier_pipeline(unengineered_dat, "alignment", classifier=et_classifier, cv_grid=et_param_grid)
print(et_model)
metrics(mod = et_model, X_t=X_test, y_t=y_test)

from sklearn.neural_network import MLPClassifier
mlp_classifier = MLPClassifier()
mlp_param_grid = {
#             'class__activation' : ['identity', 'logistic', 'tanh', 'relu'],
#             'class__solver' : ['sgd','adam'],
#             'class__learning_rate': ['constant', 'invscaling', 'adaptive'],
#             'class__max_iter' : np.arange(200,1000,50),
             }
mlp_model, X_test, y_test = classifier_pipeline(unengineered_dat, "alignment", classifier=et_classifier, cv_grid=mlp_param_grid)
metrics(mod = mlp_model, X_t=X_test, y_t=y_test)


