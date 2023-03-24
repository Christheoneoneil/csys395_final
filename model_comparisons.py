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
                 'class__n_estimators': np.arange(10, 110, 10),
                 'class__max_depth': np.arange(1, 10, 1)
             }
rf_model, X_test, y_test = classifier_pipeline(unengineered_dat, "alignment", classifier=rf_classifier, cv_grid=param_grid)
metrics(mod=rf_model, X_t=X_test, y_t=y_test) 

from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
lr_param_grid = {
                    "class__penalty": ["l2", "none"],
                    "class__max_iter": np.arange(100, 10000, 100),
                    "class__multi_class": ["ovr","auto"]
                }
lr_model, X_test, y_test = classifier_pipeline(unengineered_dat, "alignment", classifier=lr_classifier, cv_grid=lr_param_grid)
metrics(mod=lr_model, X_t=X_test, y_t=y_test)

from sklearn.ensemble import ExtraTreesClassifier
et_classifier = ExtraTreesClassifier()
et_param_grid = {
                    "class__n_estimators": np.arange(50,500,25), 
                }
et_model, X_test, y_test = classifier_pipeline(unengineered_dat, "alignment", classifier=et_classifier, cv_grid=et_param_grid)
metrics(mod = et_model, X_t=X_test, y_t=y_test)


