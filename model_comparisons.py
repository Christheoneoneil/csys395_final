from classifier_pipe import classifier_pipeline
import pandas as pd
import numpy as np


def metrics(mod, X_t, y_t):
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
    preds = mod.predict(X_t)
    print(classification_report(preds, y_t)) 


unengineered_dat = pd.read_csv("data/merged_dat.csv")
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
param_grid = {
                 'class__n_estimators': np.arange(10, 110, 10),
                 'class__max_depth': np.arange(1, 10, 1)
             }

model, X_test, y_test = classifier_pipeline(unengineered_dat, "alignment", classifier=rf_classifier, cv_grid=param_grid)
metrics(mod=model, X_t=X_test, y_t=y_test)