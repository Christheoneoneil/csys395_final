import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

def classifier_pipeline(data: pd.DataFrame, target_var: str, classifier, cv_grid: dict, oversampler=RandomOverSampler()):
    """
    creates a random foreset pipeline for classification

    Params: 
    data: pandas data frame used for classification
    target_var: variables name for target var
    classifier: provided classifier for pipeline
    cv_grid: parameter grid for cross validation

    Returns: 
    best_mod: best model chosen from cross validation
    X_test: feature data used for testing
    y_test: target data used for testing
    """
    
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from imblearn.pipeline import Pipeline
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.compose import ColumnTransformer
    
    data = data.copy()
    data.dropna(subset=(target_var), inplace=True)
    X = data.drop(target_var, axis="columns")
    y = data[target_var]
  
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                         train_size=.8, 
                                                         random_state=117)
    
    num_cols = list(data.select_dtypes(include="float64").columns)
    cat_cols = list(data.select_dtypes(include="object").columns)
    cat_cols.remove(target_var)
    
    num_pipe = Pipeline(steps=[("it_impu", IterativeImputer()), 
                                    ("min_max", MinMaxScaler())])
    cat_pipe = Pipeline(steps=[("cat_impu", SimpleImputer(strategy="most_frequent")),
                         ('encoder', OneHotEncoder(handle_unknown="ignore"))])
    
    col_t = ColumnTransformer([("num_transfrom", num_pipe, num_cols), 
                              ('cat_transform', cat_pipe, cat_cols)])
    
    
    class_pipe = Pipeline(steps=[("col_trans", col_t),
                                ("samp", oversampler),
                                ("class", classifier)])
    
    grid_clf = RandomizedSearchCV(estimator=class_pipe, 
                                  param_distributions=cv_grid,
                                  scoring="accuracy", cv=10, 
                                  verbose=False)
    grid_clf.fit(X_train, y_train)
    best_mod = grid_clf.best_estimator_
    best_mod.fit(X_train, y_train)

    return best_mod, X_test, y_test
    



