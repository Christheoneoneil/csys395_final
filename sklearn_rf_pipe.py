import pandas as pd

def classifier_pipeline(data: pd.DataFrame, target_var: str, classifier):
    """
    creates a random foreset pipeline for classification

    Params: 
    data: pandas data frame used for classification
    target_var: variables name for target var

    Returns: 
    None
    
    """
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import classification_report
    
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
    
    num_pipe = Pipeline([("it_impu", IterativeImputer()), 
                                    ("min_max", MinMaxScaler())])
    cat_pipe = Pipeline([("cat_impu", SimpleImputer(strategy="most_frequent")),
                         ('encoder', OneHotEncoder(handle_unknown="ignore"))])
    
    col_t = ColumnTransformer([("num_transfrom", num_pipe, num_cols), 
                              ('cat_transform', cat_pipe, cat_cols)])
    
    rf_pipe = Pipeline([("col_transform", col_t),
                        ("random_forst_class", classifier)])
    
    rf_pipe.fit(X_train, y_train)
    preds = rf_pipe.predict(X_test)

    print(classification_report(preds, y_test))

unengineered_dat = pd.read_csv("data/merged_dat.csv")
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
classifier_pipeline(unengineered_dat, "alignment", classifier=rf_classifier)
