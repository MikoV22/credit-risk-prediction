from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Trenuje model regresji logistycznej z wagami zbalansowanymi wg klas.
    
    Finalnie rekomendowany model w tym projekcie - patrz README, sekcja "Porównanie modeli" 
    """
    model = LogisticRegression(class_weight='balanced', random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=42):
    """
    Trenuje Random Forest z domyślnymi hiperparametrami

    Model testowy/porównawczy
    """
    model = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    model.fit(X_train, y_train)
    return model

def tune_random_forest(X_train, y_train, random_state=42):
    """
    Dostraja hiperparametry Random Forest przez GridSearchCV (5-fold CV, scoring=roc_auc)

    Model testowy/porównawczy
    """
    param_grid ={
        'n_estimators':[100,200,300],
        'max_depth':[5,10,15,None],
        'min_samples_split':[2,5,10],
        'min_samples_leaf':[1,2,4]
    }

    rf_base = RandomForestClassifier(class_weight='balanced', random_state=random_state)

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_