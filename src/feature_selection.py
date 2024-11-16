from sklearn.feature_selection import SelectKBest, f_regression

def select_features(X, y, k=5):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected
