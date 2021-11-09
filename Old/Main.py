def objective(trial):


    tfidf__analyzer = trial.suggest_categorical('tfidf__analyzer', ['word', 'char', 'char_wb'])
    tfidf__lowercase = trial.suggest_categorical('tfidf__lowercase', [False, True])
    tfidf__max_features = trial.suggest_int('tfidf__max_features', 500, 10_000)
    lgbc__num_leaves = trial.suggest_int('lgbc__num_leaves', 2, 150)
    lgbc__max_depth = trial.suggest_int('lgbc__max_depth', 2, 100)
    lgbc__n_estimators = trial.suggest_int('lgbc__n_estimators', 10, 200)
    lgbc__subsample_for_bin = trial.suggest_int('lgbc__subsample_for_bin', 2000, 300_000)
    lgbc__min_child_samples = trial.suggest_int('lgbc__min_child_samples', 20, 500)
    lgbc__reg_alpha = trial.suggest_uniform('lgbc__reg_alpha', 0.0, 1.0)
    lgbc__colsample_bytree = trial.suggest_uniform('lgbc__colsample_bytree', 0.6, 1.0)
    lgbc__learning_rate = trial.suggest_loguniform('lgbc__learning_rate', 1e-5, 1e-0)

    params = {
        'tfidf__analyzer': tfidf__analyzer,
        'tfidf__lowercase': tfidf__lowercase,
        'tfidf__max_features': tfidf__max_features,
        'lgbc__num_leaves': lgbc__num_leaves,
        'lgbc__max_depth': lgbc__max_depth,
        'lgbc__n_estimators': lgbc__n_estimators,
        'lgbc__subsample_for_bin': lgbc__subsample_for_bin,
        'lgbc__min_child_samples': lgbc__min_child_samples,
        'lgbc__reg_alpha': lgbc__reg_alpha,
        'lgbc__colsample_bytree': lgbc__colsample_bytree,
        'lgbc__learning_rate': lgbc__learning_rate
    }

    model.set_params(**params)

    return - np.mean(cross_val_score(model, X, y, cv=8))


study = optuna.create_study()
study.optimize(objective, timeout=600)

[I 2019-02-25 17:10:36,508]
Finished a trial resulted in value: -0.0669992430578283.
Current best value is -0.0669992430578283 with parameters:
{'tfidf__analyzer': 'word',
'tfidf__lowercase': True,
'tfidf__max_features': 7346,
'lgbc__num_leaves': 88,
'lgbc__max_depth': 77,
'lgbc__n_estimators': 20,
'lgbc__subsample_for_bin': 137472,
'lgbc__min_child_samples': 464,
'lgbc__reg_alpha': 0.9216346635999628,
'lgbc__colsample_bytree': 0.9932423286475682,
'lgbc__learning_rate': 0.025721930853054423}.
[I 2019-02-25 17:10:38,567]
Finished a trial resulted in value: -0.0669992430578283.
Current best value is -0.0669992430578283 with parameters:
{'tfidf__analyzer': 'word',
'tfidf__lowercase': True,
'tfidf__max_features': 7346,
'lgbc__num_leaves': 88,
'lgbc__max_depth': 77,
'lgbc__n_estimators': 20,
'lgbc__subsample_for_bin': 137472,
'lgbc__min_child_samples': 464,
'lgbc__reg_alpha': 0.9216346635999628,
'lgbc__colsample_bytree': 0.9932423286475682,
'lgbc__learning_rate': 0.025721930853054423}.