Python 3.6.3rc1 (v3.6.3rc1:d8c174a, Sep 19 2017, 16:39:51) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> def forward_feature_selector(model, x_train, y_train, n_features = None, cv = 5, metric = 'roc_auc', verbose = True):
    
    '''
    Feedforward Feature Selection 
        Input:
            model: 
                The base estimator to build models on.
            x_train: 
                Training data without target
            y_train:
                Training Target labels
            n_features: Default = None
                Max features to extract
            cv: Default = 5
                Number of cross validations to do for CV-Score Calculation
            metric: str, Default = 'roc_auc'
                Metric to be evaluated on. Available metrics supported with sklearn cross_val_score
            verbose: bool, Default = True
                Whether to keep verbosity or not
          
        Returns: 
            list of most useful features
    
    Prerequisites:
    
    >>> import numpy as np
    >>> import multiprocessing
    >>> from joblib import Parallel, delayed
    >>> from sklearn.model_selection import cross_val_score
    '''
      
    useful_columns = []
    columns_indices = list(range(x_train.shape[1]))
    num_cores = multiprocessing.cpu_count()
    
    selecting = True
    i = 1
    
    if verbose:
        print("Starting Forward Feature Selection...")
        print("Number of tasks = {}".format(x_train.shape[1]))
        
    cv_scores = []
    
    while selecting == True:
        
        
        if verbose:
            start = datetime.now()
            print(f'{start}: Features: {i}/{x_train.shape[1]}: ', end = '')


        def model_feat(column_index, useful_columns):   
            
            #list of all the columns to model on
            columns_to_model_on = useful_columns
            columns_to_model_on.append(column_index)

            
            cv_score_column = cross_val_score(model, x_train[:,columns_to_model_on], y_train, cv = cv, scoring = metric)
#             cv_scores_bank[column_index] = cv_score_column.mean()
            
            return (column_index, cv_score_column.mean())
        
        #parallely building the model for feature selection
        cv_scores_bank = Parallel(n_jobs = num_cores)(delayed(model_feat)(col, useful_columns) for col in columns_indices)
            
        cv_scores_bank = dict(cv_scores_bank)
        #fetching index of best column
        highest_cv_column = list(cv_scores_bank.keys())[np.argmax(list(cv_scores_bank.values()))]
        
        #adding the index of useful column to set
        useful_columns.append(highest_cv_column)
        #removing the index of column from column_indices
        columns_indices.remove(highest_cv_column)
        
        best_cv_score = cv_scores_bank[highest_cv_column]
        cv_scores.append(best_cv_score)
        
        if verbose:
            print(f' CV_Score = {best_cv_score}')
            print(f"Time Elapsed = {datetime.now() - start}\n")
        
        if n_features:
            if i == n_features:
                if verbose:
                    print("No further improvement in CV score")
                    print("Stopping further Selection")
                selecting = False 
                
        if i>1:
            if cv_scores[i-1] <= cv_scores[i-2]:
                
                if verbose:
                    print("No further improvement in CV score")
                    print("Stopping further Selection")
                selecting = False
        i += 1
                
    return useful_columns
