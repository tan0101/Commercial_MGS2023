# -*- coding: utf-8 -*-
import numpy as np
from numpy.core.numeric import indices
from numpy.lib.index_tricks import s_
import pandas as pd
import sys
import os
import pickle

from sklearn.feature_selection import VarianceThreshold, SelectFromModel, chi2
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_validate, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, cohen_kappa_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC, SMOTE
from pathlib import Path
#from sklearn.pipeline import Pipeline


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category= FutureWarning)
simplefilter(action='ignore', category= UserWarning)
simplefilter(action='ignore', category= DeprecationWarning)

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn),
           'auc': 'roc_auc',
           'acc': make_scorer(accuracy_score),
           'kappa': make_scorer(cohen_kappa_score)}

def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()  

if __name__ == "__main__":
    name_dataset = "Broiler" 
    folder = "Broiler"
    results_folder = "Results"
    data_folder = "Ecoli"
    farms = ["All"] #, "Shandong", "Henan", "Liaoning"]
    type_farm = "Location" #"Farm"
    
    
    data_MGS_ARG_faeces = pd.read_csv('Broiler faeces/Broiler_faeces_ARG_data.csv', header = [0], index_col=[0])
    data_MGS_SA_faeces = pd.read_csv('Broiler faeces/Broiler_faeces_species_abudances_data.csv', header = [0], index_col=[0])
    data_MGS = pd.concat([data_MGS_ARG_faeces, data_MGS_SA_faeces], axis=1)

    data_MGS = data_MGS.fillna(0)
    features_name = data_MGS.columns
    sample_name = data_MGS.index
    
    data_txt = np.array(data_MGS)
    print(data_txt.shape)

    # Load Metadata:
    metadata_df = pd.read_csv("Broiler faeces/Broiler_faeces_metadata.csv", header = [0])
    metadata_samples = metadata_df[metadata_df.columns[0]]

    # Load Antibiotic Data:
    antibiotic_df = pd.read_csv('Broiler faeces/Broiler_faeces_AMR_'+data_folder+'_data_RSI.csv', header = [0])
    
    samples = np.array(antibiotic_df[antibiotic_df.columns[0]])
    print("samples ini length = {}".format(len(samples)))

    order_id = []
    meta_order_id = []
    for s_name in samples:
        idx = np.where(sample_name == s_name)[0]        
        if len(idx)>0:
            order_id.append(idx[0])
        else:
            print(s_name)
            input("cont")

        idx_meta = np.where(metadata_samples == s_name)[0]
        if len(idx_meta)>0:
            meta_order_id.append(idx_meta[0])

    data_txt = data_txt[order_id,:]
    sample_name = sample_name[order_id]

    metadata_df = metadata_df.iloc[meta_order_id, :].reset_index()
    metadata_df.drop(columns="index",axis=1,inplace=True)
    
    print(np.array_equal(samples, antibiotic_df[antibiotic_df.columns[0]]))
    print(np.array_equal(samples, metadata_df[metadata_df.columns[0]]))

    print(data_txt.shape)
    input('cont')
    
    # Nested Cross Validation:
    inner_loop_cv = 3   
    outer_loop_cv = 5
    
    # Number of random trials:
    NUM_TRIALS = 30
    
    # Grid of Parameters:
    C_grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    est_grid = {"clf__n_estimators": [2, 4, 8, 16, 32, 64]}
    MLP_grid = {"clf__alpha": [0.001, 0.01, 0.1, 1, 10, 100], "clf__learning_rate_init": [0.001, 0.01, 0.1, 1],
        "clf__hidden_layer_sizes": [10, 20, 40, 100, 200, 300, 400, 500]}
    SVC_grid = {"clf__gamma": [0.0001, 0.001, 0.01, 0.1], "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    DT_grid = {"clf__max_depth": [10, 20, 30, 50, 100]}
    XGBoost_grid = {"clf__n_estimators": [2, 4, 8, 16, 32, 64], "clf__learning_rate": [0.001, 0.01, 0.1, 1]}
        
    # Classifiers:
    names = ["Logistic Regression", "Linear SVM", "RBF SVM",
        "Extra Trees", "Random Forest", "AdaBoost", "XGBoost"]

    classifiers = [
        LogisticRegression(),
        LinearSVC(loss='hinge'),
        SVC(),
        ExtraTreesClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier()]
    
    for farm_name in farms:
        print("Farm name = {}".format(farm_name))
        print(antibiotic_df.columns[1:])
        for name_antibiotic in antibiotic_df.columns[1:]:
            print("Antibiotic: {}".format(name_antibiotic))
            #if name_antibiotic not in ["CAZ"]:
            #    continue

            target_str = np.array(antibiotic_df[name_antibiotic])
            
            target = np.zeros(len(target_str)).astype(int)
            idx_S = np.where(target_str == 'S')[0]
            idx_R = np.where(target_str == 'R')[0]
            idx_NaN = np.where((target_str != 'R') & (target_str != 'S'))[0]
            target[idx_R] = 1    

            if len(idx_NaN) > 0:
                target = np.delete(target,idx_NaN)
                data = np.delete(data_txt,idx_NaN,axis=0)
                metadata_anti_df = metadata_df.drop(idx_NaN, axis=0).reset_index()
                metadata_anti_df.drop(columns="index",axis=1,inplace=True)
                print("Correct number of isolates: {}".format(len(target)))
            else:
                data = data_txt
                metadata_anti_df = metadata_df.copy()

            if farm_name != "All":
                if isinstance(farm_name, list):
                    farm_ids = []
                    for f_name in farm_name:
                        id_f = list(np.where(metadata_anti_df[type_farm] == f_name)[0])
                        farm_ids += id_f
                else:
                    farm_ids = np.where(metadata_anti_df[type_farm] == farm_name)[0]

                target = target[farm_ids]
                target_str = target_str[farm_ids]
                data = data[farm_ids, :]
                metadata_anti_df = metadata_anti_df.iloc[farm_ids,:].reset_index()
            
            # Check minimum number of samples:
            count_class = Counter(target)
            print(count_class)
            if count_class[0] < 12 or count_class[1] < 12:
                continue

            # Remove low variance:
            print("Before removing low variance:{}".format(data.shape))
            selector = VarianceThreshold(threshold=0)
            selector.fit_transform(data)
            cols=selector.get_support(indices=True)
            data = data[:,cols]
            features_anti = features_name[cols]
            n_features = len(features_anti)
            print("After removing low variance:{}".format(data.shape))
            
            sm = SMOTE(random_state=42)
            data, target = sm.fit_resample(data, target)
            print('Resampled dataset shape %s' % Counter(target))
            
            # Preprocessing - Feature Selection
            std_scaler = MinMaxScaler()
            data = std_scaler.fit_transform(data)
            
            _, pvalue = chi2(data, target)

            threshold = 0.01
            cols_model = np.where(pvalue < threshold)[0]
            coef_sel = pvalue[cols_model]
            print(len(cols_model))
            
            if len(cols_model) == 0:
                continue
                
            features_anti = features_anti[cols_model]
            n_features = len(features_anti)
            data = data[:, cols_model]
            
            print("After select from model:{}".format(data.shape))
            
            # Initialize Variables:
            scores_auc = np.zeros([NUM_TRIALS,len(classifiers)])
            scores_acc = np.zeros([NUM_TRIALS,len(classifiers)])
            scores_sens = np.zeros([NUM_TRIALS,len(classifiers)])
            scores_spec = np.zeros([NUM_TRIALS,len(classifiers)])
            scores_kappa = np.zeros([NUM_TRIALS,len(classifiers)])
            scores_prec = np.zeros([NUM_TRIALS,len(classifiers)])

            # Loop for each trial
            update_progress(0)
            for i in range(NUM_TRIALS):
                #print("Trial = {}".format(i))
            
                inner_cv = StratifiedKFold(n_splits=inner_loop_cv, shuffle=True, random_state=i)
                outer_cv = StratifiedKFold(n_splits=outer_loop_cv, shuffle=True, random_state=i)
            
                k = 0
            
                for name, clf in zip(names, classifiers):
                    model = Pipeline([('clf', clf)])
                    
                    if name == "RBF SVM":
                        grid = SVC_grid              
                    elif name == "Random Forest" or name == "AdaBoost" or name == "Extra Trees":
                        grid = est_grid
                    elif name == "Neural Net":
                        grid = MLP_grid
                    elif name == "Linear SVM":
                        grid = C_grid
                    elif name == "Decision Tree":
                        grid = DT_grid
                    elif name == "XGBoost":
                        grid = XGBoost_grid
                    else:
                        grid = C_grid
        
                    # Inner Search
                    classif = GridSearchCV(estimator=model, param_grid=grid, cv=inner_cv)
                    classif.fit(data, target)
                
                    # Outer Search
                    cv_results = cross_validate(classif, data, target, scoring=scoring, cv=outer_cv, return_estimator=True)

                    tp = cv_results['test_tp']
                    tn = cv_results['test_tn']
                    fp = cv_results['test_fp']
                    fn = cv_results['test_fn']
                    
                    sens = np.zeros(outer_loop_cv)
                    spec = np.zeros(outer_loop_cv)
                    prec = np.zeros(outer_loop_cv)
                    
                    for j in range(outer_loop_cv):
                        TP = tp[j]
                        TN = tn[j]
                        FP = fp[j]
                        FN = fn[j]
                        
                        # Sensitivity, hit rate, recall, or true positive rate
                        sens[j] = TP/(TP+FN)
                        
                        # Fall out or false positive rate
                        FPR = FP/(FP+TN)
                        spec[j] = 1 - FPR
                        if TP + FP > 0:
                            prec[j] = TP / (TP + FP)
        
                    scores_sens[i,k] = sens.mean()
                    scores_spec[i,k] = spec.mean()
                    scores_prec[i,k] = prec.mean()
                    scores_auc[i,k] = cv_results['test_auc'].mean()
                    scores_acc[i,k] = cv_results['test_acc'].mean()
                    scores_kappa[i,k] = cv_results['test_kappa'].mean()
                    
                    k = k + 1
                    
                update_progress((i+1)/NUM_TRIALS)

            results = np.zeros((12,len(classifiers)))
            scores = [scores_auc, scores_acc, scores_sens, scores_spec, scores_kappa, scores_prec]
            for counter_scr, scr in enumerate(scores):
                results[2*counter_scr,:] = np.mean(scr,axis=0)
                results[2*counter_scr + 1,:] = np.std(scr,axis=0)
                
            names_scr = ["AUC_Mean", "AUC_Std", "Acc_Mean", "Acc_Std", 
                "Sens_Mean", "Sens_Std", "Spec_Mean", "Spec_Std", 
                "Kappa_Mean", "Kappa_Std", "Prec_Mean", "Prec_Std"]

            if isinstance(farm_name, list):
                farm_full_name = ' '.join(farm_name)
            else:
                farm_full_name = farm_name
                
            directory = folder+"/"+results_folder+"/"+data_folder+"/Faeces/Chi2 - "+farm_full_name
            if not os.path.exists(directory):
                os.makedirs(directory)

            results_df=pd.DataFrame(results, columns=names, index=names_scr)
            results_df.to_csv(directory+"/SMOTE_results_"+name_dataset+"_"+name_antibiotic+".csv")

            df_auc = pd.DataFrame(scores_auc, columns=names)
            df_auc.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_auc.csv")
            
            df_acc = pd.DataFrame(scores_acc, columns=names)
            df_acc.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_acc.csv")
            
            df_sens = pd.DataFrame(scores_sens, columns=names)
            df_sens.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_sens.csv")
            
            df_spec = pd.DataFrame(scores_spec, columns=names)
            df_spec.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_spec.csv")
            
            df_kappa = pd.DataFrame(scores_kappa, columns=names)
            df_kappa.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_kappa.csv")
            
            df_prec = pd.DataFrame(scores_prec, columns=names)
            df_prec.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_prec.csv")
            
            df_features = pd.DataFrame(coef_sel, columns = ["p-value chi2"], index=features_anti)
            df_features.to_csv(directory+"/features_"+name_dataset+"_"+name_antibiotic+".csv")
