# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os

from sklearn.feature_selection import VarianceThreshold, chi2
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pointbiserialr, pearsonr, spearmanr, kendalltau
from scipy import stats
from pathlib import Path

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category= FutureWarning)
simplefilter(action='ignore', category= UserWarning)
simplefilter(action='ignore', category= DeprecationWarning)

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
    reg_type = "Humidity"
    name_dataset = "Broiler" 
    results_folder = "Results"
    type_data =  "Combination"
    data_folder = "Faeces"#type_data
    type_farm = "Location" #"Farm"
    farms = ["All", "Shandong", "Henan", "Liaoning"]
    #farms = [["Pilot1", "Pilot2"], ["HN2", "HN3"], ["SD1", "SD3"], ["Pilot1", "Pilot2", "SD1", "SD3"], "HN2", "HN3", "SD1", "SD3"]#["All", "Shandong", "Henan", "Liaoning"]# ["HN1", "HN2", "HN3", "SD1", "SD2", "SD3", "Pilot1", "Pilot2", "LN2", "LN3"] #
    
    # Load Data:
    if type_data == "Combination":
        data_MGS_ARG_faeces = pd.read_csv('Broiler faeces/Broiler_faeces_ARG_data.csv', header = [0], index_col=[0])
        data_MGS_SA_faeces = pd.read_csv('Broiler faeces/Broiler_faeces_species_abudances_data.csv', header = [0], index_col=[0])
        data_faeces = pd.concat([data_MGS_ARG_faeces, data_MGS_SA_faeces], axis=1)

        data_MGS_ARG_feather = pd.read_csv('Broiler feather/Broiler_feather_ARG_data.csv', header = [0], index_col=[0])
        data_MGS_SA_feather = pd.read_csv('Broiler feather/Broiler_feather_species_abudances_data.csv', header = [0], index_col=[0])
        data_feather = pd.concat([data_MGS_ARG_feather, data_MGS_SA_feather], axis=1)

        data_MGS = pd.concat([data_faeces, data_feather], axis = 0)
        data_MGS = data_MGS.fillna(0)
        features_name = data_MGS.columns
        sample_name = data_MGS.index
        
        data_txt = np.array(data_MGS)
        print(data_txt.shape)
    else:
        data_faeces = pd.read_csv('Broiler faeces/Broiler_faeces_'+type_data+'_data.csv', header = [0], index_col=[0])
        data_feather = pd.read_csv('Broiler feather/Broiler_feather_'+type_data+'_data.csv', header = [0], index_col=[0])
        
        data_MGS = pd.concat([data_faeces, data_feather], axis = 0)
        data_MGS = data_MGS.fillna(0)
        features_name = data_MGS.columns
        sample_name = data_MGS.index
        
        data_txt = np.array(data_MGS)
        print(data_txt.shape) 
        
    # Load Metadata:
    metadata_df_faeces = pd.read_csv("Broiler faeces/Broiler_faeces_metadata.csv", header = [0])
    metadata_df_feather = pd.read_csv("Broiler feather/Broiler_feather_metadata.csv", header = [0])
    metadata_df = pd.concat([metadata_df_faeces, metadata_df_feather], axis=0)
    
    metadata_samples = metadata_df[metadata_df.columns[0]]

    # Load Antibiotic Data:
    antibiotic_df_faeces = pd.read_csv('Broiler faeces/Broiler_faeces_AMR_Ecoli_data_RSI.csv', header = [0])
    antibiotic_df_feather = pd.read_csv('Broiler feather/Broiler_feather_AMR_Ecoli_data_RSI.csv', header = [0])
    antibiotic_df = pd.concat([antibiotic_df_faeces, antibiotic_df_feather], axis=0)
    
    samples = np.array(antibiotic_df[antibiotic_df.columns[0]])
    print("samples ini length = {}".format(len(samples)))

    order_id = []
    meta_order_id = []
    for s_name in samples:
        idx = np.where(sample_name == s_name)[0]
        if len(idx)>0:
            order_id.append(idx[0])

        idx_meta = np.where(metadata_samples == s_name)[0]
        if len(idx_meta)>0:
            meta_order_id.append(idx_meta[0])

    data_txt = data_txt[order_id,:]
    sample_name = sample_name[order_id]

    metadata_df = metadata_df.iloc[meta_order_id, :].reset_index()
    metadata_df.drop(columns="index",axis=1,inplace=True)

    print(data_txt.shape)
    
    args_data = pd.read_csv("CARD_ARG_drugclass_NCBI.csv", header=[0])
    
    for farm_name in farms:    
        
        df_regression = pd.DataFrame()
    
        print(antibiotic_df.columns[1:])
        for name_antibiotic in antibiotic_df.columns[1:]:
            print("Antibiotic: {}".format(name_antibiotic))
            metadata_anti_df = metadata_df.copy()
            data = data_txt

            if farm_name != "All":
                if isinstance(farm_name, list):
                    farm_ids = []
                    for f_name in farm_name:
                        id_f = list(np.where(metadata_anti_df[type_farm] == f_name)[0])
                        farm_ids += id_f
                else:
                    farm_ids = np.where(metadata_anti_df[type_farm] == farm_name)[0]

                data = data[farm_ids, :]
                metadata_anti_df = metadata_anti_df.iloc[farm_ids,:].reset_index()
                
            if isinstance(farm_name, list):
                farm_full_name = ' '.join(farm_name)
            else:
                farm_full_name = farm_name
                
            directory = name_dataset+"/"+results_folder+"/Ecoli/"+data_folder+"/Chi2 - " + farm_full_name
            file_name = directory+"/features_"+name_dataset+"_"+name_antibiotic+".csv"
            my_file = Path(file_name)

            try:
                my_abs_path = my_file.resolve(strict=True)
            except FileNotFoundError:
                continue
            
            df_features = pd.read_csv(file_name, header=[0])
            n_features = df_features.shape[0]
            
            
            # Preprocessing - Feature Selection
            std_scaler = MinMaxScaler()
            data = std_scaler.fit_transform(data)

            y = np.array(metadata_anti_df[reg_type])
            
            update_progress(0)
            n_feat = len(df_regression)
            for j in range(n_features):
                id_feature = np.where(features_name == df_features.iloc[j,0])[0]
                x = data[:,id_feature[0]]
                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                id_arg = np.where(args_data["Source"] == df_features.iloc[j,0])[0]
                df_regression.loc[n_feat+j,"feature"] = df_features.iloc[j,0]
                if len(id_arg)>0:
                    df_regression.loc[n_feat+j,"Antibiotic Class (CARD)"] = args_data.iloc[id_arg[0],1]
                else:
                    df_regression.loc[n_feat+j,"Antibiotic Class (CARD)"] = ""
                df_regression.loc[n_feat+j,"slope"] = slope
                df_regression.loc[n_feat+j,"intercept"] = intercept
                df_regression.loc[n_feat+j,"r2 value"] = r_value
                df_regression.loc[n_feat+j,"p-value"] = p_value
                df_regression.loc[n_feat+j,"std_err"] = std_err
                
                
                update_progress((j+1)/n_features)
                
            
        
        if len(df_regression) == 0:
            continue
        
        cols = np.where(df_regression["p-value"] < 0.05)[0]
        
        df_regression = df_regression.loc[cols,:]
        
        df_regression.drop_duplicates(subset=None, keep='first', inplace=True)
        
        directory = name_dataset+"/"+results_folder+"/Ecoli/"+data_folder

        if not os.path.exists(directory):
            os.makedirs(directory)
            
        df_regression.to_csv(directory+"/features_"+farm_full_name+"_"+reg_type+"_"+name_dataset+".csv", index=False)
