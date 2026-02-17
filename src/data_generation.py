# -*- coding: utf-8 -*-
import utils
from models.model_loader import model_loader
from models.model_operation import gpr_operations

def gen_dft_pred_for_db():
    
    db_id_pro, db_sim_pro, db_smiles_pro, db_mw_pro = utils.load_dict('db_pro_info')
    
    model = model_loader.load(
        model_type='gg',
        split_type='all',
        clean='with_errors'
    )
    
    model_ER = model_loader.load(
        model_type='gg',
        split_type='all',
        clean='error_removed'
    )

    db_pred_pro = gpr_operations.predict_tf(model, db_sim_pro)
    db_pred_pro_ER = gpr_operations.predict_tf(model_ER, db_sim_pro)
    
    utils.save_npy(db_pred_pro,'update_db_pred_pro')
    utils.save_npy(db_pred_pro_ER, 'update_db_pred_pro_ER')

def gen_dft_pred_for_nist():
    
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    import numpy as np
    
    from data import data_loader, data_cleaner
    from data.data_processing import DatasetContainer
    from configs import ERROR_IDS, D_SUB_IDS # , CL_SUB_IDS
    
    def k_fold_pred(dataset: DatasetContainer, model_type='gg', clean_type='with_errors'):
   
        kfold_sim_list = []
        kfold_exp_list = []
        kfold_id_list = []
        kfold_mw_list = []
        kfold_smiles_list = []
        kfold_pred_list = []
        
        id_arr = np.array(dataset.ids)
        smiles_arr = np.array(dataset.smiles)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=23)
        i = 0
        for train_index, test_index in kf.split(dataset.features): 
            i += 1
            
            # train_x, x_val = dataset.features[train_index], dataset.features[test_index]
            # train_y, y_val = dataset.targets[train_index], dataset.targets[test_index]
            # train_id, id_val = id_arr[train_index], id_arr[test_index]
            # train_mw, mw_val = dataset.mws[train_index], dataset.mws[test_index]
            # train_smiles, smiles_val = smiles_arr[train_index], smiles_arr[test_index]
            
            x_val = dataset.features[test_index]
            y_val = dataset.targets[test_index]
            id_val = id_arr[test_index]
            mw_val = dataset.mws[test_index]
            smiles_val = smiles_arr[test_index]

            model = model_loader.load(
                model_type=model_type,
                split_type='kfold',
                fold_num=i,
                clean=clean_type
            )
            
            y_pred = gpr_operations.predict_tf(model, x_val)
            
            kfold_id_list.extend(id_val)
            kfold_pred_list.extend(y_pred)
            kfold_sim_list.extend(x_val)
            kfold_exp_list.extend(y_val)
            kfold_mw_list.extend(mw_val)
            kfold_smiles_list.extend(smiles_val)
        
        kfold_dataset = DatasetContainer(
            features=kfold_sim_list, 
            targets=kfold_exp_list, 
            ids=kfold_id_list, 
            mws=kfold_mw_list, 
            smiles=kfold_smiles_list)
        
        return kfold_dataset, kfold_pred_list
    
    def test_pred(dataset: DatasetContainer, model_type='gg', clean_type='with_errors'):
        
        model = model_loader.load(
            model_type=model_type,
            split_type='all',
            clean=clean_type
        )
        
        return gpr_operations.predict_tf(model, dataset.features)

    nist_data = data_loader.load_nist_data() 

    # split
    (x_train, x_test,
     y_train, y_test,
     ids_train, ids_test,
     mw_train, mw_test,
     smiles_train, smiles_test) = train_test_split(
        nist_data['sim_list'],
        nist_data['exp_list'],
        nist_data['id_list'],
        nist_data['mw_int_list'],
        nist_data['smiles_list'],
        test_size=0.5,
        random_state=23
    )
    
    # train data for k_fold training
    train_data = DatasetContainer(
        features=x_train,
        targets=y_train,
        ids=ids_train,
        mws=mw_train,
        smiles=smiles_train
    )
    
    kfold_dataset, kfold_pred_list = k_fold_pred(train_data, model_type='gg', clean_type='with_errors')
    
    # test data
    test_data = DatasetContainer(
        features=x_test,
        targets=y_test,
        ids=ids_test,
        mws=mw_test,
        smiles=smiles_test
    )
    
    test_pred_list = test_pred(test_data, model_type='gg', clean_type='with_errors')
    
    cleaned_train_data = data_cleaner.remove_error_data(
        train_data,
        nist_data['id_fgs_map'], ERROR_IDS, D_SUB_IDS
    )
    
    kfold_dataset_ER, kfold_pred_list_ER = k_fold_pred(cleaned_train_data, model_type='gg', clean_type='error_removed')

    cleaned_test_data = data_cleaner.remove_error_data(
        test_data,
        nist_data['id_fgs_map'], ERROR_IDS, D_SUB_IDS
    )
    
    test_pred_list_ER = test_pred(cleaned_test_data, model_type='gg', clean_type='error_removed')
    
    nist_dataset = DatasetContainer(
        features = np.concatenate((kfold_dataset.features, test_data.features)), 
        targets = np.concatenate((kfold_dataset.targets, test_data.targets)), 
        ids = np.concatenate((kfold_dataset.ids, test_data.ids)), 
        mws = np.concatenate((kfold_dataset.mws, test_data.mws)), 
        smiles = np.concatenate((kfold_dataset.smiles, test_data.smiles))
    )
    
    nist_dataset_ER = DatasetContainer(
        features = np.concatenate((kfold_dataset_ER.features, cleaned_test_data.features)), 
        targets = np.concatenate((kfold_dataset_ER.targets, cleaned_test_data.targets)), 
        ids = np.concatenate((kfold_dataset_ER.ids, cleaned_test_data.ids)), 
        mws = np.concatenate((kfold_dataset_ER.mws, cleaned_test_data.mws)), 
        smiles = np.concatenate((kfold_dataset_ER.smiles, cleaned_test_data.smiles))
    )
    
    nist_pred_list = np.concatenate((kfold_pred_list, test_pred_list))
    nist_pred_list_ER = np.concatenate((kfold_pred_list_ER, test_pred_list_ER))
    
    utils.save_dict(nist_dataset, 'update_nist_dataset')
    utils.save_dict(nist_dataset_ER, 'update_nist_dataset_ER')
    utils.save_npy(nist_pred_list, 'update_nist_pred_list')
    utils.save_npy(nist_pred_list_ER, 'update_nist_pred_list_ER')

