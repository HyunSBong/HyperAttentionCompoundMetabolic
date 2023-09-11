import random
import os
import csv
import sys

from dataset import CustomDataSet, collate_fn_new, collate_fn_new_dacon

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
from pytorchtools import EarlyStopping  
import timeit
from tensorboardX import SummaryWriter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from model import AttentionDTI, RMSELoss, RMSLELoss

def show_result(DATASET,lable,Accuracy_List,Precision_List,Recall_List,AUC_List,AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print("The {} model's results:".format(lable))
    with open("./{}/results.txt".format(DATASET), 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))

def load_tensor(file_name, dtype):
    # return [dtype(d).to(hp.device) for d in np.load(file_name + '.npy', allow_pickle=True)]
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]

def test_precess(model,pbar,LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, labels = data
            compounds = compounds.cuda()
            proteins = proteins.cuda()
            labels = labels.cuda()

            predicted_scores = model(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)  
    return Y, P, test_loss, Accuracy, Precision, Reacll, AUC, PRC

def test_model(dataset_load,save_path,DATASET, LOSS, dataset = "Train",lable = "best",save = True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_precess(model,test_pbar, LOSS)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET,dataset,lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(lable, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results,Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test


def get_kfold_data(i, datasets, k=5):
    
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def MAE(true, pred):
    return np.mean(np.abs(true-pred))

import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if __name__ == "__main__":
    """select seed"""
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    """init hyperparameters"""
    hp = hyperparameter()

    """Load preprocessed data."""
    weight_CE = None
    dir_input = './data/compound/'
    
    # mol = pd.read_csv(dir_input + 'train_mol2vec.csv')
    # labels = pd.read_csv(dir_input + 'train.csv')[['MLM','HLM']]
    
    with open(dir_input + 'train.csv', "r") as f:
        lines = f.read().strip().split('\n')
    df = lines[1:]
    # print(df[:2])
    
    # df = pd.read_csv(dir_input + 'train.csv')
    # df_new = df.copy()
    # test = pd.read_csv(dir_input + 'test.csv')
    # df = df.apply(lambda row: ','.join(map(str, row)), axis=1).tolist()
    
    with open(dir_input + 'test.csv', "r") as f:
        lines = f.read().strip().split('\n')
    dacon_test_df = lines[1:]
    
    print("load finished")

    ## Scaler
    if hp.target == 'HLM':
        target = 3
    elif hp.target == 'MLM':
        target = 2
        
    dataset = CustomDataSet(df)
    dacon_test = CustomDataSet(dacon_test_df)
    train_size = int(len(dataset) * 0.8)
    valid_size = int(len(dataset) * 0.2)
    test_size = int(len(dataset) * 0.05)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size+1])
    valid_dataset, test_dataset = torch.utils.data.random_split(valid_dataset, [valid_size-test_size+1, test_size])
    
    train_dataset = list(train_dataset)
    valid_dataset = list(valid_dataset)
    test_dataset = list(test_dataset)
    dacon_test_dataset = list(dacon_test)
    print(train_dataset[:3])
    
    train_column2_values = [float(data.split(',')[target]) for data in train_dataset]
    valid_column2_values = [float(data.split(',')[target]) for data in valid_dataset]
    test_column2_values = [float(data.split(',')[target]) for data in test_dataset]
    dacon_test_dataset_vec = [float(data.split(',')[target]) for data in dacon_test_dataset]
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(np.array(train_column2_values).reshape(-1, 1))
    valid_scaled = scaler.transform(np.array(valid_column2_values).reshape(-1, 1))
    test_scaled = scaler.transform(np.array(test_column2_values).reshape(-1, 1))
    dacon_test_scaled = scaler.transform(np.array(dacon_test_dataset_vec).reshape(-1, 1))
    
    for i in range(len(train_dataset)):
        parts = train_dataset[i].split(',')
        scaled_value = train_scaled[i]
        parts[target] = str(scaled_value)[1:-2]
        train_dataset[i] = ','.join(parts)
    
    for i in range(len(valid_dataset)):
        parts = valid_dataset[i].split(',')
        scaled_value = valid_scaled[i]
        parts[target] = str(scaled_value)[1:-2]
        valid_dataset[i] = ','.join(parts)
        
    for i in range(len(test_dataset)):
        parts = test_dataset[i].split(',')
        scaled_value = test_scaled[i]
        parts[target] = str(scaled_value)[1:-2]
        test_dataset[i] = ','.join(parts)
        
    for i in range(len(dacon_test_dataset)):
        parts = dacon_test_dataset[i].split(',')
        scaled_value = dacon_test_scaled[i]
        parts[target] = str(scaled_value)[1:-2]
        dacon_test_dataset[i] = ','.join(parts)
        
    train_dataset = CustomDataSet(train_dataset)
    valid_dataset = CustomDataSet(valid_dataset)
    test_dataset = CustomDataSet(test_dataset)
    dacon_test_dataset = CustomDataSet(dacon_test_dataset)
    
    
    K_Fold = 1 # 5

    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)
        
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=hp.Batch_size, 
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=collate_fn_new)
        
        val_loader = DataLoader(dataset=valid_dataset,
                                batch_size=hp.Batch_size, 
                                shuffle=False,
                                num_workers=0,
                                collate_fn=collate_fn_new)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=hp.Batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=collate_fn_new)
        
        dacon_test_loader = DataLoader(dataset=dacon_test_dataset,
                                 batch_size=hp.Batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=collate_fn_new_dacon)

        """ create model"""
        model = AttentionDTI(hp).cuda()
        """weight initialize"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        """load trained model"""
        # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))

        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)
        # optimizer = optim.Adam(model.parameters(), lr = hp.Learning_rate)
        
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False, step_size_up=len(train_loader) // hp.Batch_size)

        # Loss = nn.MSELoss(size_average=False, reduction="sum")
        # Loss = RMSLELoss()
        Loss = RMSELoss()
        # print(model)
        
        save_path = "./" + 'dacon' + "/{}".format(i_fold)
        note = ''
        writer = SummaryWriter(log_dir=save_path, comment=note)

        """Output files."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path+'The_results_of_whole_dataset.txt'

        with open(file_results, 'w') as f:
            hp_attr = '\n'.join(['%s:%s' % item for item in hp.__dict__.items()])
            f.write(hp_attr + '\n')

        
        early_stopping = EarlyStopping(savepath = save_path,patience=hp.Patience, verbose=True, delta=0)
        # print("Before train,test the model:")
        # _,_,_,_,_,_ = test_model(test_dataset_load, save_path, DATASET, Loss, dataset="Test",lable="untrain",save=False)
        """Start training."""
        print('Training...')
        start = timeit.default_timer()

        for epoch in range(1, hp.Epoch + 1):
            trian_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(train_loader)),
                total=len(train_loader))
            """train"""
            train_losses_in_epoch = []
            model.train()
            for batch_idx, train_data in trian_pbar:
                '''data preparation '''
                trian_compounds, trian_features, trian_labels = train_data
                trian_compounds = trian_compounds.cuda()
                trian_features = trian_features.cuda()
                trian_labels = trian_labels.cuda()
               
                optimizer.zero_grad()
               
                pred = model(trian_compounds, trian_features)
                
                train_labels = list(trian_labels.to('cpu').data.numpy())
                train_preds = pred.to('cpu').data.numpy()
                flatten_np = lambda pred: [item for sublist in pred for item in sublist]
                flatten_train_preds = flatten_np(train_preds)

                # print(train_labels)
                # print(flatten_train_preds)
                # print(xd)
                
                train_loss = Loss(pred, trian_labels)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                optimizer.step()
                scheduler.step()
                
            train_loss_a_epoch = np.average(train_losses_in_epoch)
            print(f'Epoch[{epoch}/{hp.Epoch}] train mse loss per epoch =====> {train_loss_a_epoch}')
            # writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)
            # avg_train_losses.append(train_loss_a_epoch)

            """valid"""
            valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(val_loader)),
                total=len(val_loader))
            valid_losses_in_epoch = []
            model.eval()
            pred_ls, true_ls = [], []
            with torch.no_grad():
                for batch_idx, valid_data in valid_pbar:
                    '''data preparation '''
                    valid_compounds, valid_features, valid_labels = valid_data
                    valid_compounds = valid_compounds.cuda()
                    valid_features = valid_features.cuda()
                    valid_labels = valid_labels.cuda()

                    pred = model(valid_compounds, valid_features)
                    valid_loss = Loss(pred, valid_labels)
                    
                    valid_labels = list(valid_labels.to('cpu').data.numpy())
                    valid_preds = list(pred.to('cpu').data.numpy())
                    
                    inverse_valid_labels = scaler.inverse_transform(np.array(valid_labels).reshape(-1, 1))
                    inverse_valid_preds = scaler.inverse_transform(np.array(valid_preds).reshape(-1, 1))
                    
                    flatten_np = lambda pred: [item for sublist in pred for item in sublist]
                    flatten_inverse_valid_labels = flatten_np(inverse_valid_labels)
                    flatten_inverse_valid_preds = flatten_np(inverse_valid_preds)

                    valid_losses_in_epoch.append(valid_loss.item())
                    pred_ls += flatten_inverse_valid_preds
                    true_ls += flatten_inverse_valid_labels
                    # print(flatten_inverse_valid_labels)
                    # print(flatten_inverse_valid_preds)
                    # print(dd)
                    
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)  
            print(f'Epoch[{epoch}/{hp.Epoch}] valid mse loss per epoch =====> {valid_loss_a_epoch}')
            # avg_valid_loss.append(valid_loss)


            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss_a_epoch, model, epoch)
            
            # print(pred_ls)
            # print(true_ls)
            mae_score = mean_absolute_error(true_ls, pred_ls)
            mse_score = mean_squared_error(true_ls, pred_ls)
            rmse_score = np.sqrt(mse_score)
            rmsle_score = mean_squared_log_error(true_ls, pred_ls)
            print('MAE SCORE : ', mae_score)
            print('MSE SCORE : ', mse_score)
            print('RMSE SCORE : ', rmse_score)
            print('RMSLE SCORE : ', rmsle_score)
            print('R2 SCORE : ', r2_score(true_ls, pred_ls))
            
            output_file = f"./valid_predictions_{hp.target}_RMSE_{rmse_score.round(3)}.csv"
            with open(output_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"True_Label_{hp.target}", f"Predicted_Label_{hp.target}"])
                for true_label, predicted_label in zip(true_ls, pred_ls):
                    writer.writerow([true_label, predicted_label])

            print(f"Valid predictions saved to {output_file}")
        
        """test"""
        test_pbar = tqdm(
            enumerate(
                BackgroundGenerator(test_loader)), total=len(test_loader))
        model.eval()
        test_losses_in_epoch = []
        pred_ls, true_ls = [], []
        with torch.no_grad():
            for batch_idx, test_data in test_pbar:
                test_compounds, test_features, test_labels = test_data
                test_compounds = test_compounds.cuda()
                test_features = test_features.cuda()
                test_labels = test_labels.cuda()

                pred = model(test_compounds, test_features)
                test_loss = Loss(pred, test_labels)

                test_labels = list(test_labels.to('cpu').data.numpy())
                test_preds = list(pred.to('cpu').data.numpy())

                inverse_test_labels = scaler.inverse_transform(np.array(test_labels).reshape(-1, 1))
                inverse_test_preds = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1))

                flatten_np = lambda pred: [item for sublist in pred for item in sublist]
                flatten_inverse_test_labels = flatten_np(inverse_test_labels)
                flatten_inverse_test_preds = flatten_np(inverse_test_preds)

                test_losses_in_epoch.append(test_loss.item())
                pred_ls += flatten_inverse_test_preds
                true_ls += flatten_inverse_test_labels
                # print(flatten_inverse_test_labels)
                # print(flatten_inverse_test_preds)
                # print(dd)
                    
            # print(pred_ls)
            # print(true_ls)
            mae_score = mean_absolute_error(true_ls, pred_ls)
            mse_score = mean_squared_error(true_ls, pred_ls)
            rmse_score = np.sqrt(mse_score)
            rmsle_score = mean_squared_log_error(true_ls, pred_ls)
            print('MAE SCORE : ', mae_score)
            print('MSE SCORE : ', mse_score)
            print('RMSE SCORE : ', rmse_score)
            print('RMSLE SCORE : ', rmsle_score)
            print('R2 SCORE : ', r2_score(true_ls, pred_ls))
            
            output_file = f"./test_predictions_{hp.target}_RMSE_{rmse_score.round(3)}.csv"
            with open(output_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"True_Label_{hp.target}", f"Predicted_Label_{hp.target}"])
                for true_label, predicted_label in zip(true_ls, pred_ls):
                    writer.writerow([true_label, predicted_label])

            print(f"Test predictions saved to {output_file}")
            
            
        """dacon test"""
        test_pbar = tqdm(
            enumerate(
                BackgroundGenerator(dacon_test_loader)), total=len(dacon_test_loader))
        model.eval()
        pred_ls = []
        with torch.no_grad():
            for batch_idx, test_data in test_pbar:
                test_compounds, test_features = test_data
                test_compounds = test_compounds.cuda()
                test_features = test_features.cuda()

                pred = model(test_compounds, test_features)
                
                test_preds = list(pred.to('cpu').data.numpy())
                inverse_test_preds = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1))

                flatten_np = lambda pred: [item for sublist in pred for item in sublist]
                flatten_inverse_test_preds = flatten_np(inverse_test_preds)

                pred_ls += flatten_inverse_test_preds
                # print(flatten_inverse_test_preds)
                # print(dd)
                    
            # print(pred_ls)
            # print(true_ls)
            
            output_file = f"./dacon_test_predictions_{hp.target}_RMSE_{rmse_score.round(3)}.csv"
            with open(output_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"Predicted_Label_{hp.target}"])
                for predicted_label in (pred_ls):
                    writer.writerow([predicted_label])

            print(f"Test predictions saved to {output_file}")

#         trainset_test_stable_results,_,_,_,_,_ = test_model(train_dataset_load, save_path, DATASET, Loss, dataset="Train", lable="stable")
#         validset_test_stable_results,_,_,_,_,_ = test_model(valid_dataset_load, save_path, DATASET, Loss, dataset="Valid", lable="stable")
#         testset_test_stable_results,Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
#             test_model(test_dataset_load, save_path, DATASET, Loss, dataset="Test", lable="stable")
#         AUC_List_stable.append(AUC_test)
#         Accuracy_List_stable.append(Accuracy_test)
#         AUPR_List_stable.append(PRC_test)
#         Recall_List_stable.append(Recall_test)
#         Precision_List_stable.append(Precision_test)
#         with open(save_path + "The_results_of_whole_dataset.txt", 'a') as f:
#             f.write("Test the stable model" + '\n')
#             f.write(trainset_test_stable_results + '\n')
#             f.write(validset_test_stable_results + '\n')
#             f.write(testset_test_stable_results + '\n')

#     show_result(DATASET, "stable",
#                 Accuracy_List_stable, Precision_List_stable, Recall_List_stable,
#                 AUC_List_stable, AUPR_List_stable)



