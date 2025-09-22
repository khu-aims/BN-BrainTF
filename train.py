import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import csv
from BN_BrainTF_model import BN_BrainTF
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class LoadDataset_from_numpy(Dataset):
    def __init__(self, x_data_file_1,x_con_file, y_data_file):
        super(LoadDataset_from_numpy, self).__init__()
        
        sch_idx = np.array([0,1,2,3, 50,51,52,53,54,55,56,57,58, 4,5,6,7,8,9,10,11,12,13,14,15,16, 59,60,61,62,63,64,65,66,67,68,69,
                            17,18,19,20,21,22,23,24, 70,71,72,73,74,75,76, 25,26,27, 77,78, 28,29,30,31,32,33,34, 79,80,81,82,83,
                            35,36,37,38,39,40, 84,85,86,87,88,89,90,91, 41,42,43,44,45,46,47,48,49,92,93,94,95,96,97,98,99, 
                            100,101,102,103,104,105,106,107,108,109,110,111,112,113])
                            
        X_sample = np.load(x_data_file_1)[:,:,sch_idx]
        X_pcc = np.load(x_con_file)[:,sch_idx][:,:,sch_idx]
        y_sample = np.load(y_data_file)
    
        self.len = X_sample.shape[0]
        self.x_data = torch.from_numpy(X_sample)
        self.pcc_data = torch.from_numpy(X_pcc)
        self.y_data = torch.from_numpy(y_sample).long()
        
        self.x_data = self.x_data.float()
        self.pcc_data = self.pcc_data.float()

    def __getitem__(self, index):
        return self.x_data[index], self.pcc_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def train(model, device, train_loader, epoch, criterion_cls,criterion_diff,optimizer):
    model.train()
    correct = 0
    total = 0
    losses = 0.0

    for batch_idx, data in enumerate(train_loader):
        X, con, y = data
        X = X.to(device)
        con = con.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output, diff_out = model(X, con)
        
        cls_loss = criterion_cls(output, y).to(device)

        batch_n = y.shape[0]
        community_target1 = torch.LongTensor([0]*batch_n).to(device)
        community_target2 = torch.LongTensor([1]*batch_n).to(device)
        community_target3 = torch.LongTensor([2]*batch_n).to(device)
        community_target4 = torch.LongTensor([3]*batch_n).to(device)
        community_target5 = torch.LongTensor([4]*batch_n).to(device)
        community_target6 = torch.LongTensor([5]*batch_n).to(device)
        community_target7 = torch.LongTensor([6]*batch_n).to(device)
        community_target8 = torch.LongTensor([7]*batch_n).to(device)
        community_target9 = torch.LongTensor([8]*batch_n).to(device)
  
        diff_loss = criterion_diff(diff_out[0],community_target1).to(device)
        diff_loss += criterion_diff(diff_out[1],community_target2).to(device)
        diff_loss += criterion_diff(diff_out[2],community_target3).to(device)
        diff_loss += criterion_diff(diff_out[3],community_target4).to(device)
        diff_loss += criterion_diff(diff_out[4],community_target5).to(device)
        diff_loss += criterion_diff(diff_out[5],community_target6).to(device)
        diff_loss += criterion_diff(diff_out[6],community_target7).to(device)
        diff_loss += criterion_diff(diff_out[7],community_target8).to(device)
        diff_loss += criterion_diff(diff_out[8],community_target9).to(device)
 

        loss = 0.9*cls_loss + 0.1*diff_loss
        loss.backward()
        optimizer.step()
        
        losses +=loss.item()*y.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        pred = pred.squeeze(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
        
    
    loss_all = losses / total      
    train_acc = correct/total*100.

    print(f'epoch: {epoch}, Training_loss: {loss_all}, Training_accuracy: {train_acc}')
                
    return train_acc, loss_all
            

def test(model, device, test_loader, criterion_cls):
    
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        losses=0.0

        total_pred = []
        total_y = []
        fusion_attn_list = []
        ca_attn_list = []
        

        for X_test, con_test, y_test in test_loader:
            X_test = X_test.to(device)
            con_test = con_test.to(device)
            y_test = y_test.to(device)

            output, _ = model(X_test, con_test)

            cls_loss =criterion_cls(output, y_test).to(device)

            loss = cls_loss
            losses += loss.item()*y_test.size(0)

            pred = output.argmax(dim=1, keepdim=True)
            pred=pred.squeeze(1)
            total += y_test.size(0)
            correct += (pred == y_test).sum().item()

            attn = model.fusion_tf.attn_score[pred==y_test].mean(dim=1)
            attn_np = attn.detach().cpu().numpy()
            attn2 = model.fusion_tf2.attn_score[pred==y_test].mean(dim=1)
            attn_np2 = attn2.detach().cpu().numpy()
            
            fusion_attn_block = np.stack([attn_np, attn_np2], axis=1)
            fusion_attn_list.append(fusion_attn_block)           
            
            ca_attn_block = []
            for block in range(4):
                ca_attn_local = []
                for local in range(8):
                    ca_attn = model.attnlist_list[local][block].attn_score[pred==y_test].mean(dim=1)
                    ca_attn_np = ca_attn.detach().cpu().numpy()
                    ca_attn_local.append(ca_attn_np) # [ [batch, roi,114]]
                ca_attn_block.append(np.concatenate(ca_attn_local,axis=1)) #[ [batch, 114,114]]
            ca_attn_list.append(np.stack(ca_attn_block, axis=1)) #  batch, block, 114, 114

            total_y.append(y_test.detach().cpu().numpy())
            total_pred.append(pred.detach().cpu().numpy())

        fusion_attn_list = np.concatenate(fusion_attn_list, axis=0)
        ca_attn_list = np.concatenate(ca_attn_list, axis=0)
        
        correct /= total
        losses /= total
        total_y = np.concatenate(total_y, axis=0).squeeze()
        total_pred = np.concatenate(total_pred, axis=0).squeeze()
        print(f"#####Test Accuracy: {100 * correct}%  Test Loss: {losses}")
        print("Confusion Matrix :")
        print(confusion_matrix(total_y, total_pred))
        print("Classification Report :")
        print(classification_report(total_y, total_pred, digits=5))
        f1 = f1_score(total_y, total_pred, average=None)
        
        test_acc = correct*100.
        
    return test_acc, losses, f1, [ca_attn_list,fusion_attn_list], total_pred


makedirs('./saved/model')
makedirs('./saved/predict')
makedirs('./saved/learning_curve')

sess = '1'

with open('results_BN_BrainTF_SEED_dataset_session{}.csv'.format(sess), mode='w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow(['Subjects', 'Accuracy', 'F1_neg', 'F1_neut', 'F1_pos'])

    n_subjects = 15
   

    for sub in range(1, n_subjects+1):
        print(f"Experiment {sub}/{n_subjects}")

        cuda = True
        cudnn.benchmark = True

        manual_seed = 1234
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        model_dir = 'BN_BrainTF_sub{}_sess{}'.format(sub, sess)

        lr = 0.00005
        batch_size = 32
        n_epoch = 500
        device = "cuda:0"

        model = BN_BrainTF().to(device)

        training_files = './Data/SEED/source_eeg/subject{}/session{}/window_10sec_de/train_30s1_sch_subco_seg_de.npy'.format(sub, sess)
        train_con = './Data/SEED/source_eeg/subject{}/session{}/window_10sec_pcc/train_30s1_sch_subco_pcc.npy'.format(sub, sess)
        test_files = './Data/SEED/source_eeg/subject{}/session{}/window_10sec_de/test_30s1_sch_subco_seg_de.npy'.format(sub, sess)
        test_con = './Data/SEED/source_eeg/subject{}/session{}/window_10sec_pcc/test_30s1_sch_subco_pcc.npy'.format(sub, sess)
        train_label = './Data/SEED/source_eeg/y_30s1_train.npy'
        test_label = './Data/SEED/source_eeg/y_30s1_test.npy'


        train_dataset = LoadDataset_from_numpy(training_files, train_con, train_label)
        test_dataset = LoadDataset_from_numpy(test_files, test_con, test_label)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    drop_last=False,
                                                    )
            
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    drop_last=False,
                                                    )
        
        best_acc = 0
        patience = 50

        criterion_cls = nn.NLLLoss()
        criterion_diff = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay =0.00002)

        for epoch in range(1, n_epoch + 1):
                train_a, train_l = train(model, device, train_loader, epoch, criterion_cls,criterion_diff, optimizer)
                test_a, test_l, test_f1, test_attn, test_pred = test(model, device, test_loader, criterion_cls)
                
                if best_acc < test_a:
                    patience = 50
                    best_acc = test_a
                    best_f1 = test_f1
                    print(f"######Best test Accuracy: {best_acc}%, epoch: {epoch}")
                    torch.save(model.state_dict(), './saved/model/BN_BrainTF_subject{}_session{}_model_state_dict.pt'.format(sub,sess))
                    np.save('./saved/model/BN_BrainTF_subject{}_session{}_ca_attnMAP.npy'.format(sub,sess), test_attn[0])
                    np.save('./saved/model/BN_BrainTF_subject{}_session{}_fu_attnMAP.npy'.format(sub,sess), test_attn[1])
                    np.save('./saved/predict/BN_BrainTF_subject{}_session{}_predict.npy'.format(sub,sess), test_pred)

                else:
                    patience-=1
                    
                log = {'epoch':epoch}
                log['train_acc'] = train_a
                log['train_loss'] = train_l
                log['test_acc'] = test_a
                log['test_loss'] = test_l
                log['test_f1_neg'] = test_f1[0]
                log['test_f1_neu'] = test_f1[1]
                log['test_f1_pos'] = test_f1[2]
                
                if epoch == 1:
                    learning_df = pd.DataFrame([log])
                else:
                    updated_learning_df = pd.DataFrame([log])
                    learning_df = pd.concat([learning_df,updated_learning_df], axis=0, ignore_index=True)

                if patience <0:
                    break

        print('session: ', sess, 'subject: ', sub, 'best ACC: ', best_acc, 'best F1: ', best_f1)
        writer.writerow([sub, best_acc, best_f1[0], best_f1[1], best_f1[2]])


        train_loss = learning_df['train_loss']
        test_loss = learning_df['test_loss']

        epochs = range(1, len(train_loss) + 1)

        plt.plot(epochs, train_loss, 'r', label='Training loss')     
        plt.plot(epochs, test_loss, 'b', label='Test loss') 
        plt.title('Training and Test loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./saved/learning_curve/loss_curve_{}.png'.format(model_dir))
        plt.close()

        train_acc = learning_df['train_acc']           
        test_acc = learning_df['test_acc']   

        plt.plot(epochs, train_acc, 'r', label='Training acc')
        plt.plot(epochs, test_acc, 'b', label='Test acc')
        plt.title('Training and Test accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('./saved/learning_curve/accuarcy_curve_{}.png'.format(model_dir))
        plt.close()

        learning_df.to_csv('./saved/learning_curve/learning_curve_{}.csv'.format(model_dir))


    







