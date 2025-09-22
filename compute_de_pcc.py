import numpy as np
import pandas as pd
import scipy.io
import os 
import pickle
from tqdm import tqdm
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    return b,a 

# 4-45 Hz bandpass
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

path = "./Data/SEED/raw/Preprocessed_EEG/"
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith(".mat")]
emotion_label = scipy.io.loadmat('./Data/SEED/raw/Preprocessed_EEG/'+file_list_py[-1])['label']


print('Compute differential entropy...')
sess_num = 3
for sess in tqdm(range(sess_num)):
    with open('./Data/SEED/source_eeg/source_eeg_session{}.pkl'.format(sess+1), 'rb') as f:
        total_source_sig = pickle.load(f)

    channel = len(total_source_sig[0])

    total_delta_list_p = []
    total_theta_list_p = []
    total_alpha_list_p = []
    total_beta_list_p = []
    total_gamma_list_p = []

    total_de = []

    for i in tqdm(range(len(total_source_sig))):
        
        delta_list_p = []
        theta_list_p = []
        alpha_list_p = []
        beta_list_p = []
        gamma_list_p = []
        
        delta_list_seg = []
        theta_list_seg = []
        alpha_list_seg = []
        beta_list_seg = []
        gamma_list_seg = []
        
        delta_psd_seg = []
        theta_psd_seg = []
        alpha_psd_seg = []
        beta_psd_seg = []
        gamma_psd_seg = []

        # decomposition for five frequency band
        for j in range(channel): 
            delta_list_p.append(butter_bandpass_filter(data=total_source_sig[i][j], lowcut=1, highcut=3, fs=200, order=5))
            theta_list_p.append(butter_bandpass_filter(data=total_source_sig[i][j], lowcut=4, highcut=7, fs=200, order=5))
            alpha_list_p.append(butter_bandpass_filter(data=total_source_sig[i][j], lowcut=8, highcut=13, fs=200, order=5))
            beta_list_p.append(butter_bandpass_filter(data=total_source_sig[i][j], lowcut=14, highcut=30, fs=200, order=5))
            gamma_list_p.append(butter_bandpass_filter(data=total_source_sig[i][j], lowcut=31, highcut=50, fs=200, order=5))
            
        # compute differential entropy every 1-second  
        for seg in range(0, len(total_source_sig[i][0])-200+1,200):
            delta_list_seg.append(1/2*np.log(2*np.pi*np.exp(1)*np.var(np.array(delta_list_p)[:,seg:seg+200], axis=-1)))
            theta_list_seg.append(1/2*np.log(2*np.pi*np.exp(1)*np.var(np.array(theta_list_p)[:,seg:seg+200], axis=-1)))
            alpha_list_seg.append(1/2*np.log(2*np.pi*np.exp(1)*np.var(np.array(alpha_list_p)[:,seg:seg+200], axis=-1)))
            beta_list_seg.append(1/2*np.log(2*np.pi*np.exp(1)*np.var(np.array(beta_list_p)[:,seg:seg+200], axis=-1)))
            gamma_list_seg.append(1/2*np.log(2*np.pi*np.exp(1)*np.var(np.array(gamma_list_p)[:,seg:seg+200], axis=-1)))
            
        delta_smooth_list = []
        theta_smooth_list = []
        alpha_smooth_list = []
        beta_smooth_list = []
        gamma_smooth_list = []
        
        # moving average filter for smoothing
        for ch in range(channel):
            N = 20
            temp_delta = np.array(delta_list_seg)[:,ch] # seg, channel
            delta_padded = np.pad(temp_delta, (N//2, N-1-N//2), mode='edge')
            delta_smooth = np.convolve(delta_padded, np.ones((N,))/N, mode='valid')
            
            temp_theta = np.array(theta_list_seg)[:,ch] # seg, channel
            theta_padded = np.pad(temp_theta, (N//2, N-1-N//2), mode='edge')
            theta_smooth = np.convolve(theta_padded, np.ones((N,))/N, mode='valid')
            
            temp_alpha = np.array(alpha_list_seg)[:,ch] # seg, channel
            alpha_padded = np.pad(temp_alpha, (N//2, N-1-N//2), mode='edge')
            alpha_smooth = np.convolve(alpha_padded, np.ones((N,))/N, mode='valid')
            
            temp_beta = np.array(beta_list_seg)[:,ch] # seg, channel
            beta_padded = np.pad(temp_beta, (N//2, N-1-N//2), mode='edge')
            beta_smooth = np.convolve(beta_padded, np.ones((N,))/N, mode='valid')
            
            temp_gamma = np.array(gamma_list_seg)[:,ch] # seg, channel
            gamma_padded = np.pad(temp_gamma, (N//2, N-1-N//2), mode='edge')
            gamma_smooth = np.convolve(gamma_padded, np.ones((N,))/N, mode='valid')
            
            delta_smooth_list.append(delta_smooth)
            theta_smooth_list.append(theta_smooth)
            alpha_smooth_list.append(alpha_smooth)
            beta_smooth_list.append(beta_smooth)
            gamma_smooth_list.append(gamma_smooth)
            
            
        
        total_de.append(np.stack((np.array(delta_smooth_list).swapaxes(0,1),
                                np.array(theta_smooth_list).swapaxes(0,1), 
                                np.array(alpha_smooth_list).swapaxes(0,1), 
                                np.array(beta_smooth_list).swapaxes(0,1), 
                                np.array(gamma_smooth_list).swapaxes(0,1)), axis=-1))

            

    sub=0 
    emotion_train_count_list = []
    emotion_test_count_list = []
    for sam in range(0,len(total_de)-15+1,15):
        

        train_de = total_de[sam:sam+9]  #0:9,15:24, 30:39
        test_de = total_de[sam+9:sam+15]   #9:15, 24:30, 39:45
        print('Train')
        train_seg_de = []
        for i in range(len(train_de)):
            count=0
            for j in range(0,len(train_de[i])-30+1,1):
                train_seg_de.append(train_de[i][j:j+30])
                count+=1

            if len(emotion_train_count_list) < 9:
                emotion_train_count_list.append(count)
            

        print('Test')
        test_seg_de = []
        for i in range(len(test_de)):
            count = 0
            for j in range(0,len(test_de[i])-30+1,1):
                test_seg_de.append(test_de[i][j:j+30])
                count+=1
            if len(emotion_test_count_list) < 6:
                emotion_test_count_list.append(count)

        train_seg_de = np.array(train_seg_de)
        test_seg_de = np.array(test_seg_de)
        
        if sess+1 == 1 and sub==0:
            # 0: negative, 1: neural, 2: positive, 
            emotion_label_trans = np.where(emotion_label == 0, 1, np.where(emotion_label == -1, 0, 2))

            y_train = np.repeat(emotion_label_trans.squeeze()[:9], emotion_train_count_list)
            y_test = np.repeat(emotion_label_trans.squeeze()[9:], emotion_test_count_list)

            np.save('./Data/SEED/source_eeg/y_30s1_train.npy', y_train)
            np.save('./Data/SEED/source_eeg/y_30s1_test.npy', y_test)
        

        sub +=1

        de_path = './Data/SEED/source_eeg/subject{}/session{}/window_10sec_de/'.format(sub, sess+1)
        if not os.path.exists(de_path):
            os.makedirs(de_path)
        
        np.save(de_path+'train_30s1_sch_subco_seg_de.npy'.format(sub, sess+1), train_seg_de)
        np.save(de_path+'test_30s1_sch_subco_seg_de.npy'.format(sub, sess+1), test_seg_de)

    


    print('Compute Pearson correlation coefficients...')
    sub=0 
    for sam in tqdm(range(0,len(total_source_sig)-15+1,15)):
        
        train_sig = total_source_sig[sam:sam+9]  
        test_sig = total_source_sig[sam+9:sam+15] 

        train_seg_sig = []
        for i in range(len(train_sig)):
            for j in range(0,len(train_sig[i][0])-6000+1,200):
                train_seg_sig.append(train_sig[i][:,j:j+6000])


        test_seg_sig = []
        for i in range(len(test_sig)):
            for j in range(0,len(test_sig[i][0])-6000+1,200):
                test_seg_sig.append(test_sig[i][:,j:j+6000])

        train_seg_sig = np.array(train_seg_sig)
        test_seg_sig = np.array(test_seg_sig)
        
        train_pcc_list = []
        for i in range(len(train_seg_sig)):
            train_pos_df = pd.DataFrame(train_seg_sig[i])
            corr_pos = train_pos_df.T.corr(method='pearson')
            train_pcc_list.append(corr_pos.values)
            
        test_pcc_list = []
        for i in range(len(test_seg_sig)):
            train_pos_df = pd.DataFrame(test_seg_sig[i])
            corr_pos = train_pos_df.T.corr(method='pearson')
            test_pcc_list.append(corr_pos.values)   
    
        sub +=1
        
        pcc_path = './Data/SEED/source_eeg/subject{}/session{}/window_10sec_pcc/'.format(sub, sess+1)
        if not os.path.exists(pcc_path):
            os.makedirs(pcc_path)

        np.save(pcc_path+'train_30s1_sch_subco_pcc.npy'.format(sub, sess+1), np.array(train_pcc_list))
        np.save(pcc_path+'test_30s1_sch_subco_pcc.npy'.format(sub, sess+1), np.array(test_pcc_list))



    
   
    




    



