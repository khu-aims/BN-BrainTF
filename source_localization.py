import scipy
import scipy.io
import os 
import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
import pickle
from tqdm import tqdm


SEED_channel = ['Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8','FT7','FC5',
                'FC3','FC1','FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8',
                'TP7','CP5','CP3','CP1','CPz','CP2','CP4','CP6','TP8','P7','P5','P3','P1','Pz','P2','P4',
                'P6','P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

sfreq = 200

path = "./Data/SEED/raw/Preprocessed_EEG/"
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith(".mat")]

print ("file_list_py: {}".format(file_list_py))
eeg_file_list = file_list_py[:-1]

emotion_label = scipy.io.loadmat('./Data/SEED/raw/Preprocessed_EEG/'+file_list_py[-1])['label']


source_eeg_path = './Data/SEED/source_eeg/'
if not os.path.exists(source_eeg_path):
    os.makedirs(source_eeg_path)
else:
    print(f'{source_eeg_path} is already exist now.')



label_names = ['Left-Thalamus-Proper',
                'Left-Caudate',
                'Left-Putamen',
                'Left-Pallidum',
                'Left-Hippocampus',
                'Left-Amygdala',
                'Left-Accumbens-area',

                'Right-Thalamus-Proper',
                'Right-Caudate',
                'Right-Putamen',
                'Right-Pallidum',
                'Right-Hippocampus',
                'Right-Amygdala',
                'Right-Accumbens-area']

X_sig_data = []
sess_num = 3

for sess in tqdm(range(sess_num)):

    for nprti in tqdm(range(sess,len(eeg_file_list),3)):
        file_name = './Data/SEED/raw/Preprocessed_EEG/%s'% eeg_file_list[nprti]
        EEG_file = scipy.io.loadmat(file_name)
        eeg_list = list(EEG_file.keys())

        for i in range(15,0,-1):
            subject_sig_data =  EEG_file[eeg_list[-i]]
            X_sig_data.append(subject_sig_data)


    total_source_sig = []

    for ins in tqdm(range(len(X_sig_data))):

        eeg_sig = X_sig_data[ins]
        eeg_std = eeg_sig.std(axis=1)
        n_channels = len(SEED_channel)

        template_1005_montage = mne.channels.make_standard_montage('standard_1005')

        # Initialize an info structure
        info = mne.create_info(
            ch_names = SEED_channel,
            ch_types = ['eeg'] * n_channels,
            sfreq = sfreq,
        )


        info.set_montage(template_1005_montage, match_case=False, match_alias=True)
        info['description'] = 'SEED'

        SEED_raw = mne.io.RawArray(eeg_sig, info) 

        # Download fsaverage files
        fs_dir = fetch_fsaverage(verbose = True)
        subjects_dir = os.path.dirname(fs_dir) # The path to the directory containing the FreeSurfer subjects reconstructions.

        # The files live in:
        subject = 'fsaverage'
        trans = 'fsaverage' # MNE has a built-in fsaverage transformation
        src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
        bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
        
        fname_model = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem.fif')

        SEED_raw.set_eeg_reference(projection = True) # common average reference
        
        labels_vol = label_names
        src = mne.setup_source_space(
            subject, spacing="ico5", add_dist=False, subjects_dir=subjects_dir)  
            
        fname_aseg = os.path.join(fs_dir, "mri",  "aseg.mgz")
        
        vol_src = mne.setup_volume_source_space(
            subject,
            mri=fname_aseg,
            pos=5.0,
            bem=fname_model,
            volume_label=labels_vol,
            subjects_dir=subjects_dir,
            add_interpolator=False, 
            verbose=True,)

        # Generate the mixed source space
        src += vol_src
        

        fwd = mne.make_forward_solution(SEED_raw.info, trans = trans, src = src,
                                        bem = bem, eeg = True, mindist=5, n_jobs = 12)

        noise_cov = mne.make_ad_hoc_cov(info, std = {'eeg': eeg_std}, verbose = None)

        inverse_operator = make_inverse_operator(
            SEED_raw.info, fwd, noise_cov, loose = 'auto', depth = 0.8
        )

        method = 'sLORETA' # could also be MNE or dSPM
        snr = 1.0 # use smaller SNR for raw data
        lambda2 = 0.05
        stc = apply_inverse_raw(SEED_raw, inverse_operator, lambda2, method, buffer_size=200, pick_ori=None)


        labels_parc = mne.read_labels_from_annot('fsaverage', parc = 'Schaefer2018_100Parcels_7Networks_order',
                                            subjects_dir= subjects_dir)

        labels_parc = labels_parc[:len(labels_parc) - 2] 

        src = inverse_operator['src']
        label_ts = mne.extract_label_time_course(
            [stc], labels=labels_parc, src=src, mode = 'mean_flip', return_generator=False)  
        
        total_source_sig.append(label_ts[0])


    

    with open(source_eeg_path+'source_eeg_session{}.pkl'.format(sess+1), 'wb') as f:
        pickle.dump(total_source_sig, f)




    
   
    



