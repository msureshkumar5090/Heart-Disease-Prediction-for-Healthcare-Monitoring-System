import numpy as np
import pandas as pd
import os
import wfdb
from keras.src.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from numpy import matlib
import biosppy.signals.ecg as ecg
from EVO import EVO
from GOA import GOA
from GSOA import GSOA
from Global_Vars import Global_Vars
from Model_Autoencoder import Model_AutoEncoder
from Model_Capsnet import Model_Capsnet
from Model_HDRLSTM_CapsNet import Model_HDRLSTM_CapsNet
from Model_LSTM import Model_LSTM
from Model_TCN import Model_TCN
from PROPOSED import PROPOSED
from RSA import RSA
from Spectral_Features import density, rms, zcr
from Spectral_Flux import spectralFlux
from THDN import THDN
import librosa
from scipy.signal import find_peaks
from objfun_cls import objfun
from plot_results import *

no_of_Dataset = 4


def spectral_centroid(x, samplerate=44100):
    magnitudes = np.abs(np.fft.rfft(x))  # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0 / samplerate)[:length + 1])  # positive frequencies
    return np.sum(magnitudes * freqs[:len(magnitudes)]) / np.sum(magnitudes)  # return weighted mean


def extract_Wave_features(ecg_signal, sampling_rate=44400):
    qrs_indices = ecg.hamilton_segmenter(signal=ecg_signal, sampling_rate=sampling_rate)[0]
    qrs_features = ecg.extract_heartbeats(signal=ecg_signal, rpeaks=qrs_indices, sampling_rate=sampling_rate)
    p_wave_features = qrs_features['templates'][:, :int(0.1 * sampling_rate)]
    t_wave_features = qrs_features['templates'][:, int(0.4 * sampling_rate):int(0.6 * sampling_rate)]
    u_wave_features = qrs_features['templates'][:, int(0.5 * sampling_rate):]
    waves = [p_wave_features[0], qrs_features[0][0], t_wave_features[0], u_wave_features[0]]
    shortest_list = min(waves, key=len)
    feature = [p_wave_features[0][:len(shortest_list)], qrs_features[0][0][:len(shortest_list)],
               t_wave_features[0][:len(shortest_list)], u_wave_features[0][:len(shortest_list)]]
    Feat = np.ravel(feature)
    return Feat


def convert_to_numeric(df, columns):
    for col in columns:
        unique_values = df[col].unique()
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        df[col] = df[col].map(value_map)
    return df


# Read the dataset 1
an = 0
if an == 1:
    Datasets = './Datasets/Dataset_1/'
    normal_file = Datasets + 'ptbdb_normal.csv'
    normal_df = pd.read_csv(normal_file)
    normal_df = np.asarray(normal_df)
    normal_tar = np.zeros(len(normal_df))
    normal_tar = normal_tar.astype('int')
    abnormal_file = Datasets + 'ptbdb_abnormal.csv'
    abnormal_df = pd.read_csv(abnormal_file)
    abnormal_df = np.asarray(abnormal_df)
    abnormal_tar = np.ones(len(abnormal_df))
    abnormal_tar = abnormal_tar.astype('int')
    Datas = np.concatenate((normal_df, abnormal_df), axis=0)
    Targets = np.concatenate((normal_tar, abnormal_tar), axis=0)
    Targets = Targets.reshape(-1, 1)
    np.save('Data_1.npy', Datas)
    np.save('Target_1.npy', Targets)


# Read the dataset 2
an = 0
if an == 1:
    Datasets = './Datasets/Dataset_2/Pre-inflation, Inflation, Post-inflation ECG Recordings/'
    dataset_path = os.listdir(Datasets)
    Data = []
    Target = []
    for i in range(len(dataset_path)):
        class_path = Datasets + dataset_path[i]
        class_dir = os.listdir(class_path)
        for j in range(len(class_dir)):
            file = class_path + '/' + class_dir[j]
            name = file.split('.ekg')[1]
            record = wfdb.rdrecord(name)
            signal = record.p_signal
            FS = record.fs
            tar = dataset_path[i]
            Data.append(signal)
            Target.append(tar)
    Datas = np.asarray(Data)
    Target = np.asarray(Target)
    label_encoder = LabelEncoder()
    Tar_encoded = label_encoder.fit_transform(Target)
    Targets = to_categorical(Tar_encoded).astype('int')
    np.save('Data_2.npy', Datas)
    np.save('Target_2.npy', Targets)


# Read the dataset 3
an = 0
if an == 1:
    Datasets = './Datasets/Dataset_3/ECGCvdata.csv'
    df = pd.read_csv(Datasets)
    df.drop(['RECORD', 'ECG_signal'], inplace=True, axis=1)
    Datas = np.asarray(df)
    Tar = pd.read_csv(Datasets, usecols=['ECG_signal'])
    Tar = np.asarray(Tar)
    label_encoder = LabelEncoder()
    Tar_encoded = label_encoder.fit_transform(Tar)
    Targets = to_categorical(Tar_encoded).astype('int')
    np.save('Data_3.npy', Datas)
    np.save('Target_3.npy', Targets)


# Read the dataset 4
an = 0
if an == 1:
    Datasets = './Datasets/Dataset_4/ecg.csv'
    df = pd.read_csv(Datasets)
    df = np.asarray(df)
    Datas = df[:, :-1]
    Targets = ((df[:, -1]).astype('int')).reshape(-1, 1)
    np.save('Data_4.npy', Datas)
    np.save('Target_4.npy', Targets)


# Read the feature set 3 for Datasets 1
an = 0
if an == 1:
    Datasets = './Datasets/Dataset_1/Feature/heart.csv'
    df = pd.read_csv(Datasets)
    df.drop(['target'], inplace=True, axis=1)
    Datas = np.asarray(df)
    np.save('Feature_3_1.npy', Datas)


# Read the feature set 3 for Datasets 2
an = 0
if an == 1:
    # pip install ucimlrepo
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    print(heart_disease.metadata)
    print(heart_disease.variables)
    Datas = np.asarray(X)
    Targets = np.asarray(y)
    np.save('Feature_3_2.npy', Datas)


# Read the feature set 3 for Datasets 3
an = 0
if an == 1:
    Datasets = './Datasets/Dataset_3/Feature/heart_disease_uci.csv'
    df = pd.read_csv(Datasets)
    df.drop(['num'], inplace=True, axis=1)
    columns_to_convert = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    Datas = convert_to_numeric(df, columns_to_convert)
    Datas = np.asarray(Datas)
    np.save('Feature_3_3.npy', Datas)


# Read the feature set 3 for Datasets 4
an = 0
if an == 1:
    file_path = './Datasets/Dataset_4/Feature/heart.csv'
    df = pd.read_csv(file_path)
    df.drop(['HeartDisease'], inplace=True, axis=1)
    columns_to_convert = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    Datas = convert_to_numeric(df, columns_to_convert)
    Datas = np.asarray(Datas).astype('int')
    Targets = pd.read_csv(file_path, usecols=['HeartDisease'])
    Targets = np.asarray(Targets)
    np.save('Feature_3_4.npy', Datas)


# Feature Extraction
# 1 Wave feature
an = 0
if an == 1:
    for n in range(no_of_Dataset):
        EEG_Signal = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)  # load the EEG_Data
        EEG_feat = []
        fs = [80, 100, 100, 100]
        for i in range(len(EEG_Signal)):
            print(i, len(EEG_Signal))
            EEG_Wave_features = extract_Wave_features((EEG_Signal[i]).astype('float'), fs[n])
            EEG_feat.append(EEG_Wave_features)
        np.save('Feature_1_' + str(n + 1) + '.npy', EEG_feat)  # Save the Feature 2


# 2 Spectral Feature Extraction from Audio Data
an = 0
if an == 1:
    for n in range(no_of_Dataset):
        Audios = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        spectral = []
        for i in range(len(Audios)):
            print(i, len(Audios))
            Sample_rate = 1000
            Audio = Audios[i]
            cetroid = spectral_centroid(Audio)
            Density = density(Audio)
            mfccs = librosa.feature.mfcc(y=Audio, sr=1000, n_mfcc=1)
            Flux = spectralFlux(Audio)
            zero_crossings = librosa.zero_crossings(Audio, pad=True)
            zero_crossing = sum(zero_crossings)
            peaks, _ = find_peaks(Audio, height=0)
            peak_amp = np.mean(peaks)
            Thdn = THDN(np.uint8(Audio), Sample_rate)
            RMS = rms(Audio)
            ZCR = zcr(Audio)
            roll_off = librosa.feature.spectral_rolloff(y=Audio, sr=Sample_rate)
            spec = [cetroid, Density, Flux, zero_crossing, mfccs, peak_amp, Thdn, RMS, ZCR, roll_off[0, 0]]
            spectral.append(spec)
        np.save('Feature_2_' + str(n + 1) + '.npy', np.asarray(spectral))


# Optimization for Classification
an = 0
if an == 1:
    Best_Sol = []
    Fitness = []
    for n in range(no_of_Dataset):
        Feat_1 = np.load('Feature_1_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat_2 = np.load('Feature_2_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat_3 = np.load('Feature_3_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Selected features
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        Feat = np.concatenate((Feat_1, Feat_2, Feat_3), axis=1)
        Global_Vars.Feat_1 = Feat_1
        Global_Vars.Feat_2 = Feat_2
        Global_Vars.Feat_3 = Feat_3
        Global_Vars.Target = Target
        Npop = 10
        Chlen = Feat_1.shape[-1] + Feat_2.shape[-1] + Feat_3.shape[-1]  # Weight for features
        xmin = np.matlib.repmat(np.asarray([0.01]), Npop, Chlen)
        xmax = np.matlib.repmat(np.asarray([0.99]), Npop, Chlen)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("RSA...")
        [bestfit1, fitness1, bestsol1, time1] = RSA(initsol, fname, xmin, xmax, Max_iter)

        print("EVO...")
        [bestfit2, fitness2, bestsol2, time2] = EVO(initsol, fname, xmin, xmax, Max_iter)

        print("GSOA...")
        [bestfit3, fitness3, bestsol3, time3] = GSOA(initsol, fname, xmin, xmax, Max_iter)

        print("GOA...")
        [bestfit4, fitness4, bestsol4, time4] = GOA(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                       bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]
        Best_Sol.append(BestSol_CLS)
        Fitness.append(fitness)

    np.save('Fitness.npy', np.asarray(Fitness))
    np.save('BestSol_Feat.npy', np.asarray(Best_Sol))


# Weighted Feature Selection
an = 0
if an == 1:
    for n in range(no_of_Dataset):
        Feature_1 = np.load('Feature_1_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 1
        Feature_2 = np.load('Feature_2_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 2
        Feature_3 = np.load('Feature_3_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feat 3
        BestSol_feature = np.load('BestSol_Feat.npy', allow_pickle=True)
        Weighted_Fused_Feature = []
        for n in range(BestSol_feature.shape[0]):
            soln = BestSol_feature[n, :]
            Weighted_feat_1 = Feature_1 * soln[:Feature_1.shape[-1]]
            Weighted_feat_2 = Feature_2 * soln[Feature_1.shape[-1]:Feature_1.shape[-1] + Feature_2.shape[-1]]
            Weighted_feat_3 = Feature_3 * soln[Feature_1.shape[-1] + Feature_2.shape[-1]:]
            Weighted_Fused_Feat = np.concatenate((Weighted_feat_1, Weighted_feat_2, Weighted_feat_3), axis=1)
            Weighted_Fused_Feature.append(Weighted_Fused_Feat)
        np.save('Weighted_Fused_Feature_' + str(n + 1) + '.npy', Weighted_Fused_Feature)


# Classification
an = 0
if an == 1:
    Eval_All = []
    for n in range(no_of_Dataset):
        Feat = np.load('Weighted_Fused_Feature_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Feature_1
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the Target
        EVAL = []
        Batch_Size = [4, 16, 32, 64, 128]
        for BS in range(len(Batch_Size)):
            learnperc = round(Feat.shape[0] * 0.75)
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((5, 25))
            Eval[0, :], pred1 = Model_AutoEncoder(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
            Eval[1, :], pred2 = Model_TCN(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
            Eval[2, :], pred3 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
            Eval[3, :], pred4 = Model_Capsnet(Train_Data, Train_Target, Test_Data, Test_Target, BS=Batch_Size[BS])
            Eval[4, :], pred5 = Model_HDRLSTM_CapsNet(Feat, Target, BS=Batch_Size[BS])
            EVAL.append(Eval)
        Eval_All.append(EVAL)
    np.save('Eval_all_BS.npy', np.asarray(Eval_All))  # Save the Eval_all_BS


plot_conv()
ROC_curve()
Plot_learning_per()
Plot_Batch_size()
plot_Activation()
