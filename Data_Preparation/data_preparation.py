#============================================================
#
#  Deep Learning BLW Filtering
#  Data preparation
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import numpy as np
import _pickle as pickle
from Data_Preparation import Prepare_QTDatabase, Prepare_NSTDB

def Data_Preparation(noise_version=1):

    print('Getting the Data ready ... ')

    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    Prepare_QTDatabase.prepare()
    Prepare_NSTDB.prepare()

    # Load QT Database
    with open('data/QTDatabase.pkl', 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)

    # Load NSTDB
    with open('data/NoiseALL.pkl', 'rb') as input:
        nstdb = pickle.load(input)

    #####################################
    # NSTDB
    #####################################

    [bw_signals, em_signals, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    em_signals = np.array(em_signals)
    ma_signals = np.array(ma_signals)


    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]
    
    em_noise_channel1_a = em_signals[0:int(em_signals.shape[0]/2), 0]
    em_noise_channel1_b = em_signals[int(em_signals.shape[0]/2):-1, 0]
    em_noise_channel2_a = em_signals[0:int(em_signals.shape[0]/2), 1]
    em_noise_channel2_b = em_signals[int(em_signals.shape[0]/2):-1, 1]
    
    ma_noise_channel1_a = ma_signals[0:int(ma_signals.shape[0]/2), 0]
    ma_noise_channel1_b = ma_signals[int(ma_signals.shape[0]/2):-1, 0]
    ma_noise_channel2_a = ma_signals[0:int(ma_signals.shape[0]/2), 1]
    ma_noise_channel2_b = ma_signals[int(ma_signals.shape[0]/2):-1, 1]



    #####################################
    # Data split
    #####################################
    if noise_version == 1:
        noise_test_bw = bw_noise_channel2_b
        noise_train_bw = bw_noise_channel1_a
        noise_test_em = em_noise_channel2_b
        noise_train_em = em_noise_channel1_a
        noise_test_ma = ma_noise_channel2_b
        noise_train_ma = ma_noise_channel1_a
    elif noise_version == 2:
        noise_test_bw = bw_noise_channel1_b
        noise_train_bw = bw_noise_channel2_a
        noise_test_em = em_noise_channel1_b
        noise_train_em = em_noise_channel2_a
        noise_test_ma = ma_noise_channel1_b
        noise_train_ma = ma_noise_channel2_a
    else:
        raise Exception("Sorry, noise_version should be 0 or 1")

    #####################################
    # QTDatabase
    #####################################

    beats_train = []
    beats_test = []

    # QTDatabese signals Dataset splitting. Considering the following link
    # https://www.physionet.org/physiobank/database/qtdb/doc/node3.html
    #  Distribution of the 105 records according to the original Database.
    #  | MIT-BIH | MIT-BIH |   MIT-BIH  |  MIT-BIH  | ESC | MIT-BIH | Sudden |
    #  | Arrhyt. |  ST DB  | Sup. Vent. | Long Term | STT | NSR DB	| Death  |
    #  |   15    |   6	   |     13     |     4     | 33  |  10	    |  24    |
    #
    # The two random signals of each pathology will be keep for testing set.
    # The following list was used
    # https://www.physionet.org/physiobank/database/qtdb/doc/node4.html
    # Selected test signal amount (14) represent ~13 % of the total

    test_set = ['sel123',  # Record from MIT-BIH Arrhythmia Database
                'sel233',  # Record from MIT-BIH Arrhythmia Database

                'sel302',  # Record from MIT-BIH ST Change Database
                'sel307',  # Record from MIT-BIH ST Change Database

                'sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database

                'sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database

                'sele0106',  # Record from European ST-T Database
                'sele0121',  # Record from European ST-T Database

                'sel32',  # Record from ``sudden death'' patients from BIH
                'sel49',  # Record from ``sudden death'' patients from BIH

                'sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'sel15814',  # Record from MIT-BIH Long-Term ECG Database
                ]


    # Creating the train and test dataset, each datapoint has 512 samples and is zero padded
    # beats bigger that 512 samples are discarded to avoid wrong split beats ans to reduce
    # computation.
    skip_beats = 0
    samples = 512
    qtdb_keys = list(qtdb.keys())

    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]

        for b in qtdb[signal_name]:

            b_np = np.zeros(samples)
            b_sq = np.array(b)

            # There are beats with more than 512 samples (could be up to 3500 samples)
            # Creating a threshold of 512 - init_padding samples max. gives a good compromise between
            # the samples amount and the discarded signals amount
            # before:
            # train: 74448  test: 13362
            # after:
            # train: 71893 test: 13306  (discarded train: ~4k datapoints test: ~50)

            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue

            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if signal_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)


    # Noise was added in a proportion from 0.2 to 2 times the ECG signal amplitude
    # Similar to
    # W. Muldrow, R.G. Mark, & Moody, G. B. (1984).
    # A noise stress test for arrhythmia detectors.
    # Computers in Cardiology, 381–384

    sn_train = []
    sn_test = []

    noise_index = 0
    
    def noise_pw(noise):
        mean = np.mean(noise)
        n = np.sqrt(np.mean((noise - mean) ** 2))
        
        return(n**2)
    
    def signal_pw(signal):
        range = max(signal) - min(signal)
        
        return(range**2 / 8)

    # Adding noise to train
    rnd_train_bw = np.random.randint(low=0, high=4, size=len(beats_train)) * 6.67
    rnd_train_em = np.random.randint(low=0, high=4, size=len(beats_train)) * 6.67
    rnd_train_ma = np.random.randint(low=0, high=4, size=len(beats_train)) * 6.67
    for i in range(len(beats_train)):
        noise_bw = noise_train_bw[noise_index:noise_index + samples]
        noise_em = noise_train_em[noise_index:noise_index + samples]
        noise_ma = noise_train_ma[noise_index:noise_index + samples]
        '''beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value_bw = np.max(noise_bw) - np.min(noise_bw)
        noise_max_value_em = np.max(noise_em) - np.min(noise_em)
        noise_max_value_ma = np.max(noise_ma) - np.min(noise_ma)
        Ase_bw = noise_max_value_bw / beat_max_value
        Ase_em = noise_max_value_em / beat_max_value
        Ase_ma = noise_max_value_ma / beat_max_value
        alpha_bw = rnd_train_bw[i] / Ase_bw
        alpha_em = rnd_train_em[i] / Ase_em
        alpha_ma = rnd_train_ma[i] / Ase_ma'''
        S = signal_pw(beats_train[i])
        N_bw = noise_pw(noise_bw)
        N_em = noise_pw(noise_em)
        N_ma = noise_pw(noise_ma)
        alpha_bw = np.sqrt(10 ** (-rnd_train_bw[i] / 10) * S / N_bw) / np.sqrt(3)
        alpha_em = np.sqrt(10 ** (-rnd_train_em[i] / 10) * S / N_em) / np.sqrt(3)
        alpha_ma = np.sqrt(10 ** (-rnd_train_ma[i] / 10) * S / N_ma) / np.sqrt(3)
        signal_noise = beats_train[i] + alpha_bw * noise_bw + alpha_em * noise_em + alpha_ma * noise_ma
        sn_train.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_train_bw) - samples) or noise_index > (len(noise_train_em) - samples) or noise_index > (len(noise_train_ma) - samples):
            noise_index = 0

    # Adding noise to test
    noise_index = 0
    rnd_test_bw = np.random.randint(low=0, high=4, size=len(beats_test)) * 6.67
    rnd_test_em = np.random.randint(low=0, high=4, size=len(beats_test)) * 6.67
    rnd_test_ma = np.random.randint(low=0, high=4, size=len(beats_test)) * 6.67

    # Saving the random array so we can use it on the amplitude segmentation tables
    np.save('rnd_test_bw.npy', rnd_test_bw)
    np.save('rnd_test_em.npy', rnd_test_em)
    np.save('rnd_test_ma.npy', rnd_test_ma)
    print('rnd_test_bw shape: ' + str(rnd_test_bw.shape))
    print('rnd_test_em shape: ' + str(rnd_test_em.shape))
    print('rnd_test_ma shape: ' + str(rnd_test_ma.shape))

    for i in range(len(beats_test)):
        noise_bw = noise_test_bw[noise_index:noise_index + samples]
        noise_em = noise_test_em[noise_index:noise_index + samples]
        noise_ma = noise_test_ma[noise_index:noise_index + samples]
        '''beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value_bw = np.max(noise_bw) - np.min(noise_bw)
        noise_max_value_em = np.max(noise_em) - np.min(noise_em)
        noise_max_value_ma = np.max(noise_ma) - np.min(noise_ma)
        Ase_bw = noise_max_value_bw / beat_max_value
        Ase_em = noise_max_value_em / beat_max_value
        Ase_ma = noise_max_value_ma / beat_max_value
        alpha_bw = rnd_train_bw[i] / Ase_bw
        alpha_em = rnd_train_em[i] / Ase_em
        alpha_ma = rnd_train_ma[i] / Ase_ma'''
        S = signal_pw(beats_test[i])
        N_bw = noise_pw(noise_bw)
        N_em = noise_pw(noise_em)
        N_ma = noise_pw(noise_ma)
        alpha_bw = np.sqrt(10 ** (-rnd_test_bw[i] / 10) * S / N_bw) / np.sqrt(3)
        alpha_em = np.sqrt(10 ** (-rnd_test_em[i] / 10) * S / N_em) / np.sqrt(3)
        alpha_ma = np.sqrt(10 ** (-rnd_test_ma[i] / 10) * S / N_ma) / np.sqrt(3)
        signal_noise = beats_test[i] + alpha_bw * noise_bw + alpha_em * noise_em + alpha_ma * noise_ma
        sn_test.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_test_bw) - samples) or noise_index > (len(noise_test_em) - samples) or noise_index > (len(noise_test_ma) - samples):
            noise_index = 0


    X_train = np.array(sn_train)
    y_train = np.array(beats_train)

    X_test = np.array(sn_test)
    y_test = np.array(beats_test)

    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)


    Dataset = [X_train, y_train, X_test, y_test]

    print('Dataset ready to use.')

    return Dataset
