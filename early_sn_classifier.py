# Copyright 2020-2021 
# Author: Emille Ishida
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import actsnclass
import pandas as pd
import numpy as np
import os

from actsnclass import DataBase
from classifier_sigmoid import get_sigmoid_features_dev


def mag2fluxcal_snana(magpsf: float, sigmapsf: float):
    """ Conversion from magnitude to Fluxcal from SNANA manual
    Parameters
    ----------
    magpsf: float
        PSF-fit magnitude from ZTF.
    sigmapsf: float
        Error on PSF-fit magnitude from ZTF. 
    
    Returns
    ----------
    fluxcal: float
        Flux cal as used by SNANA
    fluxcal_err: float
        Absolute error on fluxcal (the derivative has a minus sign)
    """
    if magpsf is None:
        return None, None
    fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
    fluxcal_err = 9.21034 * 10 ** 10 * np.exp(-0.921034 * magpsf) * sigmapsf

    return fluxcal, fluxcal_err


def convert_full_dataset(pdf: pd.DataFrame):
    """Convert an entire data set from mag to fluxcal.
    
    Parameters
    ----------
    pdf: pd.DataFrame
        Read directly from parquet files.
        
    Returns
    -------
    pd.DataFrame
        Columns are ['objectId', 'type', 'MJD', 'FLT', 
        'FLUXCAL', 'FLUXCALERR'].
    """
    # Ia types in TNS
    Ia_group = ['SN Ia', 'SN Ia-91T-like', 'SN Ia-91bg-like', 'SN Ia-CSM', 
                'SN Ia-pec', 'SN Iax[02cx-like]']
    
    # hard code ZTF filters
    filters = ['g', 'r']
    
    lc_flux_sig = []

    for index in range(pdf.shape[0]):

        name = pdf['candid'].values[index]
        sntype_orig = pdf['TNS'].values[index]
        
        if sntype_orig in Ia_group:
            sntype = 'Ia'
        else:
            sntype = sntype_orig
    
        for f in range(1,3):
            
            if isinstance(pdf.iloc[index]['cfid'], str):
                ffs = np.array([int(item) for item in pdf.iloc[index]['cfid'][1:-1].split()])
                filter_flag = ffs == f
                mjd = np.array([float(item) for item in pdf.iloc[index]['cjd'][1:-1].split()])[filter_flag]
                mag = np.array([float(item) for item in pdf.iloc[index]['cmagpsf'][1:-1].split()])[filter_flag]
                magerr = np.array([float(item) for item in pdf.iloc[index]['csigmapsf'][1:-1].split()])[filter_flag]
            else:
                filter_flag = pdf['cfid'].values[index] == f
                mjd = pdf['cjd'].values[index][filter_flag]
                mag = pdf['cmagpsf'].values[index][filter_flag]
                magerr = pdf['csigmapsf'].values[index][filter_flag] 

            fluxcal = []
            fluxcal_err = []
            for k in range(len(mag)):
                f1, f1err = mag2fluxcal_snana(mag[k], magerr[k])
                fluxcal.append(f1)
                fluxcal_err.append(f1err)
        
            for i in range(len(fluxcal)):
                lc_flux_sig.append([name, sntype, mjd[i], filters[f - 1],
                                    fluxcal[i], fluxcal_err[i]])

    lc_flux_sig = pd.DataFrame(lc_flux_sig, columns=['id', 'type', 'MJD', 
                                                     'FLT', 'FLUXCAL', 
                                                     'FLUXCALERR'])

    return lc_flux_sig


def featurize_full_dataset(lc: pd.DataFrame):
    """Get complete feature matrix for all objects in the data set.
    
    Parameters
    ----------
    lc: pd.DataFrame
        Columns should be: ['objectId', 'type', 'MJD', 'FLT', 
        'FLUXCAL', 'FLUXCALERR'].
        
    Returns
    -------
    pd.DataFrame
        Features for all objects in the data set. Columns are:
        ['objectId', 'type', 'a_g', 'b_g', 'c_g', 'snratio_g', 
        'chisq_g', 'nrise_g', 'a_r', 'b_r', 'c_r', 'snratio_r',
        'chisq_r', 'nrise_r']
    """
    
    # columns in output data matrix
    columns = ['id', 'type', 'a_g', 'b_g', 'c_g', 
               'snratio_g', 'chisq_g', 'nrise_g', 'a_r', 'b_r', 'c_r',
               'snratio_r', 'chisq_r', 'nrise_r']

    features_all = []

    for indx in range(np.unique(lc['id'].values).shape[0]):

        print('indx: ', indx)
        
        name = np.unique(lc['id'].values)[indx]

        obj_flag = lc['id'].values == name
        sntype = lc[obj_flag].iloc[0]['type']
    
        line = [name, sntype]
    
        features = get_sigmoid_features_dev(lc[obj_flag][['MJD',
                                                          'FLT',
                                                          'FLUXCAL',
                                                          'FLUXCALERR']])
        
        for j in range(len(features)):
            line.append(features[j])
        
        features_all.append(line)
    
    feature_matrix = pd.DataFrame(features_all, columns=columns)

    return feature_matrix


# this was taken from https://github.com/COINtoolbox/ActSNClass/blob/master/actsnclass/database.py
def build_samples(features: pd.DataFrame, initial_training: int,
                 frac_Ia=0.5, screen=False):
    """Build initial samples for Active Learning loop.
    
    Parameters
    ----------
    features: pd.DataFrame
        Complete feature matrix. Columns are: ['objectId', 'type', 
        'a_g', 'b_g', 'c_g', 'snratio_g', 'chisq_g', 'nrise_g', 
        'a_r', 'b_r', 'c_r', 'snratio_r', 'chisq_r', 'nrise_r']
        
    initial_training: int
        Number of objects in the training sample.
    frac_Ia: float (optional)
        Fraction of Ia in training. Default is 0.5.
    screen: bool (optional)
        If True, print intermediary information to screen.
        Default is False.
        
    Returns
    -------
    actsnclass.DataBase
        DataBase for active learning loop
    """
    data = DataBase()
    
    # initialize the temporary label holder
    train_indexes = np.random.choice(np.arange(0, features.shape[0]),
                                     size=initial_training, replace=False)
    
    Ia_flag = features['type'].values == 'Ia'
    Ia_indx = np.arange(0, features.shape[0])[Ia_flag]
    nonIa_indx =  np.arange(0, features.shape[0])[~Ia_flag]
    
    indx_Ia_choice = np.random.choice(Ia_indx, size=max(1, initial_training // 2),
                                      replace=False)
    indx_nonIa_choice = np.random.choice(nonIa_indx, 
                        size=initial_training - max(1, initial_training // 2),
                        replace=False)
    train_indexes = list(indx_Ia_choice) + list(indx_nonIa_choice)
    
    temp_labels = features['type'].values[np.array(train_indexes)]

    if screen:
        print('\n temp_labels = ', temp_labels, '\n')

    # set training
    train_flag = np.array([item in train_indexes for item in range(features.shape[0])])
    
    train_Ia_flag = features['type'].values[train_flag] == 'Ia'
    data.train_labels = train_Ia_flag.astype(int)
    data.train_features = features[train_flag].values[:,2:]
    data.train_metadata = features[['id', 'type']][train_flag]
    
    # set test set as all objs apart from those in training
    test_indexes = np.array([i for i in range(features.shape[0])
                             if i not in train_indexes])
    test_ia_flag = features['type'].values[test_indexes] == 'Ia'
    data.test_labels = test_ia_flag.astype(int)
    data.test_features = features[~train_flag].values[:, 2:]
    data.test_metadata = features[['id', 'type']][~train_flag]
    
    # set metadata names
    data.metadata_names = ['id', 'type']
    
    # set everyone to queryable
    data.queryable_ids = data.test_metadata['id'].values
    
    if screen:
        print('Training set size: ', data.train_metadata.shape[0])
        print('Test set size: ', data.test_metadata.shape[0])
        print('  from which queryable: ', len(data.queryable_ids))
        
    return data


# This was slightly modified from https://github.com/COINtoolbox/ActSNClass/blob/master/actsnclass/learn_loop.py
def learn_loop(data: actsnclass.DataBase, nloops: int, strategy: str,
               output_metrics_file: str, output_queried_file: str,
               classifier='RandomForest', batch=1, screen=True, 
               output_prob_root=None, seed=42, nest=1000):
    """Perform the active learning loop. All results are saved to file.
    
    Parameters
    ----------
    data: actsnclass.DataBase
        Output from the build_samples function.
    nloops: int
        Number of active learning loops to run.
    strategy: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    output_metrics_file: str
        Full path to output file to store metric values of each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently only 'RandomForest' is implemented.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    n_est: int (optional)
        Number of trees. Default is 1000.
    output_prob_root: str or None (optional)
        If str, root to file name where probabilities without extension!
        Default is None.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    seed: int (optional)
        Random seed.
    """

    for loop in range(nloops):

        if screen:
            print('Processing... ', loop)

        # classify
        data.classify(method=classifier, seed=seed, n_est=nest)
        
        if isinstance(output_prob_root, str):
            data_temp = data.test_metadata.copy(deep=True)
            data_temp['prob_Ia'] = data.classprob[:,1]
            data_temp.to_csv(output_prob_root + '_loop_' + str(loop) + '.csv', index=False)
            
        # calculate metrics
        data.evaluate_classification(screen=screen)

        # choose object to query
        indx = data.make_query(strategy=strategy, batch=batch, seed=seed, screen=screen)
        print('indx: ', indx)
        
        # update training and test samples
        data.update_samples(indx, loop=loop)

        # save metrics for current state
        data.save_metrics(loop=loop, output_metrics_file=output_metrics_file,
                          batch=batch, epoch=loop)

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop,
                                 full_sample=False)
        
        
        
def build_matrix(fname_output: str, dirname_input: str, n: int,
                fname_raw_output=None, new_raw_file=False, 
                input_raw_file=None):
    """Build full feature matrix to file.
    
    Parameters
    ----------
    fname_output: str
        Full path to output file.  
    dirname_input: str
        Full path to directory including all parquet files.
    n: int
        Number of simbad objects to choose.
    fname_raw_output: str (optinal)
        Full path to filename containing raw data.
        Only used if new_raw_file == True.
    input_raw_file: str (optional)
        Full path to input raw data matrix.
        Only used if new_raw_file == False.
    new_raw_file: bool (optional)
        If True generate new input matrix from parquet files.
        Default is False.
    
        
    Returns
    -------
    pd.DataFrame
        Features matrix including all non-Ias in TNS until Mar/2021
        and Ias in Jan-April/2020 + Sep and Nov/2019.
    """
    if new_raw_file:
        data_temp = []
        flist = os.listdir(dirname_input)
        simbad = False
        tns = False
        # read all tns file and 1 random simbad file
        for name in flist:
            if ('simbad' in name and not simbad):
                d1 = pd.read_parquet(dirname_input + name)
                data_temp.append(d1.sample(n, replace=False))
                simbad = True
            elif 'tns' in name and not tns:
                d1 = pd.read_parquet(dirname_input + name)
                data_temp.append(d1)
                #tns = True

        pdf7 = pd.concat(data_temp, ignore_index=True)
        pdf7.fillna(-99, inplace=True)
        pdf7.to_csv(fname_raw_output, index=False)
        
    else:
        pdf7 = pd.read_csv(input_raw_file, index_col=False)
        if ' ' in pdf7.keys()[0]:
            pdf7 = pd.read_csv(input_raw_file, delim_whitespace=True)

    # convert data to appropriate format
    lcs2 = convert_full_dataset(pdf7)

    # build feature matrix
    m = featurize_full_dataset(lcs2)

    # drop zeros
    m_final2 = m.replace(0, np.nan).dropna()
    m_final3 = m_final2.sample(frac=1).reset_index(drop=True)
    
    matrix = m_final3
    
    matrix.to_csv(fname_output, index=False)

    print('Extract features from  ', matrix.shape[0], 'objects.')
    
    return matrix


def main():

    
    create_matrix = True
    fname_features_matrix = 'data/features_for_emille.csv'
    #fname_features_matrix = 'data/features_05JUL2021_15k.csv'
    fname_raw_output = 'data/data_for_emille.dat'
    dirname_input = '../../data/AL_data/'
    dirname_output = 'results_for_emille/'
    
    nloops = 60
    strategy = 'UncSampling'
    initial_training = 10
    frac_Ia_tot = 0.5
    n_realizations = 100
    marco = False             # flag input matrix created by Marco
    n = 15000
    new_raw_file = False
    input_raw_file = fname_raw_output
    
    
    features_names = ['a_g', 'b_g', 'c_g', 'snratio_g', 'chisq_g', 'nrise_g', 
                          'a_r', 'b_r', 'c_r', 'snratio_r', 'chisq_r', 'nrise_r']
    
    if create_matrix:
        matrix_clean = build_matrix(fname_output=fname_features_matrix, dirname_input=dirname_input, 
                                    n=n, fname_raw_output=fname_raw_output, new_raw_file=new_raw_file,
                                    input_raw_file=input_raw_file)
        print(np.unique(matrix_clean['type'].values))
        
    else:
        matrix_clean = pd.read_csv(fname_features_matrix, comment='#')
        if marco:
            matrix_clean.drop(labels=['id', 'redshift', 'code', 'sample'], inplace=True, axis=1)
            matrix_clean.rename(columns={'index': 'id'}, inplace=True)
    
    for v in range(n_realizations):
        
    
        output_metrics_file = dirname_output + '/' + strategy + '/metrics/metrics_' + strategy + '_v' + str(v) + '.dat'
        output_queried_file = dirname_output + '/' + strategy + '/queries/queried_' + strategy + '_v'+ str(v) + '.dat'
        output_prob_root = dirname_output + '/' + strategy + '/class_prob/v' + str(v) + '/class_prob_' + strategy + '_'
    
        for name in [dirname_output + '/', dirname_output + '/' + strategy + '/', 
                     dirname_output + '/' + strategy + '/class_prob/',
                     dirname_output + '/' + strategy + '/class_prob/v' + str(v) + '/',
                     dirname_output + '/' + strategy + '/metrics/', 
                     dirname_output + '/' + strategy + '/queries/',
                     dirname_output + '/' + strategy + '/training_samples/', 
                     dirname_output + '/' + strategy + '/test_samples/']:
            if not os.path.isdir(name):
                os.makedirs(name)    
    
        # build samples        
        data = build_samples(matrix_clean, initial_training=initial_training, screen=True)
        
        # save initial data        
        train = pd.DataFrame(data.train_features, columns=features_names)
        train['objectId'] = data.train_metadata['id'].values
        train['type'] = data.train_metadata['type'].values
        train.to_csv(dirname_output + '/' + strategy + '/training_samples/initialtrain_v' + str(v) + '.csv', index=False)
        
        test = pd.DataFrame(data.test_features, columns=features_names)
        test['objectId'] = data.test_metadata['id'].values
        test['type'] = data.test_metadata['type'].values
        test.to_csv(dirname_output + '/' + strategy + '/test_samples/initial_test_v' + str(v) + '.csv', index=False)        
    
        # perform learnin loop
        learn_loop(data, nloops=nloops, strategy=strategy, 
                   output_metrics_file=output_metrics_file, 
                   output_queried_file=output_queried_file,
                   classifier='RandomForest', seed=None,
                   batch=1, screen=True, output_prob_root=output_prob_root)
    
if __name__ == '__main__':
    main()
