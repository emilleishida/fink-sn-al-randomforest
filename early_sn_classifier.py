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
        PSF-fit magnitude from ZTF
    sigmapsf: float
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

    # hard code ZTF filters
    filters = ['g', 'r']
    
    lc_flux_sig = []

    for index in range(pdf.shape[0]):

        name = pdf['objectId'].values[index]
        sntype = pdf['TNS'].values[index]
    
        for f in range(1,3):
            filter_flag = pdf['cfid'].values[index] == f
    
            mjd = pdf['cjd'].values[index][filter_flag]
            mag = pdf['cmagpsf'].values[index][filter_flag]
            magerr = pdf['csigmapsf'].values[index][filter_flag] 

            fluxcal, fluxcal_err = mag2fluxcal_snana(mag, magerr)
        
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
    temp_labels = features['type'].values[train_indexes]

    # make sure there are the correct ratio of Ias and non-Ias
    while sum(temp_labels == 'SN Ia') != max(1, initial_training // 2):
        # this is an array of 5 indexes
        train_indexes = np.random.choice(np.arange(0, features.shape[0]),
                                                  size=initial_training,
                                                  replace=False)
        temp_labels = features['type'].values[train_indexes]

    if screen:
        print('\n temp_labels = ', temp_labels, '\n')

    # set training
    train_flag = np.array([item in train_indexes for item in range(features.shape[0])])
    
    train_Ia_flag = features['type'].values[train_flag] == 'SN Ia'
    data.train_labels = train_Ia_flag.astype(int)
    data.train_features = features[train_flag].values[:,2:]
    data.train_metadata = features[['id', 'type']][train_flag]
    
    # set test set as all objs apart from those in training
    test_indexes = np.array([i for i in range(features.shape[0])
                             if i not in train_indexes])
    test_ia_flag = features['type'].values[test_indexes] == 'SN Ia'
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
        data.evaluate_classification()

        # choose object to query
        indx = data.make_query(strategy=strategy, batch=batch)

        # update training and test samples
        data.update_samples(indx, loop=loop)

        # save metrics for current state
        data.save_metrics(loop=loop, output_metrics_file=output_metrics_file,
                          batch=batch, epoch=loop)

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop,
                                 full_sample=False)

    
def main():

    
    create_matrix = False
    clean = False
    
    nloops = 40
    strategy = 'RandomSampling'
    initial_training = 6
    
    seed = np.random.seed()
    
    for v in range(50):
        
    
        output_metrics_file = 'results/metrics/' + strategy + '/metrics_' + strategy + '_' + str(v) + '.dat'
        output_queried_file = 'results/metrics/' + strategy + '/queried_' + strategy + '_'+ str(v) + '.dat'
        output_prob_root = 'results/class_prob/' + strategy + '/v' + str(0) + '/class_prob_' + strategy + 'v' + str(v)
    
        for name in ['results/', 'results/' + strategy + '/', 'results/class_prob/', 'results/class_prob/' + strategy,
                     'results/class_prob/' + strategy + '/v' + str(0), 'results/metrics/', 'results/metrics/' + strategy]:
            if not os.path.isdir(name):
                os.makedirs(name)
    
        if create_matrix:
            # read data
            pdf1 = pd.read_parquet('data/fink_cross_tns_nov2019.parquet')
            pdf2 = pd.read_parquet('data/fink_cross_tns_sept2020.parquet')
            pdf = pd.concat([pdf1, pdf2], ignore_index=True)
    
            # convert data to appropriate format
            lc_flux = convert_full_dataset(pdf)
    
            # build feature matrix
            matrix = featurize_full_dataset(lc_flux)
      
            # clean matrix
            if clean:
                flag1 = np.logical_and(matrix['a_g'].values == 0, matrix['b_g'].values == 0)
                flag2 = np.logical_and(matrix['a_r'].values == 0, matrix['b_r'].values == 0)
                pop = np.logical_and(~flag1, ~flag2)
    
                matrix_clean = matrix[pop]
            else:
                matrix_clean = matrix
        
            matrix_clean.to_csv('data/features_matrix.csv', index=False)
        
        else:
            matrix_clean = pd.read_csv('data/features_matrix.csv', index_col=False)
    
    
        # build samples
        data = build_samples(matrix_clean, initial_training=initial_training, screen=True)
    
        # perform learnin loop
        learn_loop(data, nloops=nloops, strategy=strategy, 
                   output_metrics_file=output_metrics_file, 
                   output_queried_file=output_queried_file,
                   classifier='RandomForest', seed=seed,
                   batch=1, screen=True, output_prob_root=output_prob_root)
    
if __name__ == '__main__':
    main()