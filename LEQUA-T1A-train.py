import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

from sklearn.metrics.pairwise import euclidean_distances

from xgboost import XGBClassifier

from em import EM
from df import HDy, DFy
from energy import EDy
from cross_validation import CV_estimator
from metrics import topsoe, geometric_mean


def absolute_error(prevs, prevs_hat):
    assert prevs.shape == prevs_hat.shape, 'wrong shape {prevs.shape} vs. {prevs_hat.shape}'
    return abs(prevs_hat - prevs).mean(axis=-1)


def relative_absolute_error(p, p_hat, eps=None):
    def __smooth(prevs, epsilon):
        n_classes = prevs.shape[-1]
        return (prevs + epsilon) / (epsilon * n_classes + 1)

    p = __smooth(p, eps)
    p_hat = __smooth(p_hat, eps)
    return (abs(p-p_hat)/p).mean(axis=-1)


def load_training_set(dfile):
    data = np.genfromtxt(dfile, skip_header=1, delimiter=',')

    X = data[:, 1:]
    y = data[:, 0].astype(int)
    return X, y


def load_testing_bag(dfile):
    X = np.genfromtxt(dfile, skip_header=1, delimiter=',')
    return X


def load_prevalences(dfile):
    data = np.genfromtxt(dfile, skip_header=1, delimiter=',')

    prevalences = data[:, 1:]
    return prevalences


def main(path, dataset, estimator_name, n_bags=1000, bag_inicial=0, master_seed=2032):

    method_name = ['EM',
                   'HDy-30b', 
                   'HDy-40b', 
                   'HDy-50b',
                   'PDFy-30b-topsoe',
                   'PDFy-40b-topsoe',
                   'PDFy-50b-topsoe',
                   'CDFy-100b-L1', 
                   'EDy',
                  ]

    X_train, y_train = load_training_set(path + dataset + '/public/training_data.txt')

    # classifiers are fitted by each object (all methods will use exactly the same predictions)
    # but they checked whether the estimator is already fitted (by a previous object) or not
    if estimator_name == 'RF':
        skf_train = StratifiedKFold(n_splits=3, shuffle=True, random_state=master_seed)
        estimator = RandomForestClassifier(n_estimators=500, max_depth=2,
                                           random_state=master_seed, class_weight='balanced')
        estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
        estimator_test = None # RandomForestClassifier(n_estimators=500, max_depth=2,
                              #                   random_state=master_seed, class_weight='balanced')
    elif estimator_name == 'CalibratedRF':
        estimator = RandomForestClassifier(n_estimators=500, max_depth=2,
                                           random_state=master_seed)
        estimator_train = CalibratedClassifierCV(estimator)
        estimator_test = None                                       
    elif estimator_name == 'LR':
        skf_train = StratifiedKFold(n_splits=20, shuffle=True, random_state=master_seed)
        estimator = LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced')
        estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
        estimator_test = None # LogisticRegression(C=0.01, class_weight='balanced')
    elif estimator_name == 'CLLR':
        skf_train = StratifiedKFold(n_splits=20, shuffle=True, random_state=master_seed)
        estimator = CalibratedClassifierCV(LogisticRegression(C=0.01, class_weight='balanced'))
        estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
        estimator_test = None # CalibratedClassifierCV(LogisticRegression(C=0.01, class_weight='balanced'))
    elif estimator_name == 'NB':
        skf_train = StratifiedKFold(n_splits=3, shuffle=True, random_state=master_seed)
        estimator = MultinomialNB()
        estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
        estimator_test = None     
    elif estimator_name == 'GP':
        #  estimator_train = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), optimizer=None)
        estimator_train = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
        estimator_test = None
    elif estimator_name == 'CalibratedGP':
        #  estimator = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), optimizer=None)
        estimator = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
        estimator_train = CalibratedClassifierCV(estimator)
        estimator_test = None    
    elif estimator_name == 'Stacking':
        estimators = [ ('rf', RandomForestClassifier(n_estimators=500, max_depth=2, random_state=master_seed, class_weight='balanced')),
                       ('lg', LogisticRegression()) ]
        estimator_train = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        estimator_test = None
    elif estimator_name == 'XGboost':
        skf_train = StratifiedKFold(n_splits=5, shuffle=True, random_state=master_seed)
        #  objective='binary:logistic' eval_metric='auc' error
        estimator = XGBClassifier(eval_metric='error', max_depth=4, n_estimators=500, 
                                  use_label_encoder=False, random_state=master_seed)
        estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
        estimator_test = None 
        #XGBClassifier(eval_metric='error', max_depth=4, n_estimators=500, 
        #                               use_label_encoder=False, random_state=master_seed)
    else:
        raise ValueError('Unknwon estimator')

    print('Fitting Training Estimator')
    estimator_train.fit(X_train, y_train)
    print('Training Estimator fitted', flush=True)
    probs_train = estimator_train.predict_proba(X_train)

    #  predictions_train = None
    print('Prediction_train computed')
    # EM
    em = EM()
    em.fit(X_train, y_train, predictions_train=probs_train)
    print('EM fitted')
    #  HDY
    # hdy4 = HDy(n_bins=4, bin_strategy='equal_width')
    # hdy4.fit(X_train, y_train, predictions_train=probs_train)
    # print('HDy 4b fitted')
    # #  HDY
    # hdy8 = HDy(n_bins=8, bin_strategy='equal_width')
    # hdy8.fit(X_train, y_train, predictions_train=probs_train)
    # print('HDy 8b fitted')
    # #  HDY
    # hdy16 = HDy(n_bins=16, bin_strategy='equal_width')
    # hdy16.fit(X_train, y_train, predictions_train=probs_train)
    # print('HDy 16b fitted')
    #  HDY
    hdy30 = HDy(n_bins=30, bin_strategy='equal_width')
    hdy30.fit(X_train, y_train, predictions_train=probs_train)
    print('HDy 4b fitted')
    #  HDY
    hdy40 = HDy(n_bins=40, bin_strategy='equal_width')
    hdy40.fit(X_train, y_train, predictions_train=probs_train)
    print('HDy 8b fitted')
    #  HDY
    hdy50 = HDy(n_bins=50, bin_strategy='equal_width')
    hdy50.fit(X_train, y_train, predictions_train=probs_train)
    print('HDy 16b fitted')
    #  PDFY - topsoe
    pdfy_topsoe30 = DFy(distribution_function='PDF', distance=topsoe, n_bins=30, bin_strategy='equal_count')
    pdfy_topsoe30.fit(X_train, y_train, predictions_train=probs_train)
    print('PDFy-topsoe 30b fitted')
    #  PDFY - topsoe
    pdfy_topsoe40 = DFy(distribution_function='PDF', distance=topsoe, n_bins=40, bin_strategy='equal_count')
    pdfy_topsoe40.fit(X_train, y_train, predictions_train=probs_train)
    print('PDFy-topsoe 40b fitted')
    #  PDFY - topsoe
    pdfy_topsoe50 = DFy(distribution_function='PDF', distance=topsoe, n_bins=50, bin_strategy='equal_count')
    pdfy_topsoe50.fit(X_train, y_train, predictions_train=probs_train)
    print('PDFy-topsoe 50b fitted')
    #  CDFy-L1
    cdfy_l1 = DFy(distribution_function='CDF', n_bins=100, distance='L1', bin_strategy='equal_count')
    cdfy_l1.fit(X_train, y_train, predictions_train=probs_train)
    print('CDF-L1 fitted')
    #  EDy
    edy = EDy()
    edy.fit(X_train, y_train, predictions_train=probs_train)
    print('EDy fitted')
    
    print('Fitting Estimator Test')
    if estimator_test is None:
        estimator_test = estimator_train
    else:
        estimator_test.fit(X_train, y_train)
    print('Estimator test fitted')

    prev_true = load_prevalences(path + dataset + '/public/dev_prevalences.txt')

    results_mae = np.zeros((n_bags, len(method_name)))
    results_rmae = np.zeros((n_bags, len(method_name)))
    results_pred = np.zeros((n_bags, len(method_name)))
    for n_bag in range(n_bags):
        print('Validation Bag #%d' % n_bag, flush=True)
        X_test = load_testing_bag(path + dataset + '/public/dev_samples/' + str(bag_inicial + n_bag) + '.txt')

        probs_test = estimator_test.predict_proba(X_test)
        prev_preds = [
            em.predict(X=None, predictions_test=probs_test),
            hdy30.predict(X=None, predictions_test=probs_test),
            hdy40.predict(X=None, predictions_test=probs_test),
            hdy50.predict(X=None, predictions_test=probs_test),
            pdfy_topsoe30.predict(X=None, predictions_test=probs_test),
            pdfy_topsoe40.predict(X=None, predictions_test=probs_test),
            pdfy_topsoe50.predict(X=None, predictions_test=probs_test),
            cdfy_l1.predict(X=None, predictions_test=probs_test),
            edy.predict(X=None, predictions_test=probs_test),
        ]
        for n_method, prev_pred in enumerate(prev_preds):
            results_mae[n_bag, n_method] = absolute_error(prev_true[bag_inicial + n_bag, :], prev_pred)
            results_rmae[n_bag, n_method] = relative_absolute_error(prev_true[bag_inicial + n_bag, :], prev_pred,
                                                                    eps=0.002)  # T1A eps=0.002, T1B eps=0.0005
            results_pred[n_bag, n_method] = prev_pred[1]

    #  printing and saving results
    filename = '/home/juanjo/cuantificacion/QU-Ant/results/Validation-' + dataset + '-' + estimator_name    
    #  all - pred
    np.savetxt(filename + '-PRED-all-' + str(n_bags) + '.txt',
               np.hstack((prev_true[bag_inicial:bag_inicial + n_bags, 1].reshape(-1, 1), results_pred)),
               fmt='%.5f', delimiter=",", header='true_p,' + ','.join(method_name))
    #  avg - mae
    file_avg = open(filename + '-MAE-avg-' + str(n_bags) + '.txt', 'w')
    avg = np.mean(results_mae, axis=0)
    print('\nMAE results')
    print('-' * 22)
    for n_method, method in enumerate(method_name):
        file_avg.write('%-15s%.5f\n' % (method, avg[n_method]))
        print('%-15s%.5f' % (method, avg[n_method]))
    #  all - mae
    np.savetxt(filename + '-MAE-all-' + str(n_bags),
               np.hstack((prev_true[bag_inicial:bag_inicial + n_bags, 1].reshape(-1, 1), results_mae)),
               fmt='%.5f', delimiter=",", header='true_p,' + ','.join(method_name))

    #  avg - rmae
    file_avg = open(filename + '-RMAE-avg-' + str(n_bags) + '.txt', 'w')
    avg = np.mean(results_rmae, axis=0)
    print('\nRMAE results')
    print('-' * 22)
    for n_method, method in enumerate(method_name):
        file_avg.write('%-15s%.5f\n' % (method, avg[n_method]))
        print('%-15s%.5f' % (method, avg[n_method]))
    #  all - rmae
    np.savetxt(filename + '-RMAE-all-' + str(n_bags) + '.txt',
               np.hstack((prev_true[bag_inicial:bag_inicial + n_bags, 1].reshape(-1, 1), results_rmae)),
               fmt='%.5f', delimiter=",", header='true_p,' + ','.join(method_name))


if __name__ == '__main__':
    main(path='../LEQUA/datasets/', dataset='T1A',
         estimator_name='LR', n_bags=1000, bag_inicial=0, master_seed=2032)
