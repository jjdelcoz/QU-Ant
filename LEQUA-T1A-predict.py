import numpy as np

from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from df import DFy
from energy import EDy
from metrics import topsoe
from cross_validation import CV_estimator


def main(dataset='T1A', n_bags=5000, first_bag=0, master_seed=2032):

    # loading data
    data = np.genfromtxt('../LEQUA/datasets/' + dataset + '/public/training_data.txt', skip_header=1, delimiter=',')
    X_train = data[:, 1:]
    y_train = data[:, 0].astype(int)

    # estimator
    skf_train = StratifiedKFold(n_splits=20, shuffle=True, random_state=master_seed)
    estimator = CalibratedClassifierCV(LogisticRegression(C=0.01, class_weight='balanced'))
    estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
    
    # fitting estimator train    
    print('Fitting Training Estimator')
    estimator_train.fit(X_train, y_train)
    print('Training Estimator fitted', flush=True)
    probs_train = estimator_train.predict_proba(X_train)
    
    # fitting quantifier
    quantifier = DFy(distribution_function='PDF', distance=topsoe, n_bins=40, bin_strategy='equal_count')
    quantifier.fit(X_train, y_train, predictions_train=probs_train)
    print('Quantifier fitted')

    # predictions file
    file_preds = open('predictions-' + dataset + '.csv', 'w')

    n_classes = 2
    
    file_preds.write('id')
    for i in range(n_classes):
        file_preds.write(',%d' % (i))
    file_preds.write('\n')
    prevalences = np.zeros((n_bags, n_classes))
    for n_bag in range(n_bags):
        
        print('Testing Bag #%d' % n_bag, flush=True)
        X_test = np.genfromtxt('/mnt/hdd/juanjo/' + dataset + '/public/test_samples/' + str(first_bag + n_bag) + '.txt', skip_header=1, delimiter=',')

        predictions_test = estimator_train.predict_proba(X_test)
        p = quantifier.predict(X=None, predictions_test=predictions_test)
        p[p < 0] = 0
        prevalences[n_bag, :] = p
        file_preds.write('%d,' % (n_bag))
        prevalences[n_bag, :].tofile(file_preds, sep=',')
        file_preds.write('\n')
    
    file_preds.close()    
             
if __name__ == '__main__':
    main(dataset='T1A', n_bags=5000, first_bag=0, master_seed=2032)
