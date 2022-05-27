import numpy as np

from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier

from energy import EDy
from cross_validation import CV_estimator


def main(dataset='T1B', estimator_name='OVRCLLR', n_bags=5000, first_bag=0, master_seed=2032):

    # loading data
    data = np.genfromtxt('../LEQUA/datasets/' + dataset + '/public/training_data.txt', skip_header=1, delimiter=',')
    X_train = data[:, 1:]
    y_train = data[:, 0].astype(int)

    # estimator
    if estimator_name == 'OVRCLLR':
        skf_train = StratifiedKFold(n_splits=10, shuffle=True, random_state=master_seed)
        estimator = OneVsRestClassifier(estimator=CalibratedClassifierCV(LogisticRegression(C=0.01, class_weight='balanced')), n_jobs=-1)
        estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
    elif estimator_name == 'LR':    
        skf_train = StratifiedKFold(n_splits=10, shuffle=True, random_state=master_seed)
        estimator = LogisticRegression(C=0.01, max_iter=1000, class_weight='balanced')
        estimator_train = CV_estimator(estimator=estimator, cv=skf_train)
    else:
        raise ValueError('Unknwon estimator')

    # fitting estimator train    
    print('Fitting Training Estimator')
    estimator_train.fit(X_train, y_train)
    print('Training Estimator fitted', flush=True)
    probs_train = estimator_train.predict_proba(X_train)
    
    # fitting quantifier
    quantifier = EDy()
    quantifier.fit(X_train, y_train, predictions_train=probs_train)
    print('Quantifier fitted')

    # predictions file
    file_preds = open('results/predictions-' + dataset + '-' + estimator_name + '.csv', 'w')

    n_classes = 28

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
    main(dataset='T1B', estimator_name='OVRCLLR', n_bags=5000, first_bag=0, master_seed=2032)
