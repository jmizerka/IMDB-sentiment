from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import accuracy


def classify(train_set,test_set, classifier_type='MNB'):

    classifier = SklearnClassifier(
        MultinomialNB() if classifier_type == 'MNB' else
        BernoulliNB() if classifier_type == 'BNB' else
        LogisticRegression() if classifier_type == 'LogReg' else
        SGDClassifier() if classifier_type == 'SGD' else
        SVC()
    )
    classifier.train(train_set)
    acc = accuracy(classifier, test_set)*100
    return acc, classifier
