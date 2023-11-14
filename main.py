from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from text_preprocessing import preprocess_data
from models import classify


def main():
    train_data = pd.read_csv('train_data (1).csv', names=['review', 'rating'])
    data, most_common_words, feature_set = preprocess_data(train_data, num_of_features=500)
    train_set, test_set = train_test_split(feature_set, test_size=0.2, random_state=42)
    classifier_type = 'MNB'
    accuracy, classifier = classify(train_set, test_set, classifier_type)
    print(f'Accuracy of {classifier_type} model is equal to: {accuracy}')

    save_classifier = open(f"{classifier_type}_classifier.pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

    # classifier_f = open("naivebayes.pickle",'rb')
    # classifier = pickle.load(classifier_f)
    # classifier_f.close()


if __name__ == "__main__":
    main()
