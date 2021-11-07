import re
from sklearn import naive_bayes
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Text classification and Evaluation utilizing Naive Bayes
def DALE_NBFUNC():
    hamspam_data = load_files(r"CONSOLIDATE")  # folder containing the 2 categories of documents in individual folders.
    X, y = hamspam_data.data, hamspam_data.target
    documents = []
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        from nltk.stem import WordNetLemmatizer

        stemmer = WordNetLemmatizer()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    # Convert the word to a vector using BOW model.

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(documents).toarray()

    ''' Creating training and test sets of the data'''
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    '''train a classifier with the data'''

    # fit the training dataset on the NB classifier
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(X_train, y_train)

    '''Now predict on the testing data'''
    y_pred = Naive.predict(X_test)

    '''Print the evaluation metrices'''
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print('---------------------- FINISHED CONSOLIDATED----------------------')
    print()


# Call Function
DALE_NBFUNC()
