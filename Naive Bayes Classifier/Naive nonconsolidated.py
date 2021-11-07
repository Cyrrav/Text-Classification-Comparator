import re
from sklearn import naive_bayes
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Non-consolidated corpus data classification and testing utilizing the Naive Bayes classifier algorithm

def DALE_NBFUNC2():
    enron_train = load_files("NONCONSOLIDATE/Training")
    X1, Y1 = enron_train.data, enron_train.target

    enron_test = load_files("NONCONSOLIDATE/Testing")
    X2, Y2 = enron_test.data, enron_test.target

    documents = []
    for sen in range(0, len(X1)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X1[sen]))

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

    documents2 = []
    for sen in range(0, len(X2)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X2[sen]))

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

        documents2.append(document)

    # Convert the word to a vector using a BOW model.
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X1_train = vectorizer.fit_transform(documents).toarray()
    X2_test = vectorizer.fit_transform(documents2).toarray()

    '''train a classifier with data'''

    Naive = naive_bayes.MultinomialNB()
    Naive.fit(X1_train, Y1)

    '''Predict the testing data'''
    Y1_pred = Naive.predict(X2_test)

    '''Print evaluation matrices'''

    print(confusion_matrix(Y2, Y1_pred))
    print(classification_report(Y2, Y1_pred))
    print("Naive Bayes accuracy =", accuracy_score(Y2, Y1_pred))
    print('------------------------ FINISHED NON-CONSOLIDATED ---------------------------')


# Call Function
DALE_NBFUNC2()
