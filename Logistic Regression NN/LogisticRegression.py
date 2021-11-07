import re
from sklearn.datasets import load_files
from nltk.corpus import stopwords

# Text classification and Evaluation utilizing a LogisticRegression Neural Network

def DALE_LRFUNC():
    hmspm_data = load_files(r"CONSOLIDATE")  # folder containing the 2 categories of documents in individual folders.
    x, y = hmspm_data.data, hmspm_data.target

    documents = []
    for sen in range(0, len(x)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(x[sen]))

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
    x = vectorizer.fit_transform(documents).toarray()

    '''Using TFIDF instead of BOW, TFIDF also takes into account the frequency instead of just the occurance.
    calculated as:
    Term frequency = (Number of Occurrences of a word)/(Total words in the document)
    IDF(word) = Log((Total number of documents)/(Number of documents containing the word))
    TF-IDF is the product of the two.
    '''
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidfconverter = TfidfTransformer()
    x = tfidfconverter.fit_transform(x).toarray()

    ''' Creating a training and test set of the data'''

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)

    '''train classifier with data'''

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train, y_train)

    '''Now predict on the testing data'''

    y_pred = classifier.predict(x_test)

    '''Print evaluation metrice'''

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("The Logistic Regression Accuracy is: ", accuracy_score(y_test, y_pred))
    print('---------------------- FINISHED CONSOLIDATED ----------------------')
    print()


# Call Function

DALE_LRFUNC()


