import os

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import PyPDF2
import pandas as pd
from tika import parser
from pymorphy2 import MorphAnalyzer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import streamlit as st

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def get_all_preprocessed_files(preprocessed_folder):
    files = []
    for obj in os.listdir(preprocessed_folder):
        path = preprocessed_folder + obj + '/'
        for file in os.listdir(path):
            files.append(f'{path}{file}')
    return files


def get_all_files_text(preprocessed_folder):
    paths = get_all_preprocessed_files(preprocessed_folder)  # пути к файлам
    for path in paths:
        with open(path, 'r') as file:
            file_class = path.split('/')[-2]
            yield file_class, file.read()


def get_files(preprocessed_folder, class_min_elem_count=0):
    files = get_all_files_text(preprocessed_folder)
    files = list(files)
    r = {}
    for file in files:
        l = r.get(file[0], [])
        r[file[0]] = l
        l.append(file[1])

    labels_name = [e for e in r.keys()]
    r = [e for e in r.items() if len(e[1]) > class_min_elem_count]
    files = [[e[0], i] for e in r for i in e[1]]

    tf_texts = [e[1] for e in files]
    score_list = [e[0] for e in files]
    return tf_texts, score_list


def preprocess_reviews(text_array):
    pattern = re.compile(r'\b\w{,3}\b')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    preprocessed_text_array = []

    for text in text_array:
        text = text.lower()

        final_text = pattern.sub('', text)

        words = word_tokenize(final_text)

        words = [word for word in words if word not in stop_words]

        words = [lemmatizer.lemmatize(word) for word in words]

        preprocessed_text = ' '.join(words)

        preprocessed_text_array.append(preprocessed_text)

    return preprocessed_text_array


def extract_text_from_uploaded_pdf(uploaded_file):
    pdf_text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page_num in range(len(pdf_reader.pages)):
        pdf_text += pdf_reader.pages[page_num].extract_text()
    return pdf_text


def create_vectors(tf_texts, score_list):
    X_train, X_test, y_train, y_test = train_test_split(tf_texts, score_list,
                                                        test_size=0.2, random_state=42,
                                                        stratify=score_list)
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                                 min_df=0.0, stop_words='english', sublinear_tf=True)

    response = vectorizer.fit_transform(X_train)
    test_vectorize = vectorizer.transform(X_test)

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(tf_texts, score_list)

    return vectorizer, model, response, test_vectorize, y_train, y_test


def categorize_text(text, model):
    category = model.predict([text])
    return category[0]


def create_and_fit_model(text_vectors, classes, c=1):
    classifier = LinearSVC(random_state=42, C=c)
    classifier.fit(text_vectors, classes)

    return classifier


def create_and_fit_model_knn(text_vectors, classes, c=1):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(text_vectors, classes)

    return classifier


def create_and_fit_model_nb(text_vectors, classes, c=1):
    nb = MultinomialNB()
    model = nb.fit(text_vectors, classes)

    return model


def preprocess_and_lemmatize_pdf(uploaded_file):
    raw_text = parser.from_buffer(uploaded_file.getvalue())['content']

    morph_analyzer = MorphAnalyzer()
    lemmatized_text = []
    for word in raw_text.split():
        parsed_word = morph_analyzer.parse(word)[0]
        lemma = parsed_word.normal_form
        lemmatized_text.append(lemma)

    return lemmatized_text


def get_f1_presicion_recall(y_pred, y_test):
    unique_classes = sorted(list(set(y_test)))
    res = precision_recall_fscore_support(y_test, y_pred, average=None, labels=unique_classes)
    results = {}
    for i in range(len(unique_classes)):
        results[unique_classes[i]] = [elem[i] for elem in res]
    res = precision_recall_fscore_support(y_test, y_pred, average='macro')
    results['all'] = res
    return results


def get_files_in_directory(directory_path):
    files_list = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    return files_list


def test_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    results = get_f1_presicion_recall(y_pred, y_test)
    accuracy = sum([e[0] == e[1] for e in zip(y_pred, y_test)]) / len(y_test)
    return results, accuracy


def main():
    st.sidebar.subheader('Параметры модели')
    x = st.sidebar.slider('C', value=1.0, min_value=0.0, max_value=2.0)
    x2 = st.sidebar.slider('Минимальное число элементов в классе', value=0, min_value=0, max_value=100)

    eng_files_preprocessed_folder = 'preprocessed_files/'
    ru_test_files_preprocessed_folder = 'collections_dml/collection_for_testing_msc_preprocessed/'
    files, classes = get_files(eng_files_preprocessed_folder, class_min_elem_count=x2)

    preprocess_files = preprocess_reviews(files)
    vectorizer, model_new, x_train, x_test, y_train, y_test = create_vectors(preprocess_files, classes)
    model = create_and_fit_model(x_train, y_train, c=x)
    data1, accuracy1 = test_model(model, x_test, y_test)


    texts, classes = get_files(ru_test_files_preprocessed_folder)
    texts_vectors = vectorizer.transform(texts)
    data2, accuracy2 = test_model(model, texts_vectors, classes)

    model_knn = create_and_fit_model_knn(x_train, y_train, c=x)
    data_knn, accuracy_knn = test_model(model_knn, x_test, y_test)

    model_nb = create_and_fit_model_nb(x_train, y_train, c=x)
    data_nb, accuracy_nb = test_model(model_nb, x_test, y_test)


    st.subheader('Результаты тестирования на англоязычных данных\n')
    st.text('Таблица метрик по каждому классу')
    d1 = pd.DataFrame(data1)
    d1 = d1.T
    d1.columns = ['precision', 'recall', 'f1', 'count']
    d1
    st.text('Таблица метрик по всем классам')
    dd1 = pd.DataFrame({'all': data1['all']}).T
    dd1.columns = ['precision', 'recall', 'f1', 'count']
    dd1
    st.text(f'Accuracy: {accuracy1}')

    d_knn = pd.DataFrame(data_knn)
    d_knn = d_knn.T
    d_knn.columns = ['precision', 'recall', 'f1', 'count']
    d_knn
    st.text('Таблица метрик по всем классам KNN')
    dd_knn = pd.DataFrame({'all': data_knn['all']}).T
    dd_knn.columns = ['precision', 'recall', 'f1', 'count']
    st.text(f'Accuracy for KNN: {accuracy_knn}')

    d_nb = pd.DataFrame(data_nb)
    d_nb = d_nb.T
    d_nb.columns = ['precision', 'recall', 'f1', 'count']
    d_nb
    st.text('Таблица метрик по всем классам NB')
    dd_nb = pd.DataFrame({'all': data_nb['all']}).T
    dd_nb.columns = ['precision', 'recall', 'f1', 'count']
    st.text(f'Accuracy for NB: {accuracy_nb}')

    st.subheader('Результаты тестирования на русскоязычных данных\n')
    st.text('Таблица метрик по каждому классу')
    d2 = pd.DataFrame(data2)
    d2 = d2.T
    d2
    st.text('Таблица метрик по всем классам')
    d2.columns = ['precision', 'recall', 'f1', 'count']
    dd2 = pd.DataFrame({'all': data2['all']}).T
    dd2.columns = ['precision', 'recall', 'f1', 'count']
    dd2
    st.text(f'Accuracy: {accuracy2}')

    st.title('Загрузка PDF файла')

    uploaded_file = st.file_uploader("Выберите PDF файл по которому будет производиться поиск тематически схожих "
                                     "документов", type="pdf")

    ## тестовый вариант загрузки файла через streamlit, для этого есть приложение на React
    if uploaded_file is not None:
        st.write("Файл успешно загружен!")
        resulting_text = extract_text_from_uploaded_pdf(uploaded_file)
        predicted_category_file = categorize_text(resulting_text, model_new)
        st.text(f'Predict category for file: {predicted_category_file}')
    else:
        st.write("Загрузите PDF файл")

    files_directory = get_files_in_directory('/collections_dml/collection_for_testing_msc/')

    if len(files_directory) > 0:
        st.write("Доступные PDF файлы:")
        for file in files_directory:
            file_path = os.path.join('/collections_dml/collection_for_testing_msc/', file)
            st.markdown(f"[{file}](./{file_path})")
    else:
        st.write("В указанной директории нет PDF файлов")


if __name__ == '__main__':
    main()
