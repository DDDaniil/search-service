import os

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
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
    # print(len(files))
    r = {}
    for file in files:
        l = r.get(file[0], [])
        r[file[0]] = l
        l.append(file[1])

    # print(r)
    labels_name = [e for e in r.keys()]
    # print(labels_name)
    r = [e for e in r.items() if len(e[1]) > class_min_elem_count]
    # print(len(r))
    files = [[e[0], i] for e in r for i in e[1]]
    # print(len(files))

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
    clf = LinearSVC(random_state=42, C=c)
    clf.fit(text_vectors, classes)
    return clf


def preprocess_and_lemmatize_pdf(uploaded_file):
    # Предобработка и извлечение текста из PDF
    raw_text = parser.from_buffer(uploaded_file.getvalue())['content']

    # Лемматизация текста
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
    # pprint(results)
    accuracy = sum([e[0] == e[1] for e in zip(y_pred, y_test)]) / len(y_test)
    # print('Accuracy: %f' % (accuracy * 100))
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

    if uploaded_file is not None:
        st.write("Файл успешно загружен!")
        resulting_text = extract_text_from_uploaded_pdf(uploaded_file)
        # lemmas = preprocess_and_lemmatize_pdf(uploaded_file)
        # new_resulting_text = ' '.join(lemmas)
        # print('XXXXXXXXXXXXXXXXXXXXX: ', new_resulting_text)
        predicted_category_file = categorize_text(resulting_text, model_new)
        st.text(f'Predict category for file: {predicted_category_file}')

        # current_directory = os.getcwd()
        # predict_directory = current_directory.join(f'/collections_dml/collection_for_testing_msc/03')

        # files = get_files_in_directory(predict_directory)

        # if len(files) > 0:
        #    st.write("Доступные PDF файлы:")
        #    for file in files:
        #        file_path = os.path.join(predict_directory, file)
        #        st.markdown(f"[{file}](./{file_path})")
        # else:
        #    st.write("В указанной директории нет PDF файлов")
    else:
        st.write("Загрузите PDF файл")

    files_directory = get_files_in_directory('C:/Users/danil/Desktop/all_projects/ProjectPython/search-service/collections_dml/collection_for_testing_msc/03')

    if len(files_directory) > 0:
        st.write("Доступные PDF файлы:")
        for file in files_directory:
            file_path = os.path.join('C:/Users/danil/Desktop/all_projects/ProjectPython/search-service/collections_dml/collection_for_testing_msc/03', file)
            st.markdown(f"[{file}](./{file_path})")
    else:
        st.write("В указанной директории нет PDF файлов")


if __name__ == '__main__':
    main()
