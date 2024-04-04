import os
from tika import parser
import re
from pymorphy2 import MorphAnalyzer

main_folder = 'arxiv_files/'
preprocessed_folder = 'preprocessed_files/'
# preprocessed_folder = 'preprocessed_files_crossover/'


def get_all_files():
    files = []
    for obj in os.listdir(main_folder):
        path = main_folder + obj + '/'
        for file in os.listdir(path):
            files.append(f'{path}{file}')
    return files


def get_all_preprocessed_files():
    files = []
    for obj in os.listdir(preprocessed_folder):
        path = preprocessed_folder + obj + '/'
        for file in os.listdir(path):
            files.append(f'{path}{file}')
    return files


def get_all_files_text():
    # возвращает название класса и текст документа
    paths = get_all_preprocessed_files()
    for path in paths:
        with open(path, 'r') as file:
            file_class = path.split('/')[1]
            yield file_class, file.read()


def preprocessed_pdf(path):
    raw = parser.from_file(path)
    text = ' '.join(re.findall(r'([a-z]+|[0-9]+)', raw['content'].lower()))
    tokens = text.split(' ')
    analyzer = MorphAnalyzer()
    lemmas = []
    for token in tokens:
        t = analyzer.parse(token)
        lemmas.append(t[0].normal_form)
    return lemmas


def main():
    files = get_all_files()
    for file in files:
        path = file.split('/')
        new_path = file.replace(main_folder, preprocessed_folder)
        new_path = new_path.replace('.pdf', '.txt')
        try:
            os.mkdir(f'{preprocessed_folder}{path[1]}')
        except:
            pass
        lemmas = preprocessed_pdf(file)
        with open(new_path, 'w') as file:
            file.write(' '.join(lemmas))
        print(file)


if __name__ == '__main__':
    main()
