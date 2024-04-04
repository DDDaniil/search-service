import os
import random

from tika import parser
import re
from pymorphy2 import MorphAnalyzer

main_folder = 'arxiv_files/'
preprocessed_folder = 'preprocessed_files/'
crossover_folder = 'preprocessed_files_crossover/'


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
    print(text)
    analyzer = MorphAnalyzer()
    lemmas = []
    for token in tokens:
        t = analyzer.parse(token)
        lemmas.append(t[0].normal_form)
    return lemmas


def crossover(files_list, split_count=3, new_texts_count_n=1):
    split_l = [[] for _ in range(split_count)]
    for file in files_list:
        words = file.split(' ')
        split_value = len(words) // split_count + 1
        new_words = [[]]
        for word in words:
            new_words[-1].append(word)
            if len(new_words[-1]) >= split_value:
                new_words.append([])
        for i in range(len(new_words)):
            small_text = ' '.join(new_words[i])
            split_l[i].append(small_text)
    result_files = []
    for i in range(len(files_list) * new_texts_count_n):
        text = ' '.join([random.choice(split_l[e]) for e in range(split_count)])
        result_files.append(text)
    return result_files


def crossover_with_max_value(files_list, max_value=30, split_count=5, new_texts_count_n=10):
    files_list.extend(crossover(files_list, split_count, new_texts_count_n))
    return files_list[:max_value]


def crossover_with_class(files, classes, split_count=3, new_texts_count_n=1, max_count=30):
    # print(files[0])
    # print(classes)
    data = list(zip(classes, files))
    r = {}
    for file in data:
        l = r.get(file[0], [])
        r[file[0]] = l
        l.append(file[1])

    res = []
    items = []
    for i in r.items():
        # print(i[0])
        # print(len(i[1]))
        # print(i[1])
        print(len(i[1]))
        d = crossover_with_max_value(i[1], max_value=max_count)
        for e in d:
            res.append(e)
            items.append(i[0])

    return res, items


def main():
    files = get_all_files_text()
    print(files)
    files = list(files)
    r = {}
    for file in files:
        l = r.get(file[0], [])
        r[file[0]] = l
        l.append(file[1])

    print(r)

    for class_name, files_list in r.items():
        all_files = crossover_with_max_value(files_list, max_value=60, split_count=3, new_texts_count_n=10)
        # all_files = []
        # all_files.extend(files_list)
        # all_files.extend(new_texts)
        for i in range(len(all_files)):
            try:
                os.mkdir(f'{crossover_folder}{class_name}')
            except:
                pass
            path = f'{crossover_folder}{class_name}/{i}'
            with open(path, 'w') as file:
                file.write(all_files[i])


if __name__ == '__main__':
    main()
