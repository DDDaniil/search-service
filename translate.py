import os
import time

from tika import parser
import re
from pymorphy2 import MorphAnalyzer

from os import path
import string


class MathFile:
    def __init__(self, pdf_dir, txt_dir, article_dir, filename):
        self.pdf_dir = pdf_dir
        self.txt_dir = txt_dir
        self.filename = filename
        self.pdf_path = path.join(pdf_dir, filename)
        self.txt_path = path.join(txt_dir, filename.replace('.pdf', '.txt'))
        self.article_path = path.join(article_dir, filename.replace('.pdf', '.txt'))
        self.translate_path = path.join(translate_dir, filename.replace('.pdf', '.txt'))

    def get_pdf(self):
        raw = parser.from_file(self.pdf_path)
        return raw

    def get_text(self):
        raw = parser.from_file(self.pdf_path)
        return raw['content']

    def get_article(self):
        raw = parser.from_file(self.pdf_path)
        text = str(raw['content'])
        text = ''.join(text.split('-\n'))
        text = ' '.join(text.split('\n'))
        words = re.findall(r'\W*[А-ЯA-Z]{3,}\W*\s*[А-ЯA-Z]{3,}\W*\W*[А-ЯA-Z]{3,}\W*\W*[А-ЯA-Z]{3,}\W*', text)
        if len(words) != 0:
            text = text[text.index(words[0]):]
        return text

    def save_article(self):
        art = self.get_article()
        with open(self.article_path, 'w') as file:
            file.write(art)


main_folder = 'collections_dml/mathcenter_merge/'
preprocessed_folder = 'collections_dml/mathcentre_preprocessed/'
article_folder = 'collections_dml/articles'
translate_dir = 'collections_dml/mathcenter_translated'

bad_files = """16-pp.
17-pp.
18-pp.
27-pp.
28-pp.
29-pp.
35-pp.
39-pp.
41-pp.
47-pp.
48-pp.
49-pp.
50-pp.
52-pp.
6-pp.""".split('\n')
# 58-pp. 67-70.pdf
def get_all_files():
    files = []
    print(len(sorted(os.listdir(main_folder))))
    for obj in sorted(os.listdir(main_folder))[2200:]:
        if obj != '':
            file = MathFile(main_folder, preprocessed_folder, article_folder, obj)
            files.append(file)
    return files


def main():
    import translators
    files = get_all_files()
    print(bad_files)
    for file in files:
        print(file.pdf_path)
        if all([e not in file.filename for e in bad_files]):
            # print(file.get_article())
            art = file.get_article()
            print(art[:100])
            # file.save_article()
            # print([e for e in art[:100] if e in string.printable or (1040 <= ord(e) <= 1103)])
            # if len([e for e in art[:100] if e in string.printable or (1040 <= ord(e) <= 1103)]) != 10:
            #     print('битый')
            # lett = [ord(e) for e in art[:10] if e not in string.printable and not 1072 <= ord(e) <= 1103 and not 1040 <= ord(e) <= 1071]
            # print(lett)
            # lett = [e for e ]
            result = translators.google(file.get_article(), if_ignore_limit_of_length=True)
            print(result)
            with open(file.translate_path, 'w') as file:
                file.write(result)
            # input('ff')
            # lemmas = preprocessed_pdf(file)
            # with open(new_path, 'w') as file:
            #     file.write(' '.join(lemmas))
            print(file)
            time.sleep(1)

"""
16-pp.
17-pp.
18-pp.
27-pp.
28-pp.
29-pp.
35-pp.
39-pp.
41-pp.
47-pp.
48-pp.
49-pp.
50-pp.
52-pp.
6-pp.
"""
bads = ['16-pp.', '16-pp.']

if __name__ == '__main__':
    main()
