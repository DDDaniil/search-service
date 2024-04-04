import time
import traceback
from pprint import pprint

import requests
import re
import os
from tika import parser

pages = ['00', '01', '03', '05', '06', '08', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22', '26',
         '28', '30', '31', '32', '33', '34', '35', '37', '39', '40', '41', '42', '43', '44', '45', '46', '47', '49',
         '51', '52', '53', '54', '55', '57', '58', '60', '62', '65', '68', '70', '74', '76', '78', '80', '81', '82',
         '83', '85', '86', '90', '91', '92', '93', '94', '97']


def get_urls(msc_class):
    url = f'https://arxiv.org/search/?query={msc_class}&searchtype=msc_class&abstracts=show&order=-announced_date_first&size=50'
    r = requests.get(url)
    print(r.text)
    urls = re.findall(r'".*?/pdf/.*?"', r.text)
    urls = [e[1:-1] for e in urls]
    pprint(urls)
    return urls


def download_pdf(url, save_path):
    s = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(s.content)


def save_all_files(classes):
    folder = 'arxiv_files/'
    for cl in classes:
        urls = get_urls(cl)
        save_folder = f'{folder}{cl}'
        try:
            os.mkdir(save_folder)
        except Exception as e:
            print(traceback.format_exc())
            pass

        urls = urls[:30]

        for url in urls:
            name = url[22:].replace('/', '.')
            file = f'{save_folder}/{name}.pdf'
            print(file)
            download_pdf(url, file)
            time.sleep(0.5)


def pars_bad_files(classes):
    folder = 'arxiv_files/'
    for cl in classes:
        print(cl)
        save_folder = f'{folder}{cl}'
        files = os.listdir(save_folder)
        for file in files:
            save_path = f'{save_folder}/{file}'
            try:
                with open(save_path, 'r') as f:
                    text = f.read()
                if '403 forbidden access denied' in text.lower():
                    file_url = f'https://arxiv.org/pdf/{file}'
                    print(file_url)
                    save_path = f'{save_folder}/{file}'
                    download_pdf(file_url, save_path)
            except UnicodeDecodeError:
                print(file, 'unicode')


if __name__ == '__main__':
    # get_urls('00')
    # download_pdf('https://arxiv.org/pdf/1306.0379')
    # print(pages[29:])
    # save_all_files(pages[29:])
    pars_bad_files(pages)
