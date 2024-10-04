import os

import PyPDF2 as PyPDF2
from flask import Flask, request, jsonify
import sklearn
import pickle


# Загрузка обученной модели
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,PATCH,DELETE')
    return response


def extract_text_from_uploaded_pdf(uploaded_file):
    pdf_text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page_num in range(len(pdf_reader.pages)):
        pdf_text += pdf_reader.pages[page_num].extract_text()
    return pdf_text


@app.route("/predictText", methods=["POST"])
def predictText():
    # Получение текста из запроса
    text = request.json["text"]

    # Извлечение текста из PDF-файла
    # text_from_pdf = extract_text_from_uploaded_pdf(text)

    # Предсказание с помощью модели
    prediction = model.predict([text])

    # Подготовка ответа
    response = {
        "prediction": str(prediction[0])
    }

    return jsonify(response)


@app.route("/predictFile", methods=["POST"])
def predictFile():
    # Получение файла из запроса
    file = request.files["file"]

    # Извлечение текста из PDF-файла
    text_from_pdf = extract_text_from_uploaded_pdf(file)

    # Предсказание с помощью модели
    prediction = model.predict([text_from_pdf])

    # Подготовка ответа
    response = {
        "prediction": str(prediction[0])
    }

    return jsonify(response)


@app.route('/api/files', methods=['POST'])
def get_files():
    # Получение текста из запроса
    path = request.json["path"]
    print('//////////////////')
    print(os.listdir(path))

    files = [f for f in os.listdir(path) if f.endswith('.pdf')]

    return jsonify(files)


@app.route("/test", methods=["GET"])
def test():
    response = {
        "prediction": str('ответ пришел с сервера')
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run()