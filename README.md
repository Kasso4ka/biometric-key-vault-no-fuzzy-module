# Версия Python: 3.12

# Недостающий файл
GitHub не пропустил файл размером более 100МБ. По ссылке ниже необходимо его скачать, это файл модели распознавания лиц (был скачан пакетом `insightface`)

https://drive.google.com/file/d/1N2qRb39TN4MVw9VJNGm1LYN0W8V2SbEP/view?usp=sharing

Поместить в корень проекта: `./rec.onnx`

# Визуализации
(Предварительно установить зависимости из `training-requirements.txt`)

Запустить `nn.ipynb`.

# Обучение
(Предварительно установить зависимости из `training-requirements.txt`)

В `nn.ipynb` есть закомменченные сегменты кода. Если нужно провести процесс создания датасета и тренировки, то нужно раскомментить соответствующие участки в конце первой и третьей ячеек.

# FastAPI сервис
(Предварительно установить зависимости из `service-requirements.txt`)

`uvicorn service:app --host 0.0.0.0 --port 8000 --limit-concurrency 10 --timeout-keep-alive 300 --h11-max-incomplete-event-size 104857600`

Либо же можно провести билд по имеющемуся здесь `Dockerfile`.