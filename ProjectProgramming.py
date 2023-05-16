#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

import typing as tp
from sklearn.ensemble import RandomForestRegressor

from pylab import rcParams
rcParams['figure.figsize'] = 15, 7

sns.set(palette='Set2', font_scale=1.3)
from warnings import filterwarnings
filterwarnings('ignore')


# # Сведение задачи прогнозирования временного ряда к регрессии

# ## 1. Различные ML модели 

# ## 1.1 Задача
# Рассмотрим [датасет](https://www.kaggle.com/c/demand-forecasting-kernels-only/overview) с kaggle соревнования по прогнозированию спроса на товары. Это довольно простой и чистый датасет. Попробуем на нем разные подходы к прогнозированию временных рядов. Данные можно скачать [здесь](https://drive.google.com/file/d/1sZ2GmRUUwBg60ZSWt6cQWeOXe2HAvDxR/view?usp=share_link)
# 
# Данные содержат следующие колонки: 
# * date &mdash; дата;
# * store &mdash; ID магазина;
# * item &mdash; ID товара;
# * sales &mdash; количество продаж.
# 
# В датасете содержится информация про 50 товаров в 10 магазинах за 5 лет.

# In[ ]:


from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


DATA_PATH = '/content/drive/MyDrive/demand-forecasting-kernels-only.zip'
get_ipython().system(' unzip $DATA_PATH')


# In[ ]:


data = pd.read_csv('train.csv', parse_dates=['date'])
data.head()


# Как мы видим, датасет содержит информацию о магазинах, товарах и продажах. Выберем первый магазин и первый товар и будем предсказывать спрос на выбранный товар в данном магазине, используя последний год для сравнения моделей.
# 
# Выделим соотвествущие данные из датасета.

# In[ ]:


# Задаем магазин и продукт
store, item = 1, 1

# Выделяем только те данные, которые относятся к данному магазину и продукту
data = data[(data['store'] == store) & (data['item'] == item)]

# ВНИМАНИЕ: Дату уставнавливаем как индекс
data = data.set_index('date')

# Выделяем данные о продажах
data = data['sales']


# In[ ]:


data.shape


# ## **Задание** 1.2 Подготовка данных
# Разделите данные на трейн и тест для обучения и тестирования результатов соотвественно. На тест отправьте данные за последний год, т.е. последние 365 элементов.Текст, выделенный полужирным шрифтом

# In[ ]:


test_size = 365
data_train = data[:-test_size]
data_test  = data[-test_size:]


# Визуализируйте полученные данные. Визуализировать данные нужно с самого начала. Это помогает провалидировать данные и выделить некоторые закономерности.

# In[ ]:


plt.figure(figsize=(15, 5))
plt.title('Дневные продажи первого товара в первом магазине')
plt.plot(data_train, label='train')
plt.plot(data_test, label='test')
plt.legend();


# Видите ли вы какую-нибудь зависимость? В продажах можно заметить некую переиодичность (наверно, товар сезонный). Также видно, что с каждым годом среднее кол-во продаж товара немного увеличивается

# ## **Задание** 1.3 Метрики
# Теперь вернемся к самой задаче. Прежде чем ее решать, зададим метрики, по которым мы будем определять, какая из моделей лучше: MSE, MAE, MAPE. Допишите функцию, которая будет считать качество моделей.
# 
# Используйте функции из `sklearn.metrics`

# In[ ]:


compare_table = None


def add_results_in_comparison_table(method: str, y_true, y_forecast) -> pd.DataFrame:
    """
    Добавляет новую строчку в таблицу compare_table с результатами текущей модели.
    Если ранее модель была уже добавлена в таблицу, то старая строчка перезапишется на новую.
    
    Параметры:
    - method: имя модели.
    - y_true: истинные значения.
    - y_forecast: предсказанные значения.
    """
    
    # Обращаемся к глобальной переменной
    global compare_table
    
    # Считаем метрики
    result_row = {
        'method': method,
        "MSE": mean_squared_error(y_true, y_forecast),
        "MAE": mean_absolute_error(y_true, y_forecast),
        "MAPE": mean_absolute_percentage_error(y_true, y_forecast)
    }
    
    # Записываем результат в таблицу
    if compare_table is None:
        compare_table = pd.DataFrame([result_row])
    else:
        if method in list(compare_table['method']):
            compare_table = compare_table[compare_table['method'] != method]

        compare_table = pd.concat([compare_table, pd.DataFrame([result_row])])
        compare_table.index = np.arange(len(compare_table))
    return compare_table


# ## **Задание** 1.4 Простая модель / baseline
# Для начала сделаем некоторую эмпирическую модель. Она будет считать среднее за каждый год с учетом дней недели. Полученное среднее как раз будет являться прогнозом на будущее.

# In[ ]:


temp_data = pd.DataFrame(data_train)
# выделяем день недели
temp_data['weekofyear'] = data_train.index.weekofyear
temp_data['dayofweek'] = data_train.index.dayofweek
# считаем среднее за каждый год с учетом дня недели
mean_sales = temp_data.groupby(['weekofyear', 'dayofweek'])['sales'].mean()
display(mean_sales)

simple_prediction = []
for index in data_test.index:
    simple_prediction.append(mean_sales.loc[(index.weekofyear, index.dayofweek)])


# Допишите функцию для реализации результатов.

# In[ ]:


def plot_results(y_to_train, y_to_test, y_forecast):
    """
        Функция для визуализации временного ряда и предсказания.
        
        Параметры:
            - y_to_train: pd.Series
                Временной ряд, на котором обучалась модель.
            - y_to_test: pd.Series
                Временной ряд, который предсказывает модель.
            - y_forecast: array
                Предсказания модели.
            - plot_conf_int: bool 
                Надо ли строить предсказательного интервал.
            - left_bound: array
                Левая граница предсказательного интервала.
            - right_bound: array
                Правая граница предсказательного интервала.
    """

    plt.figure(figsize=(15, 5))
    plt.title('Дневные продажи объекта 1 в магазине 1', fontsize=15)
    plt.plot(y_to_train, label='train')
    plt.plot(y_to_test, label='test')
    plt.plot(y_to_test.index, y_forecast, label='prediction')
    plt.legend()
    plt.show()


# In[ ]:


plot_results(data_train, data_test, simple_prediction)


# Хорошо ли выглядит результат?
# 
# Для простой модели выглядит очень даже неплохо. Посчитаем метрики и сохраним результат.

# In[ ]:


add_results_in_comparison_table('Simple mean model', data_test, simple_prediction)


# ## **Задание** 1.5 Работа с признаками 1
# Далее мы обучим с страндартную модель регрессии. Для начала научимся извлекать признаки из дат.
# 
# Вспомним, что мы установили дату индексом датафрейма, причем дата была распознана при считывании датасета с помощью параметра `parse_dates`

# In[ ]:


data_train.index


# In[ ]:


idx = data_train.index[0]
idx


# Видим, индекс датафрейма (`data_train.index`) эквивалентен списку, а его элементами являются объекты типа `Timestamp`. Для таких объектов можно сразу извлекать различные части дат. Пример:

# In[ ]:


idx.year, idx.month, idx.day


# Для начала преобразуем дату, выделив из даты день, месяц, год и т.д. Для этого допишите функцию ниже. Почти для всех ключей названия атрибутов совпадают.

# In[ ]:


def create_date_features(date):
    """Создает фичи из даты"""
    
    row = {}
    row['dayofweek'] = date.dayofweek
    row['quarter'] = date.quarter
    row['month'] = date.month
    row['year'] = date.year
    row['dayofyear'] = date.dayofyear
    row['dayofmonth'] = date.day
    row['weekofyear'] = date.weekofyear
    return row


# С помощью следующей функции создадим датасет для обучения.

# In[ ]:


def create_only_date_train_features(y_series):
    """
        Создает обучающий датасет из признаков, полученных из дат для y_series
    """
    
    time_features = pd.DataFrame([create_date_features(date) for date in y_series.index])
    return time_features, y_series


# In[ ]:


X_train, y_train = create_only_date_train_features(data_train)
display(X_train.head())
display(y_train.head())


# ## **Задание** 1.5 Работа с признаками 2
# 
# Поработаем еще с признаками. На этот раз добавим сдвиги по времени. Таким образом модель сможет использовать информацию из прошлого, для составления прогноза на будущее.
# 
# В библиотеке pandas удобным образом реализованы сдвиги методом `shift` у датафреймов. Посмотрим как он работает

# In[ ]:


sample = data_train[:10]
sample


# Теперь применим сдвиг

# In[ ]:


sample.shift(1, axis=0)


# Видно, что значения во втором столбце сдвинулись на единицу, причем в данных образовались пропуски, поскольку для начального значения предыдущего нет. Допишите функцию для создания данных со сдвигом. Помните, что у нас данные поступают каждый день, то есть сдвиг на 1 отвечает сдвигу на 1 день, сдвиг 7 отвечает сдвигу на 1 неделю, сдвиг на 365 отвечает сдвигу на год.

# In[ ]:


def create_date_and_shifted_train_features(
    y_series, shifts=5, week_seasonal_shifts=1, year_seasonal_shifts=1
):
    """
    Создает обучающий датасет из признаков, полученных из дат и значений ряда ранее.
    При этом используются значения ряда со сдвигами на неделю и год назад.
    Параметры:
        - y_series
            временной ряд.
        - shifts
            дневной сдвиг (сколько дней учитываем).
        - week_seasonal_shifts
            недельный сдвиг (сколько недель учитываем).
        - year_seasonal_shifts
            годовой сдвиг (сколько лет учитываем).
    """
    
    curr_df, y = create_only_date_train_features(y_series)
    curr_df.index = y_series.index

    # применяем сдвиг по дням
    for shift in range(1, shifts + 1):
        curr_df[f'shift_{shift}'] = y_series.shift(shift, axis=0)

    # применяем сдвиг по неделям
    for shift in range(1, week_seasonal_shifts + 1):
        curr_df[f'week_seasonal_shift_{shift}'] = y_series.shift(shift*7, axis=0)
    
    # применяем сдвиг по годам
    for shift in range(1, year_seasonal_shifts + 1):
        curr_df[f'year_seasonal_shift_{shift}'] = y_series.shift(shift*365, axis=0)
    y = y_series
    
    # удалим первые строчки с nan
    drop_indices = curr_df.index[curr_df.isna().sum(axis=1) > 0]
    curr_df = curr_df.drop(index=drop_indices)
    y = y.drop(index=drop_indices)
    return curr_df, y


# *Также* зададим функцию для того, чтобы получать аналогичные признаки на тесте.

# In[ ]:


def date_and_shift_features_generator_for_test(date, previous_y):
    """Функция создания признаков из дат исдвигов ряда для тестовых дат"""
    
    row = create_date_features(date)
    for shift in range(1, SHIFT + 1):
        row[f'shift_{shift}'] = previous_y[-1 * shift]
    for shift in range(1, WEEK_SHIFT + 1):
        row[f'week_seasonal_shift_{shift}'] = previous_y[-1 * shift * 7]
    for shift in range(1, YEAR_SHIFT + 1):
        row[f'year_seasonal_shift_{shift}'] = previous_y[-1 * shift * 365]
    return row


# Зададим сами сдвиги.

# In[ ]:


SHIFT = 5       # дневной сдвиг
WEEK_SHIFT = 2  # недельный сдвиг
YEAR_SHIFT = 1  # годовой сдвиг


# Получим новые признаки.

# In[ ]:


X_train, y_train = create_date_and_shifted_train_features(
    data_train, 
    shifts=SHIFT, 
    week_seasonal_shifts=WEEK_SHIFT,
    year_seasonal_shifts=YEAR_SHIFT
)


# In[ ]:


X_train.head(5)


# ## **Задание** 1.6 Реализация рекурсивной стратегии
# 
# Реализуйте рекурсивную стратегию предсказания моделью.

# In[ ]:


def recursive_prediction(model, test_dates, y_to_train, features_creation_function):
    """
    Функция для рекурсивного предсказания для дат, указанных в test_dates.
    
    Параметры:
        - model
            МЛ-модель.
        - test_dates
            массив с датами, в которые надо сделать предсказания.
        - features_creation_function
            функция для создания тестовых признаков
    """
    predictions = []
    previous_y = list(y_to_train)
    
    for date in test_dates:

        # получаем признаки для тестовых данных из тестовой даты и предыдущих значений
        row = features_creation_function(date, previous_y)
        curr_test = pd.DataFrame([row])

        # выоплняем предсказание моделью
        curr_prediction = model.predict(curr_test)[0]

        # добавляем текущее предсказание к предыдущем значениям
        previous_y.append(curr_prediction)
        # сохраняем текущее предсказание для вердикта на тесте
        predictions.append(curr_prediction)
    return np.array(predictions)


# Посмотрим, как пользоваться нашими функциями на примере модели случайного леса `RandomForest` и двух видов признаков (даты, даты + предыдущие значения)

# Подготовка данных

# In[ ]:


X_train, y_train = create_only_date_train_features(data_train)

# Если мы хотим использовать и сдвинутые значения, данные готовятся так

# X_train, y_train = create_date_and_shifted_train_features(
#     data_train, 
#     shifts=SHIFT, 
#     week_seasonal_shifts=WEEK_SHIFT,
#     year_seasonal_shifts=YEAR_SHIFT
# )


# Обучим модель на признаках из дат

# In[ ]:


get_ipython().run_cell_magic('time', '', 'random_forest = RandomForestRegressor(n_estimators=300)\nrandom_forest.fit(X_train, y_train)')


# Получим предсказания. В данном случае для функции `recursive_prediction` в качестве аргумента `features_creation_function` нужно использовать функцию, создающие признаки из дат, предыдущие значения в данном случае не нужны. Однако в функции выше мы использовали в качестве аргументов и даты, и предыдущие значения, поэтому нам понадобится конструкция 
# 
# `lambda date, previous_y: create_date_features(date)`
# 
# которая задаст функцию уже от двух аргументов.

# In[ ]:


random_forest_predictions = recursive_prediction(
    random_forest, data_test.index, data_train, 
    lambda date, previous_y: create_date_features(date)
)


# Для дополнительного использования сдвинутых дат в эту функцию в качестве последнего аргумента нужно подавать функцию `date_and_shift_features_generator_for_test`, определенную выше.
# 
# А для подготовки данных нужно использовать функцию `create_date_and_shifted_train_features`.

# ## **Задание** 1.7 Обучение классической модели
# 
# Обучите модель `RandomForest` на двух видах признаков. Визуализируйте результаты в каждом случае и добавляйте метрики в таблицу.

# ---
# 
# Используем только даты

# In[ ]:


X_train, y_train = create_only_date_train_features(data_train)

random_forest = RandomForestRegressor(n_estimators=300)
random_forest.fit(X_train, y_train)


# In[ ]:


random_forest_predictions = recursive_prediction(
    random_forest, data_test.index, data_train, 
    lambda date, previous_y: create_date_features(date)
)


# Отобразим результаты.

# In[ ]:


plot_results(data_train, data_test, random_forest_predictions)


# Посчитаем метрики

# In[ ]:


add_results_in_comparison_table('RandomForest', data_test, random_forest_predictions)


# ---- 
# 
# Используем дополнительно сдвинутые признаки

# In[ ]:


X_train, y_train = create_date_and_shifted_train_features(data_train, shifts=SHIFT, week_seasonal_shifts=WEEK_SHIFT, year_seasonal_shifts=YEAR_SHIFT)

random_forest = RandomForestRegressor(n_estimators=300)
random_forest.fit(X_train, y_train)


# In[ ]:


random_forest_predictions = recursive_prediction(
    random_forest, data_test.index, data_train, 
    date_and_shift_features_generator_for_test
)


# Отобразим результаты

# In[ ]:


plot_results(data_train, data_test, random_forest_predictions)


# Посчитаем метрики.

# In[ ]:


add_results_in_comparison_table('RandomForest+shifted features', data_test, random_forest_predictions)


# **Вывод:** После того, как мы применили random forest к модели, а потом использовали shift с каждым разом мы получали всё лучшие результаты, об этом можно судить по таблице выше. Признаки, к которым мы применяли shift также можно видеть выше.

# ---
# 
# ## 2. Обучение нейронных сетей
# 
# ## **Задание** 2.1 `Conv1d`
# 
# В pytorch 1d свертка реализована в классе `nn.Conv1d`. Параметры у этого класса такие же, как у свертки 2d, однако на вход она ожидает уже трехмерный тензор вида `(batch_size, C_in, seq_len)`, где 
# * `batch_size` - размер батча
# * `seq_len` - длина последовательности
# * `C_in` - число входных каналов, которое совпадает с размерностью каждого вектора в последовательности
# 
# Посмотрим, как она работает.

# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


batch_size = 10
seq_len = 16
in_channels = 32

inp = torch.randn((batch_size, in_channels, seq_len))


# In[ ]:


# параметры свертки
out_channels = 64
ksize = 5

conv = nn.Conv1d(in_channels, out_channels, ksize)
conv(inp).shape


# Для разных значений паддинга посмотрите на размерность выхода.

# In[ ]:


for padding in range(5):
  conv = nn.Conv1d(in_channels, out_channels, ksize, padding=padding)
  print(padding, conv(inp).shape)


# Для разных значений страйда посмотрите на размерность выхода

# In[ ]:


for stride in range(1, 5):
  conv = nn.Conv1d(in_channels, out_channels, ksize, padding=2, stride=stride)
  print(stride, conv(inp).shape)


# согласуются ли результаты с обычными свертками? Да, согласуются. Размер увеличивается с ростом padding и уменьшается с ростом stride

# ## **Задание** 2.2 
# 
# ## Параметры и применение рекуррентных сетей
# 
# Перед тем, как использовать нейронные сети, вспомним, какие гиперпараметры у них есть и как следует их использовать.
# 
# ![](https://cdn.shortpixel.ai/client/q_glossy,ret_img,w_1200/http://dprogrammer.org/wp-content/uploads/2019/04/RNN-vs-LSTM-vs-GRU-1200x361.png)

# ![proga.jpg](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQECAgMCAgICAgQDAwIDBQQFBQUEBAQFBgcGBQUHBgQEBgkGBwgICAgIBQYJCgkICgcICAj/2wBDAQEBAQICAgQCAgQIBQQFCAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAj/wAARCAEVA6oDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/90ABAAF/9oADAMBAAIRAxEAPwD+/UcrwNtGTyDn60jZ3ZAoLNkigB4ORmgYGeAKZn1yD64p3yqAO1ADePYDNP8AXgZpB3w2403gccflQA7njGB7UcjuMe9ICpPPX3oGGHHBoAdkZxnmmnj5vujvSEr1GM047flzQB//0P79h3HGfagA5OcEUmOR0/KkU5O0/MKAF/i4bNO9jjNM4B5I/Kg4Cj+KgB3JxkCgZ4yQPam8cc8+tGR8uMmgCSkACjimEAbV6047T8x5FADqbye2Pxo42jdSf3PWgD//0f79+fQfnQOOOB6U3KkAtikx1zx/wGgB59jigEHOKaWwwHQUDnIJDCgB9FNAHGMikAAbqc0AO7cc0cDJpMAdwDSjoKAEHcjGfrTqj4H+elLzxz/47QB//9L+/ik4PoRTRjA4/SjjhTyaAHdvSlqPHJP/ANejjH95f5UASUnPPT2phxwOn4UfLhvvYoAkpBkHGOKYcZ6jP0oOOMDtQA/n0H50tR7h8uOBQMbQATjpQB//0/79+c+1LUY2jaehpePl6rQA7nPtQB7AGmfKc4JWg56/MBQBJSYHoKjJyRkYFL8u3+LFADxjoO1LTOc9Wx9KToVIwBjvQBJSZB6HNR7tvC9KVmBHvQB//9T+/c47nH40tRjaeMtRu69z/OgB+SemPzpaZxgKx5pDgrkc8YoAfnPTB/GlqPgY52n6UnHP+FAEtJ24po69MHvTflJAA4oAkz3HIpaiOAOMj+tL06k49xQB/9X+/c5xxyaWo+CRjr64oOOTjJzigBxJzgYzS8+g/OmnBzkbiKZjG3PIoAk7npn606o/l/2j6U1sZ4oAlyME9qAc54IoHA6YpvYkHA+nSgBTgEcc06m8qO7Gl69QKAP/1v79z0PFNyM9D6UuQCQc0AAcd6AFIyCKWm5UketIcHPIz70AOPQ5HFBOBnFI2CDSdsHp9KAHdBwM01V7kYanZ4z2pu77uKAHe3P1ox3AGaBg8jk0YwSeaAP/1/79jjuOKUdOmKODkUEgDJoATPBODS8+g/Omg9Rn9KBkZLGgBcr0Pb1pDg4PH59KTjg5XH0pw4xk5oAOPvYJpcDrjFI3/Ase1JtXkk5oAX5SR606mADjH8qcCD0NAH//0P79iQTtNLn14ppGMsQCaX5fT9KADjpyMUvAPuabnrj+VLzweB60AO/CmZAXIBxRnj5f5UuRjHI5xQAHnIxzilHbA4pCR0/HpTTk4BB60AOwPQ80h+bjr9DS5HPpSHPPJ/KgD//R/v26Dp0p1Nz07D3ozgZHSgAOeeARQehyDilx6cUZ+v5UAIdvGQBS5PoaRsfexupeMe1ADflB24p2B6CkHXPOe9Jklc8MaAH03v8AdOaTAIyTz60E55DYH0oA/9L+/Yd/lxQD2CkUvy5zxmgYxx0oAQgYJxk0DoDjnFGdvU0gI449ulAClsYxzS856cUhOACRzQCAOT+lAB0HQ0uR7/lSZyByAaTcfQ/lQAvz/wCzSE+nTPNHX2z70E49x9aAP//T/v1JPYqPxpPl/h+9RnoP505efm7mgBMg4OQcUn8f8NL78FutJx7A/lQAuMnORj2ox274pP8Av5Sg4B6cdqAEOMjBAp3ylevFIATgk5FHHOfx9qADIHGfpSAAd89qXBznp6UgIBOeuaAP/9T+/TCgHkZry343fF3wb8APg58VPjp8RLq6svAXg7w9qHibWZYIjLKlnaW7zymNByz7I2wvc4r1Jc8Z4Ffnr/wVsH/Grn/gof0P/FmPGH/ppuKAPwwP/B5P/wAEuwXVfAX7TjqDgEaBb4P/AJMV9Tfsy/8AB0r/AMEl/wBpHxvovw9ufin4v+CHiLUp0trKTxrpDWVlLMxwqPdozxRZJAzIVX1Ir8ff+DOn9mn9nj43fsV/tTa58Y/gZ8Jfinrdp8UVs7S78Q+HrXUJraD+ybNvLjeZGKpuZm2jjJJr3v8A4OhP+CS37DXhX/gnr49/a4+EXwP+HnwN+NPg7VNLljvfDGnx6dFrNrcXK28ltcW8QWNz++WQPt3Ax9cEggH9mVvPb3UEN1azxXNtIgeORGDK6kZBBHBBHevw+/4KS/8ABwB+xD/wS8+OPhz9n39oCw+LWtfEHUNDt/EbR+H9JS4itLKaaWKNpHeRPmLW8vC54X3ry3/g12+N/wASvjn/AMEgvgff/E/VtT8Qan4e1TVPCmnX145eWfTbWbFupc8sI1fygT2jA7V/NF/wcXfBPwx+0p/wcefsc/s8eMzdR+GvHXg/wb4RuZYG2yW327VdTtlmU+sbSrJ77MUAf3/fsyftJ/CH9r34FfDn9oz4EeJ4PF/wt8UWIvtMvFQo2AxR45EPKSRuroynkMpFdN8b/i74T+AXwf8AiZ8bvHf9o/8ACF+FNEu9f1T7JF5sxtreIyP5acbmwpwMjmv4Tv8Ag2U/a7+JP7A37bP7QP8AwRZ/an1O50iOXxBet4NjvWKpZ65DnzYYN3SK8hVJ0A4LJkcua/sP/wCCp+f+Hbv7cXQD/hWHiD/0jkoA5v8A4Jq/8FOP2e/+Cpvwc8U/Gv8AZ1sfHOmeGdH1x9AvYNfsltrhLhYkkyoVnBUrIvOa8g/4Kff8Fpv2SP8Agk5ffCrR/wBo+z+JGs694vhu7rSrTw7pq3LLBAyLJJIXdAo3SKAMkmvxX/4MsMH/AIJ9/tEDOP8Ai50n/pBb18If8Hl2j6X4g/a1/wCCZHh/W4ZbjRr6y1Ozuo0bazwyalZo4B7EhjzQB//V/sk/YO/by/Z0/wCCjX7P+jftGfs0eKbjxB4LuLubTr21u4fIvdGvosF7W6hydkgDxuOSGV1YEg19iahfQ6bY3uo3O4W9vC88hUZIVQScD6A1/nFf8EjPjB4w/wCCEH/BZ/4p/wDBO7496zeaH+zh8StQtrTStQv3Mdp5k436RqYZvl2yLJ9lkfoGOD9zj/Rf8YHPg/xUcYP9m3P/AKKagD8xf+CZ/wDwWF/ZX/4Kqal8etL/AGbbL4jWV18PLnT7fWv7f01bUTrdtdLBJAVdtwJsZ8g4I445r9XQFO3JBNfwJf8ABkkM+Nv+Cph6fv8AwV/6O16v77gR6j6CgAyCT7UnGfvcfWl/3ttByR1H5UAJg8Z27qXkYGQaTr824flRht2floABztztHpQRu6njtS9OW25o4xkAGgD/1v79C3Oeo7U7AJ/Sg55ORj3pCcc9T0oAUkY5/lRx90HBpDgMDkCjpnp1oAQ7Rn7wpec5zz6UFRzg7aUDpg8UAIDgZZhincdM032yu2k2/N1VqAHHBO00nOBgk0AnP3loHB6qBmgD/9f+/XHPUY60Y64wBR/dX5aX/vnpQAdwMnNHHvjpTQMdHGKCePf60AOwe4XFISQcDbilO3BAIFIN3HK4oAB91TnFLgDocUHkHkUhG7GD8tAC7ge4oPQDg84qrdzNDaXMyY3pGzLnnkA/4V/Ih/wbx/8ABZb9tf8A4KPftf8A7ZXwa/ab174f6z4I8I6cdQ0CLSdAisJbNv7Rkg2GRDmRdgX7+TkZzQB//9D+/UqegOBQMBsDNIV4JJya/JP/AILH/wDBVf4f/wDBJX9leT43+IfDkXj74ka1enRPBfhk3XkLquoeWXLyuAWW3iUb3IGeVUYLA0AfrbxwwGe1BxzzjNfwJ/CD4v8A/B3T/wAFAvAejftQ/BzUfhv8GvhVrcI1Pw7pckOmaRHf2bjdG0MF0JJ3iZcFXkb5gQckGv19/wCCOP7eP/BX3xh+0/48/Yl/4Kr/ALNP/CK65pvhibxFonj+y0g2dtqIhniiaB5YN1pcbhMGDxlGG07lOcgA/pw4HQqKaDgDHPrX8bn7TfxP/wCDt2w/aG+NVj+zx8OvhDc/AmLxLfx+EJG0rQpWk0kTMLZmeZvMZjHsyX5znNfjp8d/+C23/By5+zd+1L4D/Yz+L+p/CLwv+0P4lOmro2hjwjos32o30zQ2375AUXe6MOTxjmgD/Su44IGDmn46Z5Nfwzj4uf8AB6CP+aafB7H/AGBfDn/xVftN/wAFYv27P2xv+Cfv/BF+y/ajgt/A+kftmW+leD9N8QPcWMd1p+ma1eSW0WoPHbg+W4V2uFQZKAkHkDBAP3x3Lmjkqd3y1/Bb/wAEQP8Ag5H/AG1v2gP24vhZ+yv/AMFBrnwBd+DfiXp3/FF63a6BFpEyag+77KqmLEc0M7RTQDjIlCqDnIr+9L/azxQB/9H+/XpuIxmjGdpJwapX17a6dZXeoX9zDaWNvE808sjbViRRlmJPQAAnNf5rH7cf/B1t/wAFIj+0N8bvEv7HV58N/CX7IGg+MZfCnhy6vvCtvqDaysZl8uea4mBbfcJbyTbEK7EZR1+YgH+ltxgYPFMOBn7ufpXiv7OXxC1j4ufs/fA/4p+IILO217xH4S0nXL2O3BEUc9xaRyuEBOQu52A9q9rXHzbaAHbl9RQT6EZoz15FIcYGf1FADeh3FgBS5GRtIFBBwMcnNJgHdgkmgBxx90k0ZXkcZpSCRwcU3kd/0oA//9L+/XnHX1o9W3YWlU55OKT+/wAgUAOPTrijPbIzSHGOcEijG3+LigAyvt60nHGf1oy2QDgU7qPmAFADSduOeKXIx82BSfxen9aXnjbjbQAmcEfjS/7R+Wg7sDByaTdyAMEUAf/T/v13Ad80vb5iDRwdwFKSAMmgBOmByRQD7j86Q5+XoKUkcckUAJtH975aXIOcGgsBjNIOV45+tAB8zY/hFGV4yc07Azmk57bcUABPoVozhc9aQ5A645oyOOSTQB//1P798j6UmV5PHpX5V/8ABUj/AIK8fsp/8EofhdYeMfjtrFx4j+ImrRyf8Ix4I0iVDqmvsnBfB4ht1JAadxtBOAGPFfy1eGP+C0v/AAcn/wDBQ6O6+Iv7An7DHhb4ZfA6eRjpOoT+HxfC6jDEcajqLxw3J4wWiiVcgigD++nI6NtzTSccEKa/g1n/AOC6X/BwX/wTxvrDxN/wUr/4J62HxJ+BqSIup63pGjPpclohO0t/aFmZ7SNu4WWJQx43DOR/Zh+x1+1T8Of22v2afhJ+1J8J4tZt/AXjDS11Kxg1CHy7m2+ZkeKRRkbldHUkEg4yODQB9O5HqKNyjuK/iD/4KW/8HBn/AAVA+C//AAVS8f8A/BPL9h34B/Aj4mXdjNZ2GiWOq6Ne32qaxcvaLcSHdHeQIB8zYXbwF6mvOPF//Bbv/g5h/Zz0a6+J37Q3/BLPwHd/C3Tl+06rLYeEtUi+zW68u8k0N7cGJccl2QquMmgD+8H1wVC0HkkZAFfiZ/wRx/4Ld/AD/gr34F8THwj4W1X4P/HPw3FFL4l8GaheLdmCJztW5tLlVTz7ctlclEZTgMvIJ6j/AILpf8FAfjB/wTQ/YD8WftQfAzQPAniP4hWviHStGtoPEdtNPZKlzIyu7RwyROzAJwN4HNAH7GZAOOAKXORwa/OP/gkx+138Q/27f+Cfv7Of7VPxV0bwt4f8f+LNLlutTtdFikisklS4kizEkju6giMHBZuvWv0b5PsKAP/V/v2DepWk4ySCPWv5ZPhJ/wAFuP2nfHP/AAcReO/+CT+reBvg7D+ztYXWr2NrqkFndLrge10BtSWR5zOYm3SJsK+UPlPByK/qbOTuAFADiAeKaMbieKd6bcYpCeQKAFJwO1Nwexx+FOB6cjNId3UH8KAAn6HvS/KfQ0hJ/hwaUcYBIzQAm7/d/OjKjpjikzx3/wAaXA3Z70Af/9b+/XPfKZpcj1GKDnB6UAk44AFAC5x1IoAxwKD6ZwabxkZ49qAAYH938KaAuB9386kG7vikw3979KAD7q+tJkDOQP8AGl6MOFxR167SKAGkADkYH4U0Zzzj8alHQU0MMcnP4UAf/9f+/TGDnKijDDaNuaMdDwR9KXORnK7qAEXHB45pmMk45p/IAztx0o3fN97igBSRweTSYxjBGPWg/KuO9AOSe2aABVOckU1VyfanAg9j6UBiTxjFAC9E9DX56f8ABW3n/gl1/wAFDv8AsjHjD/003FfoSPulcfNX56/8FbOf+CXX/BQ45z/xZjxh/wCmm4oA/9D8ov8Agg1/wUk/4Ka/sT/s7fGzwb+w/wD8E/1/a5+Hup+Mf7Y1rW/7O1K4/sq/FjBH9mzauqj91HHJgjPz+lenfEX9vD9vL/g4a/as+Hv/AATe/al+Kfwg/wCCfXw0GtJdaj4XurK8s5dRvojhbYJMGkuLwB2MUErxR5G7JYKD+3f/AAZL/wDJiv7XH/ZWV/8ATPZVx3/B3z+wBbxfDX4Q/wDBUL4IafJ4X+MPgjXLLRPFl9pamGa4sJGZ7PUGZcfvba5RI9/XbcDJxGBQB/Wt+xn+yT8Jf2GP2avhX+y58EdPnsfh94V08WlvJcENcX87EvNdTsAA0ssju7HplsDAAFfxH/8ABZ//AJWxf+CXn/XT4Yf+pHe1/XH/AMEg/wBuWx/4KHf8E+v2ev2k5bqzfxzeaSuleLreLGLXXbX9zdDaPuq7IJlH9yVa/kd/4LQ/8rY3/BL7/rr8MP8A1I72gD3j/g7J/wCCf/ir4e+Ifg7/AMFh/wBmFNQ8I/FrwdqenWfjTUNLXZNG0EiHTdWJUffidVgZj1Uwg8LX7AaZ+374U/4KS/8ABv8A/tA/tM6JLY2/im7+Euv6Z4t06Ej/AIlWuQWLLcxYycKSVkUH+CRa/cn41fB/4fftB/CP4jfBD4r+H7fxT8N/FejXWha1YS/duLWeMo4B6hgGyGHIIBHSv8wv4YfEf4l/8EPP2hP+Co//AASs+N+sX8vwI+IngbXbLw/d3SEQzXpspW0rUouw8+Im2kxwXC5/1YoA/ok/4Mr/APlH5+0N/wBlOk/9N9vXw9/weMEn9sr/AIJZ88eVff8Ap0sq+4f+DK84/wCCff7RHIH/ABc6T/0gt6+H/wDg8Z/5PL/4JZ/9cr//ANOtlQB+h/8AwdRf8EzJf2qf2I/B/wC2V8LvDxvPjh8IdKS5vzaQ5udS8MkB50yo3P8AZ3JuFH8KmYjqc/dH/BBL/gpnb/8ABSH/AIJsRan4w8RJqn7Q3gPTZPCnjiKWTNxcSpbt9mv2GSStxGud56yJKO1fvDpdhZ6r4O03TNTtLa/0650yOC4gmQPHNG0QVlZTwVIJBB4Ir/N5ey1v/g21/wCC63ibw1J/aOhfsQ/FuxmtbeVtxtBoV7IwjJPQvYXXyn+JY8no/IB//9Gx/wAGSP8AyO//AAVL/wCu3gr/ANHa9X99pPTOCK/gS/4MksHxr/wVLxgjzvBOCDwf32vV/ffzk5xtoAQYwc5x70A4wCeaU4wc9KAemARQA0HGcnJ9Kcc44ODSfw/3aRd3AxgUAKn3RTW5xj5qcD0AyaT72cMc0AGcbu1IRu+YYFOYnHTNJz8x6j60Af/S/v0wDzlvSgBs98+tOLL9aT+9k49KAAnO4YpCp4FOz1zxzR93+81ADV+YfebNAyDzyDQef++qdu6j5s0AR4+bnCipPmP+zSHJZe1APA49qAAcnqQBR1B42ilDDvwaMgnAbmgD/9P+/Rh91eaT1O+nblbANLnHpj1zQA3P4jFKTwMYIo57f/rp2eOhIoAQn5ecqfam9d3zfLT8j1FISQOBQBn6l/yD7/nd+4k/9BNf53v/AAZ4f8pFP+CiX/YAP/p4kr/RD1HJ06+6j9y/H/ATX+d5/wAGeGf+Hin/AAUTx1/4R8/+niSgD/RRGB8wPy1/Dz/wexfAn4m+NPgB+xp8ePDWn6pqfw58Ga7r2l+IxCpZLKTUY7I2txKB0XNjNHu6AyAd6/uG9Ry3415/8T/hf8O/jV4B8V/Cv4seDtB+IHw512yk0/V9G1S3E1tf27jDI6H27jkHBBBGaAP/1P6SP+CN3/BWb9kX/goB+zD8FtA+G3jfwf4O+N+jeGrDSfEHw+uLiO3vtKuba3SJ/ssBIM1qdm5HTPyEAgEED9rO3ynpX8Jn7en/AAZ9po+r6t8a/wDglr8cdZ+GHjCzdtS07wXr+ozII5Vyyxafq6HzYXyBs87IB6uOtZv/AAQw/wCC6v7Z3w//AGytF/4JPf8ABVMeJLzx7Lfy+FtC8ReIoQms6Nq8cbNDY6jKv/H1HNsEcdwdzFpIyXZW3AA/vFweo6e1f50P/BdQn/iKR/YPH/TT4bf+niev9GHgegr/ADn/APgup/ytIfsG/wDXT4b/APp4noA/0XO+DgZ61/NT/wAHav8Ayhb+NH/Y3eFf/TnFX9LB3DnIAr+af/g7Uz/w5a+M+f8AobfCn/pzioA/j+/a4/Zj8S+CP+CKn/BFz/gqB8JI7jRvH3gKe78PapqdmCstuV8Q315ps7MOQY545lDdjIB6V/pNfsEftT+Hf21f2PP2fP2nPDUsElv4s8N2l/dpGwItr4JsuYjjoUmSVce1fhF/wS8/ZK8L/tzf8Gv/AMJv2WfFUEDReLfAPiGz024cZ/s/VF1a+ksrpfeK4jhf3CkHgmvgz/gz4/a78UeGLX9qj/glx8YpZtN8beBNbufEnh20umxJBGZvs2pWag9o544pQB/z3kPQUAfs/wD8HIX7dMf7Dn/BMX4uXuha0um/FT4g/wDFAeGESTbMXuo2+1TL3xHbLOcjoWQfxV/Dv/wUY/YUn/Yo/wCCCP8AwTUvfFeinSviz8SviFffEDxAsqbZ4I7rSS1nbyA8hktvKJU8hpHHav1y/wCCoOq3f/BZb/g4p/Zr/wCCfnh+ebXvgJ8H5z/wkyQktbq8ZjvNVkkI45EVraZ9RtHJr6J/4PY7O0079kD9h/TrC3itLG38f30MMSKFWKNdLZVUAdAAAMUAf//V/tP/AGHMf8MX/snk9P8AhXHhzj/uHQV9U18r/sOf8mYfsnf9k58Of+m6CvqTOQeMfjQAoz8396nY4xSfgB75oPc8g4oAT7q8gGl7bvlzRnI+71pcnGcc+lABjGcYFAHAzjNGeSOmKbngbhQA7HUjGabtOWPSjIAAI/Cl7njHvQB//9b+/UgnPQd/pRjgcbqUgYPApPm6lQaAF6424xSAYb04pTkgY4NDNjIHWgAwcc8+1Jg4xwfSlOccgE0nbPy9eKAFOfbFIQTwfWnYHXHNBYA4NACLu5yc0dsDkdDS8496TODjAAoA/9f+/QDrzgGlwR91uKd06Ckz36j+VACA55IH1peucHinUhyOgzQA0Y4xgE0pHrg/hSADpzxTh64waAGkYJYgGq11c29lbXF5dSpDaxRtJK7cBEAJJPsADVyvK/jfaarqHwW+L1joZZNbn8L6rDZkdRO1pIEI6/xEUAf5xP7G/wAO3/4OIf8Ag4E+NPxa+OqXniv9lrwJPe6xHpdyxa3k0Kzu/s2l6cU6KLhmE0ijggTcnv8A6V+iaLo3hnRtM8P+H9K07QtBsYEtbOytIVhgtIUUKsccagKqKAAABgAV/n0f8GRGs6Fp3xW/4KE+DtTEUHjmbRfDN3BHJxL9mhub6Occ84Ek9vn3xX9h3/BUb4c/8FEfif8As02vhz/gmT8afBHwJ/aKXxHZXM+ta9BBLBLpCpMJ4E8+2uYxIztA2TH0RgCM0Af/0P74NY0fSfEWlajoevaZp+taLeQvb3dndwrNDdRMCGjkjYFWUgkEEEEGuU+GXwu+HXwW8D6F8NPhL4I8M/Dr4f6Wjx6douj2iWtnYozs5WKJAFQFmY4A6mv4Ev8AgoX43/4Oqf8Agmn+zpf/ALT3x/8A+Cifwr1T4e2+sWWiyReH9G0O5vDcXJYRkRyaNGu35Dk7uOOK/rk/4Iu/tI/F39rj/gmR+yh+0H8dvEcXi/4r+ItBlm1nU47SK2+3TR3U0XmGKFVjUlY1yEUDOcAUAfx5eMv+VzTQ+3/FWWn/AKYxX+hr4q1Dw3pPhnX9T8Y3Wk2PhKCzml1Ka+dVto7UITIZS3y7Nu7OeMV/mI/8FHfg/wDHv4+f8HRXxB+E/wCzF8Z7n9nv45axr1jBoPjCG5ngfRZhpCMZBJB+8XKqy/Lz81dd/wAFif8Agm1/wXs/Za+AE/xH/aa/bl+LP7bH7KNvPEviuLRPG2qONIhZwA97Y3O3dASQvmKJFVipYKOaAPTv+DZrQIPEP/BfT9sjxt+z9buv7NdjZeNPLls0ItBps+qqNOTj5QCFDL7IcV/Q9/wdz7v+HOnjjBJH/CdeHP8A0dJXr/8AwbYaP/wTuuP+Ce3hrx9+wN4EvvBtzqV19l+Iaa3dC715PEEMaiSK9uMANEAyvEECx7JAQqsXFeQf8Hc3/KHTxx/2PXhz/wBHS0AfWP8Awbe5/wCHMX7E/TH9h3f/AKXT1+4h3dwpr8Ov+Db4/wDGmL9ifA5/sO7/APS6ev3G/AZ7UAf5537OH/K7J8UemP7b8Uf+oTLX+hj/AN8hq/zzf2cAD/wex/FIjkf234o/9Qqav9DPse1AH//R/v04xjA6U75/9mlzyPek6YIAxQAmOTjaaMfLjhqcO/rTcn1SgBdvb+H0pMHdngUoPOCMUg3DcTQAhUnnOaBwp6EU8+4zSA9fQUAL6jimBfuktSg7uwxQD0GM8UAf/9L+/bB7YFJg552k07AznHNJk7RtFACFB1zik/790/HGDzSYb+9+lACkHjBxUZ2j7pwadkevWnY6kdaAGFt2Bg9acB06ZFNJAJyM0AZ68DtQAh3cBjgUq4HBABox3I/Km8f7WO9AH//T/vz5AI7etJ8232qQ7d3TJpuSRnpjpQAq7cnAP1pPurgrQdoAyDS8EcDH4UAJwze386OnzL0pAQOikml554P/AH1QAEHkYyfWvlD9ur4KeLf2kP2Lf2rv2f8AwJcaVa+NfG3w81/wvpMl7KY7dLy7sJoIjK4BKpvkXJAOBng19X89di4oCr6N+NAH86v/AAbe/wDBLn9pT/glh+y18b/hV+07ceAH8Z+JfHZ8RWcfh3UmvYYrQWFvbjfI0afOWhc4A4GOa/ZP9sj9mzwr+2B+y58c/wBmjxkkB0Txl4cvNGMsiBhazPGfKmAPdJBG/wDwGvpfkjptHelJbAwMGgD/1P1+/wCDcL/gmH+3x/wS28M/tK/B79qfUvhbqPwk1nUrPWfCa+H9ckvpI70B47hnjaJBErxrbnqTkGuJ/wCCiP8AwRp/a0/ah/4LrfsRf8FD/hve/C+L9n/wGfCD+IBqGrPDqSHS9WubyYQ24jIk3JMgU7hznNf1XAA4IGKCB7j6UAN5/iUk1/NB/wAHD/8AwQu1T/gq74F+HHxN+AV74O8K/tXeEPM0+2uNXkNvaeItGkJY2dxMqsUaKT95G5BA8yUY+YEf0vrkAgDmnnPTGRQB/Pr/AMG6v/BMX9oH/glt+yD8QvhH+0nf+Brj4g6/4ym8QCDw/fteW9rb/Z4oUBmZE3MfLJwBxkV8yf8ABwV/wRy/a1/4KZftC/sN/E79nK7+GEPhzwE9xF4hGvas9nKiSXttOHhVY38zCwvxkc4r+qIdOAeB0pWxkZxQBm6RayWGk6XYyMpkgt4omIPBKqAcflX4sf8ABdj/AIJKaX/wVm/ZNtvh54ZvPD3hj9oDwtfnWfBGt6gu2KOVl2T2c0igssEyhc4Bw0cbYO2v26OcjjIpP7zfw0Afyvf8G0P/AARu/ar/AOCUFj+13qf7Ut/8NG1bx4/h2DSbTw5qj34iisBfl5JZDGgBY36gKAfuE96/qiGMDFIOAM9fbvQOgAagD//V/v1/hbjbTsevNISAcEcUK2R6mgBBkqMcGghs5DUZPOOeaB0GBmgBCeMYGelOyMckGm9TuK8UHvheKAA4IGAP8KP+/dOJ65yKPvAggigBhxkZ249qdlRleRSn2XJpASBkgfhQB//W/v0PAz/FS8Z/gxTs+xpvORxxQAp298ZppXOM7Vp2cqSM03qT/CO9AByV4C0D0OMdaUE5XggUnQZ5z9KAFwoGSMUArtPHy/zpOqk/MaXPBCkk0AI3bH5UAcZyM+3Sk9Op/wAad0b2NAH/1/79fkODwKXKjHHWk5z77adkZx3oAj5xnCnvS8AjaAT9aN3H3cLRjggDPNAFPUR/xLr/AKD9w/8A6Ca/zWv+DVX9ob4Afs7/ALfv7feu/tAfHP4O/AvQ7/R3tbG98ZeJrLRYL2ZdWkYxwyXckau4HJVSSBzX+lc6JIjxSfNGylT7g8Yr+Xnxv/waI/8ABJHxv4w8V+Mrpf2nNFutW1G41KW0sPGUK21s8shdkiElo7hAWOAzMQO5oA/aMf8ABT7/AIJqdT/wUN/YZz7fFrQP/kuvmz9p7/gt3+wH+x7+0h8Gf2evj38UH8JWnjzwrF4s0DxtHELvw1LayzPHDvvoS4VZPLLiYAxbSpLAHNfmUf8Agzm/4JE/9BH9rQf9zpa//IVfql+0b/wRY/4J5/tX/s9fBr9nD44fBmXxb4Y+H/h228MeENbTUJbbXdGs4YkjUJfxbWYny1ZlcNGWySnOKAPref8Abh/Ywh8CSfE5v2tP2bX+HaW5u21uPxvpr2XlAZ3CZZip49Dmv8+3W/Hnhv8A4LM/8HSHwi+Kv7HOj6lrXwb8J69oOoal4qgtXt4bvTtCQSz6lKxAKRzPEtvFvwZA0Qx82K/aRf8Agyy/4JyjxSNXb4//ALWx8NeZu/sg6hpfK/3fP+x7/wAa/oc/YS/4Jn/sZ/8ABNvwReeB/wBk74Raf4FF8qDVdXup5L3VdZZejXN5KS7DPOwbUBPyqvSgD//Q/v0KnAz81f50P/BdT/laR/YP/wCunw2/9PE9f6L55PYj+Vfkv+0x/wAEXv2KP2sP22fhD+338V9P+J7/AB68Gf2WdM/s3Xjb6bcmwnae2NxbeWxba7tnY6bhwc0AfrOdwHRcV/NR/wAHaZz/AMEWvjQc/wDM2+FOP+4nFX9K/G7bgYr47/br/Yd+Bv8AwUR/Zy8WfsuftFW/iy4+GOr3dlfTnRNR+xXkM9tOs0Txy7WAwyDIZWBBPFAH57/8G14/40ofsNeh0fVv/TzfV/Jd/wAFmZPiV/wQ2/4Lx+Ff+Cg3wV8LQ3vgD4h6bdeJ7fT2JistRu5IWtdSsZCuMYla3usf9Nk96/0CP2Q/2UvhL+xJ+zp8Mf2Xfgbb6/a/C/wlaS2elLql6bu7ZZJ5J3aWbC7mMk0h4AAzgADivnP/AIKQ/wDBK/8AZO/4Ko/D3wP8OP2p9I8aS2PhvVJNW0bUfD2piwvrKV4/LkUStHIrRuoXchUglVPBANAH86//AAaEfsk+Ibv4cftK/wDBTn41pda/8Yfipr1xpWk6tfLunfT45zPe3Ck9PtF24zjtbL2rlv8Ag9z/AOTTf2Kz6/ETUf8A02vX9gP7OH7PPws/ZP8Agd8NP2dfgpoMnhv4XeE9Mj0nSLSSZppVhXktJI3MkjMzMzHklia+Tv8AgpD/AMEsf2WP+CqHw/8Ah58N/wBqWH4hvoPhjWX1zSpPDmsf2fOs7xGJg7GOQOhQkYxkdQRQB79+w5/yZh+ycOD/AMW58Of+m6CvqTI+b7v+NcX8O/Anh74W+AfBXwz8IwXNr4V8P6VaaLpsc0plkS2t4lijDueWIVFyx612mMd//HaAP//R/v0P8Py54pc4xjP0pwGB3NNxlf8ACgB3DD1FN6jcdy0KMDpzQenBI7fSgAz2+XdQABggZpecnP3aTHByADQAmQeAMn3p2QDjn6UY6An8+9L78A0AGeuRimZUchTTuQeB8tJnjsPWgD//0v79s+xIpOFzndQTx0PNLzgHOaAEzkKSDn2ozxgqx/ClwM9SPx601AfcUAOb5VoDL0FH1GBim4bIOeaAFY46qDR/CPu/jTjxjnvTMcYwetADl6YyDRkKBnik/hGQWpfUlsigD//T/v1OOe39aVsnIx+NIM/5FHJH8WfyoAXavTGKPTI7elIOeR8ppVGB6GgBCQcqeKayrIGSRVdCCCDyCPpT93JBxSL/ABDJNAH+Y18b7L4qf8G0v/BdrVf2h7LwTrupfsk+PL/U7u1W0iIttU8N6hcCa5sIW4T7RZy+WVQnIEcRPD5P+hJ+yx+3x+yD+2h8NtA+Kv7OXx9+HHxC8O38CyNbw6nFHf6e5HMN3Zuwmt5VOQUkUHuMggnsf2o/2SP2c/20fhbqfwY/ab+FXhr4s/D26Jc2eoxnfay4wJbedCJIJQCcPGyt71/KR8W/+DL79mW/8Vaj4g/Zm/bU+P8A+zvp9xIWXTrzTIdcS2QnPlxzJNaSlRzjezn1JoA92/4O7vjV8Hpf+CV+pfC2D4qfDu4+JV/480C6stBi1m3fUbqGJ5jLIlsHMhRARubGBkZPNfot/wAG43/KF/8AYcHIP/CP3f8A6cLmvyn+Av8AwZqfsTeEvFFl4r/ae/aP+Pv7U9xEytJYMI9Bs7vH8Mxjea5Kn0WdT71/WF8Gfg38M/2fPhf4K+DPwa8HaV4B+GPh2yTTdH0ixUiGygXoq5JJOSSSSSSSSSaAP//U8e8ZH/js10Inn/irLT/0xiv9A74vfC/wn8bfhZ8QvhF470231bwf4l0a70XUbeZA6yQTxNG3ByMgNke4r8kNS/4IXfs16p/wVNtP+Cr9x8TPjIPjJDdR3w8OCa0/sY3C2n2YP/qfPxs52+Z1/Kv23ByfmI4oA/zqv+DcT4reJ/8AgmZ/wWN/av8A+CUvxk1CfR9G8S6tqOi6Ot25SGTV9OMs1pNHnjF1Z7ijfx5iA5YV/U5/wcO/sp+Nf2vf+CTn7THw8+G2j3fiPx7o1tb+MtM022jMk+oNp8nnyQwoOXlaETbVHLNgDJIrD/bD/wCCCH7MP7Xn7fXwo/4KJ3/xN+M/wf8Ajj4YuNGvJIvCs9pFa61cadMJIJbjzYXcSFVSFmVhuRFHbNfuowBzkbh0wR1oA/jK/wCDUX/gqr+z34l/Y90L9gX4s/Ebw18Nfj74Fvr1dD0/W7tLIeI9JmmaVfsryFVknieSRGhB3gBWAIyR/Ur+0l+2Z+zN+yZ8KPFfxm+Ovxk8A+BvBekWUl5I11qkInvCoJWG2h3b5pXOFVEBJJFfhV+3/wD8GrP7AH7Z/wAQtZ+NPw417x9+yJ8W9SuHvNRu/CccVxpeoXLEkzvp0u0RyEnJMMkQPUjOTXxD8L/+DMD4A23inSdW/aU/bs+Pv7QPhm0lDjR7LRotEWdR/wAs3nkubx1U9CY9h9CKAPiX/g3H8MePP+ChH/Bbn9tP/gq/q/hzUdH+Gdpd+IbrSppISsaXeqyNBa2gfozw2G8PjuVJxmv9C87e+K+ef2Yv2VvgD+xv8I/DvwM/Zt+Gnh/4W/DPTF/cafYRkmaQgBpp5WJeaZsDdI5LHua+hQDj29KADcB94YNOwBjAFGM4zkUtAH//1f79Q3PTC0oYE+9Jtycltwp2Oc4FAB1P0pDt/OkJOQAQKVs4PQigBPQjaBTuFHoK+aP2bv2i7b9om3+L1zZeDtV8K2/hHxtqfgmSWe5ini1K5smVJpYHj42B2KEHkMrDtX0qB8uT9aAFy2enFNPUZzu9qoanqVhounajrGrXtrp2l2sD3NzcTOEjgiRdzO7HgKACST0Arg/hD8Xvht8evhx4U+Lvwh8YaP48+HGuWwvNK1awk8yC8iyRuU/VSMdcg0Aem5+brxik4OQAMinYPqabwykLQB//1v79gO+3DUmwe9Lz0LfNQAMDr+dABj5RwGpG4ycc+tJuT+7+lAJIbI3UAB9BwM0Bsd9xpTnG4L81JubGRtxQAYAGef8AGjqCP4qU/wCyVApu0/3RQANnoTk0pGCNxyKDnjoooOMc7VNAH//X/v0+ZV6rikyOOSOKXbuG7oaBuOD0FAC9G/Ck5BB578UvI44I7UhDH7xAFADTjj71OAGMsMUYAI+b5qX8s4oAbzgcn2peu7J/PtQAcEjkmjaSOSOKAFwf4iMU+mEZ29xTj14xuoA//9D+/QD170ucgHJxn86Ubv4vlpTnI9KAGLwDu6UEqMqMilIOAF6U7GBgcUAN546D2oORnJz+FAVe3NLsX0oAQjlOaM9DyPT0oH3idrCjDD7uMUABbI+6SK+Pv25vD3xc8V/s8694b+DieI5NbvdX0eDV00abytQfQjfw/wBpLaPlSJmtfPUYIbk7TnFfYfqMcU3bjksaAP/R/pM+GviX4z/s2/F7S/B/gP4X+GvDfhr4ufE24bQfC/iXWZbI+GvC2maOgur5I40mEc9xOkTBCesybhuclf1E/aB+OHgP9mn4JfFD4+/FLV7fQvAPhPRbjW9TuJGACxxITtUnqzHaijuzAd66vxZ8M/h3471Xwjr3jbwL4R8W61oF21/ol3qWnxXM2kXJXaZrZ3UmJyvBZcHFfwwf8Hgf7fHxI16Lwd/wTm+CeleNL/w3GsHin4k3mm2M7wzEDfZ2EkiKVKAEXDDPURZ6UAf08/8ABIL/AIKp/C//AIKwfszTfGjwbYQ+EPHekatcaL4p8NGXfJpU6sWhkXJyYpYTG6t6716rX6v/AN35q/yE/wDg3h/b1+LX/BPb9tfwr41Hg34ieKP2Z/GdxF4V8erpmmXNzDbRM2Ir0eWjL5ltI6yEdTGXHcV/rv2F7aapY2epWUyXNncRLNDIvR0YAgj6gigC+d3OMU3p3Hp1pQD359qUA9yelAATt5yTTRyc5+XrS7W4y1Lg4PPNACMcZ6c0c8bcbaCu4gjGKO4PXigD/9L+/T5snHTNHZuTnNKAc89P50AELjvQAhJz1J9KdlTgEg01UIIPFOAOMNzQA0fKGycGg4yTuxSkZyQTSYPU8+1ADcnIJzinc4YZzSkchcHbSkcYx9PagBpb7w6GnZAxgjFNIPOSDxS9wOSMUAf/0/79AerHig9O+fr0owQdxpcE/wAdACHqMk4/nSjjAyMdqQKc5zg0EEY5O2gBfl4fpSk992BSFcnIODTWz8uSCaAFA6Et70fdbtigY6k4NKRjBJPFAAfbdj2pVJPJximlWY+gpdnqSaAP/9T+/Xr6gj8abk5Pyk/UUpXpk5FKR97Hyn1oACuWB+WmHgEd81IQPcfSmnocdPrQA75QSTjNNHToOtL/AHsbs0AYwx4oAXtndTT35AHWlwcd8/Wm8kcHcKAH5BB9KUnHQE0m3AO3rSkcYHFAH//V/o6/am/4KCXv7MH7eniy31y3+M3jb4FeEvhXpmoeK9G8O2ttPaabqGp60lta3sxmeMh1RCojRmdxLlUOK/bCCaOeOKQHO5FfDcHB6ZHavzP/AGh/+Cbfhz486j+0Lq0XxS8UeEPEXxB8UeCfE11dx2UV1HYDw61s9tZCKQ7ZIJJLVpWVv4pD6Vz/AOyF8PPiroX7ZH7b/jn4uXX7QVy0+q6bpPhu+1iVF8PXXh600uzEUtuBhTNJeXGrSFY1xGAd2CVyRV1brr/X42+QTf2vT+vwv8z9W8n0xn3oDcgbcV/O3/wTr/4Lr/Df9t//AIKXftofsR2s+i23hzwzOjfDPU4m58SRWYaHUwT/ABESBZY8dYy/93n+ibIHcUANJyOgJpeSSCOKXqO4o6nkDFADQcgkAZox0zwPrTiARg0YGMdqAGZzkZGfWnd+eD9aM9ADz70cYzgZoA//1v79s8ZGCPrQrZ7GgdOMfhS44x1+tACZ9du2jd82KGXdigjPp70AA9SB9aMjOcjFJgAEHpml4A4HP0oAUcegFISBtyOaOcjp70pA9AaAEPRsYozkc4H1pMdSNuKUjjAwKAP/1/79sgY96UdeuTR6U0jGPrQApJ2k9DSg5ANLTRgdz+NAATxnAIpecdMmkI3DnigAA5FAC5H0pOozgGk6ll7UEY2/WgBSeOOfpRxwCAKP9k/NSEbgD0NAH//Q/v365GRScenPXFKowMUhBIIzQAddrdKXAHPApMk56UfL2Ck0AIOx4Pvml+b0GPSlGMfLik7HbigBMkk8cfzp2eQKWkwOgxQA08MMLzTjjv0oB4+bANIRjkHbQB//0f79ie+Rtrzv4r+N4vhp8L/iL8RZrW6v00LQ77VzBAheSfyYGkCIoySx24AHc16KTjuBTRyfmVaAP58P2Xfh342+AXi7/gnF4Sv5/Huj+KNV8N+JPij8S5ElnZfEOtau8ZOkmInyz5d1q81yVIBjSy3E8c/qv+yh+0J4s/aLg+NfibVPBB8J+CNJ8cal4a8K3B2+Zq1nYlbaeaUiRwX+2Q3yDaFXYidTk19cnDKR26V5H8Jfgp4B+COjXPh/4f6ffabo8lxPOsE11JMsBluJbh1jDE7VMtxM/qS5yTxQB/Lh/wAHY/8AwVJX9lP9la2/Yv8AhV4mFl8cfipZyR6x9llxPo/hnJSZ2IOUNwwaFfVRLX5hf8Gf3/BVc6Dr3iT/AIJlfGXXv+JRqLS698Mrq5l4t7r5mvNNyT0kG2aMD+JZR/EK++f+C3v/AAbx/CT4z+H/ANuL/gpd8R/2n/jj4j+K2keDdX8Vafo0kFoNOtI7CykktrCP5d626iMLwc8sepr83/8Aggj/AMG8/wAHv2r/ANlv9nT/AIKKR/tL/G74RfGu28S3l9YR6DDamCynsL5kiZTIpY5EY3AnByR0oA/0TweBnAPWkycYADcevWqtpHPDa28NxcNdTpGqvJt2+YwABbA6ZOTirfQHoKAAY5zjGaMH0/WlxxjijavoKAP/0v79eOAOf6UvGM496F+71FIAPQD1oAQnPRsCjoM8n1obC4xwaOc8EcUAN+X0P1pVzzjpTd3oc14r8aP2hvhD+z7oSa/8V/GmmeGYJMra2zEyXV4392GBAXc8joMDuRWNWrGEXObsl3JlJRV3oj2wYz0OMUoxnheRX5tp/wAFA9b8SA3Pwx/ZK/aE8a6Qx/c309lFp8U49V85gcU//htr47DkfsM/F/P/AGFbH/4uvGfE2BTt7T8H/kcjzOh/Mfo/t4IBPFAXIwCa/OL/AIbb+O3/AEY18X//AAa2P/xdJ/w218dsY/4Ya+MB/wC4rY//ABdP/WfA/wDPz8H/AJEf2nQ/m/M//9P+/DGepJ9KdtOf1r84T+218dj/AM2NfGAf9xWx/wDi6Q/ts/HbOR+w18Xx/wBxWx/+Lrwf9ZsD/wA/Pwf+Rwf2nQ/m/M/R7HuD+NGOmeK/OL/htv474/5Ma+L+f+wrY/8AxdH/AA218dcY/wCGGfi/j/sK2P8A8XR/rNgf+fn4P/IP7Tofzfmfo8Vxx+VG0jIHNfnF/wANt/Hb/oxr4v8A/g1sf/i6P+G2/jt/0Y18X/8Awa2P/wAXR/rNgf8An5+D/wAg/tOh/N+Z+juzjsPrSlTx9a/OAfttfHUdP2Gfi/8A+DWx/wDi6X/htv47f9GNfF//AMGtj/8AF0f6z4H/AJ+fg/8AIX9p0P5vzP0eKH2pNuACTivzhH7bXx2HT9hr4wf+DWx/+Lo/4ba+O2c/8MNfGDP/AGFbH/4uj/WfA/8APz8H/kH9p0P5vzP/1P78yuBgnFJtPHHFfnD/AMNs/HbO7/hhr4wZ/wCwrY//ABdB/bb+OxBB/Ya+L+P+wrY//F14P+s2B/5+fg/8jz/7Tofzfmfo9txxnPegrwckZr84T+218dj/AM2N/GAf9xWx/wDi6P8Ahtr47Yx/ww18YMf9hWx/+Lo/1mwP/Pz8H/kH9p0P5vzP0fCcggjFGzHSvzh/4bb+O2c/8MNfGD/wa2P/AMXSf8NtfHY8H9hn4v4/7Ctj/wDF0f6z4H/n5+D/AMg/tOh/N+Z+j23j/PpRsbHWvzhP7bXx2Ix/ww18X/8Awa2P/wAXQf22vjsRj/hhr4v/APg1sf8A4ul/rNgf+fn4P/IP7Tofzfmfo9gckjIzXifx88KeG9Z+DPxjTUdB0u8afwvqiSu8Cl3BtJBy+M/jmvkw/ttfHbqf2GvjAP8AuK2P/wAXXDfFH9s345X/AMM/iHYS/sQ/Fy1jm0K/iaVtUsSIwbdwWID9BnNC4mwP/Pz8H/kV/alD+b8z/9X9Jv8Agz/sNNuv+Cbvxgt7vT7C5W1+MuvQwmSFWKJ9msjgEjPU1/WWFVVCqoVQMADgfSv4e/8Ag1o/ab8afB79hf44aPYfs5/FP4n+HW+L2s3VxqWhNDL9lkazs8wmEsHZgACSBj5hX9YHwa/bm+A3xk8Q/wDCD2+rax8P/iOOD4e8TWT6deO3pGJAFkPspJ9q87+1sM6roqa5l02OZYuk5cnNqfZJA5wBmmkAEcZ4p2Sw+U03+9x8teidIbQ2OCKAFIzjFL3GMnvR97GOfegBGx83y80oUYBxmk3c8bmpTgkNuxQAgHoPzp2OD8q5pMbTnOBS9O/U0Af/1v79ABuIwKQDp9aXJzjp35p3oQR/jQAwjGTjjPpTmA29ApoHAznK0m7cdvQUAAAwDwDSkLgnGaTvyNoxSgjnLA9qAEKg54GaUBeOADXO6l4s8L6L/wAhjxLoOlDOMXN3HFz6fMRWN/ws74bnBHj/AMFt/wBxWD/4qsHVgnZtCckt2d0QMHoDSFRgkE1w/wDws74bf9FA8GY/7CsH/wAVR/ws/wCG+M/8LB8Fgf8AYUg/+KpfWKf8y+8XOj//1/78wMMOT+VOG3oMVwv/AAs34bj/AJn/AMFY6n/iaQ//ABdB+J3w2/6KB4KI7/8AE1h/+Lrn9vT/AJl94udHckgN70bcjturhv8AhZvw2/6H/wAFf+DSH/4uj/hZ/wANsjPxA8FZ99Ug/wDiqPrFP+ZfeLnR3QUcEZppA5HT3riB8Tvhv/0UDwZ/4NIP/iqX/hZ3w2/6H/wX7/8AE0g/+Ko+sU/5l94c6O36YAGePzowcjBIH06Vw/8Aws/4b5/5KB4K/wDBpB/8XSD4n/DYkgfEDwVnv/xNIP8A4uj29P8AmX3hzo7kY9j+FGD3CgVw3/Czvht/0UDwYfb+1IP/AIqj/hZ3w3/6H/wV/wCDSH/4uj6xT/mX3hzo/9D+/EZyMHmntx2WuG/4Wd8Nv+h/8FEf9hSD/wCKo/4Wd8NscfEDwUR7apD/APF1z/WKf8y+8nnR3QXk5ApNuSuB8tcOPif8Nu3xA8E/+DSD/wCLpP8AhZ/w3HX4geCh/wBxSD/4uj6xT/mX3hzo7nqD9e1NIHOB3rhz8UPhso5+IPgpfrqsH/xVL/ws74bj/mf/AAV/4NIf/i6Pb0/5l94c6O6TkZyaMA/KWzXDf8LO+G3f4geCv/BpD/8AF0g+J3w37/EDwVn/ALCkP/xdHt6f8y+8fOjuzg45HWvyk/4LN+Mf2wPD37BXxg8L/sLfCTx58WP2i/FdsfDOlL4fVBPocFwpS4vizugUpFvCEHIdkPav0g/4Wd8Nc4/4WB4Jz6f2rB0/76prfEz4aNjd4+8FHA/6CkH/AMVR9Yp/zL7w50f/0f5l/wDgnN+zH/wU9+DP/BQa21z9mn9mn4oeMv2kfgvr1pqfjLw1ZtELmxgaXZLbXRMgG2ZPMjOCfvZr/Yw8DeIbzxf4L8JeKtQ8Oa54N1DUtMtr640jU4wl3pcskSu1vOoJAljLFGAJGVOCetfyIf8ABGvxN4Xtv+C/v/Bc2+ufEegwWF1JpotZpLuNY7g/a2/1bFsN+Ga/sQs7m2ulaS2uIbiLPytG4YH8RU88XomJMunGRzg0uOSeuaa3QfNijAJbH3qoYEDB78U4HIFNPyjPJ/GjhQMEf40AIQPpz1pScYVetD/dpx6dvxoAacHHBP8ASjr0HenL06YpBtzxigD/0v7+KaCD0peOOntSdMdx65oAXA9BTT2Hf6UYwTk9aUjI5xmgBO4J9KXggdhSYXg8Yqpe3lpp1rcX9/c29lZQoZJZpnCJGgGSzMeAB6mgC2OnbbR94ckivzs8Xf8ABSH4O23iC98IfB3wp8Rf2iPE1u/lzL4U055rSJ/RrpsIQP7wyPesQftp/tI3GZbP9hf4jfZj937Rr9jG/wCKluK8KvxFgoS5XUV/K7/I4p5jQi7OR+lgI4yM8Uu5T/CK/NL/AIbL/ad/6MY8df8AhS2H/wAVS/8ADZf7Tn/Ri3jn/wAKWx/+Kqf9acB/P+D/AMiP7Uofzfgz/9P+/QuCOmaTKdcCvzV/4bK/ae/6MY8cf+FLYf8AxVJ/w2X+05/0Yt45/wDClsf/AIqvA/1pwH8/4P8AyPP/ALUofzfgz9LN4A7mjcp5Ir80z+2Z+06en7C/jkf9zJYf/FUn/DZX7Tn/AEYv44/8KSw/+Ko/1pwH8/4P/IP7Uofzfgz9LCV54yaCQecYNfmp/wANl/tOf9GLeOf/AApbH/4qgftl/tOD/mxbxx/4Ulh/8VR/rTgP5/wf+Qf2pQ/m/Bn6WZXk4yaN69MCvzT/AOGy/wBpzt+wt44H/cyWH/xVA/bL/acAx/wwt45/8KSw/wDiqP8AWnAfz/g/8g/tSh/N+DP0s3896QsD1zivzU/4bM/ac/6MW8cf+FJYf/FUH9sv9pw/82LeOP8AwpLD/wCKo/1pwH8/4P8AyD+1KH834M//1P78yy9lH5UbhjHf2r81P+Gy/wBp3/oxfxz/AOFJYf8AxVH/AA2X+05/0Yr43/8ACksP/iq8D/WnAfz/AIP/ACPP/tSh/N+DP0sLA5GKTcozha/NP/hsv9pzt+wv45A/7GWw/wDiqX/hsv8Aaczn/hhbxx/4Ulh/8VR/rTgP5/wf+Qf2pQ/m/Bn6Wbl9OKNy/wB0V+aZ/bL/AGnD/wA2LeOP/CksP/iqX/hsz9p3/oxbxx/4Ulh/8VR/rTgP5/wf+Qf2pQ/m/Bn6VEg46YpQy88da/NL/hsr9pz/AKMX8cf+FJYf/FUo/bM/adHX9hfxyf8AuZLD/wCKo/1pwH8/4P8AyD+1KH834M/S7CkdOPrS49s1+Z//AA2b+073/YY8df8AhS2H/wAVUL/8FEtV8GFZfjf+yx8cvhnpAbEuqQWyanaQD1d4CcClHifAt29p+D/yKjmlB/aP/9X+/UEHgDjFN/d15T8JfjX8Lvjn4Yg8X/Cvxno3i/RWwHa3k/eW7/3Jomw8bD+6wBr1ccj5QMVnTqRnFSi7pijJNXWx+ef/AAVtx/w69/4KD9/+LO+LO3/UMnr82v8Ag1QGf+CLf7Pn/Yb8R/8Apxlr9Jv+Ctgx/wAEvP8AgoPjp/wp7xZ/6bJ6/Nj/AINUMf8ADlz9nz1/tvxF/wCnKWtBn9GoA7AU3GQflApT1HIHtSJ92gBxIHU00bcD7tGQ38WKcOgoAQKB0pNuepJoyT/dIoOcnCgigD//1v79DwM4yfenfNt/2qaW4+ZaMdSB+dAHzv8AtP8Ax4079nb4Q698QbizfVtcZ49O0TT0HzajqUx2QRD2Lcn/AGQa+Nvgj+zmbO9X4yfHWZfiL8ftVH2q7vrweZDogb5ltbKJsiJEzjI5JzzXRftwxJrnxz/Yk8J6iPP0OXxXealLCeUkmt7RniJHfDAGvpkdDnr6Zr8r4vxs54n2N/djb73rc+WzivKVXk6IOd3tSgnGW4pPQ/eo5zjHFfKnkgWApWzg460nvwT2o4zjHvQA6kIBz1oJxznijueMe9AH/9f+2CimZz93/wDVQeNoxkY9K/AT4AU9OhNGMEfShTkZNL35H40AJjP3gM06oxnqev0p2QcigBSM9yKOxC0cgd2pMnHYtQA6m4Hv60HHGBk9qCCDxjr6UAf/0P7X+g5NJ0Hc0vrjrS1+AnwA3GMEHbS9Mnmk2n1GPTFGTnpn+lACggjjpQR164pvXkgZ+lA5xkAHrQA7HTk1xXxJG/4d+P0POdEvh/5LvXa+vP8A9auQ+IK7/AfjhPlOdHvR/wCQHpx3KjufzG/8Glsm79hn9pWDPEfxu1pf/JGwNf0jfFz4HfDf426G2iePNBivZFGbTUICI73TpOokgmA3I6kAjtkdDX81/wDwaXkr+xd+1dCOCnxy1j/032H+Ff1V9QpwM16+f/75U9Tpx38aR//R/rg/Y/8Ai5430Dxr4s/ZN+NWuT+I/HOg2g1Pw5rk4PmeItELbQ8h7zREqjHvnPY1+iQwB0/Kvyi+LKL4f/bL/Ys8Y2i+Rd3l1rPh65dODNBJbq6o3qoYMfxr9XgR0I5718vwrjp18L+8d3F2/I8vKa7nStLdaDj128YpACAcdcUvYAgUAA7s+tfUHqBleTxSfdwMZNOPc9xRgBcHGKAEB4+7kUhK9iB+FO4U+5oAAyOMUAV5ZY4Y3mlIWJFLsT0AA5P6Vyfw/wDH/gv4p+CfCXxH+HniPS/F3gbXdOg1bSNUs33wahZyoHjmjbujKwIPoa66WOORXSRFZSCGUjIYV8R/CP8Aa5+Alzrnxr+Gek6FN8Gvh98KNSi8J3GrapZRaT4eDpFBiCzmJWJUj86KMJx2wMYpK3z/AK/4Anf5H//S/v1xkKOMUHqMDj6V8b/GH9tv4PfDb4LfEP43eB762/aC0Dw1Zy3mp2ngjUrO/ltkSNpGaV/NEcSBI2JLNnoACSAfqvwzrdv4l8P6F4itY3htr60hvEQkEoJEDbTjuN1AGL488d+Dvhl4O8S+PvHviHTvC3hDSLSS+1C/u5AkdtCgyzEn6YAHJJwOTX5dQ63+0v8AtyTxeJbLxZ4s/ZX/AGT5x5mm2mmRCDxb4ygP3bia4cEafbOMFY0UysrZLLxjpf2pry2/aN/at+F37Ikw+2/Dfw3pUfxK8e255h1IfaGg0ywmHRkaaGadkPBEScYNfcyosSRxxKqxAAKoGAB6AV+H8b8U1q2JlgcPJxpw0k07OT6q61stnbd3voj5LNcdKpUdKDtFb26v/JfmfFmmf8E8P2OrMNLq/wAFPD/jrVW5l1DxJLNqt3O3dnmuHdiTWyP2A/2Kh/zbH8HPx0WL/Cvr0VmaxrWj+HrCfVtf1bTdE02P/WXF3OsMaemWYgCvzb6jRevIvuR48cHGT5YxTb8j5WP7Af7FQGD+zH8G/wDwSxf4U3/hgL9iw9P2Y/g2P+4NF/hX0r4b8b+D/F4uG8J+K/DviZIj+9+w3kc3l+m4KTiup7daX9n0f5F9yLr5eqcnCpDla6NWZ8hH9gP9ivHP7MfwcP8A3BYv8KX/AIYD/Yq/6Nj+Dn/gli/wr69yDg4pCT2xR9RofyL7kY/V6f8AKvuP/9P+sr/hgT9ir/o2P4O/+CWL/Ck/4YA/Yp/6Nj+Dn/gli/wr6+3L6j86XIHQV/Iv1Gh/IvuR+b/V6f8AKvuPkEfsCfsVnbn9mH4OZ/7AsX+FL/wwJ+xVz/xjF8HP/BNF/hX1yAcdh+FPo+o0P5F9yH9Wp/yr7kfIf/DAX7Ff/RsPwc/8E0X+FL/wwF+xV2/Zh+Do/wC4NF/hX0z4h8X+FPCFrDd+KvEmieG7aRtsb310kIc+gLEZP0qxoniLQPE2nrqvhzW9J8QaaxKi4s7hZoyR1G5SRn2o+o0P5F9yN3lbVP2rp+53tp99j5c/4YE/Yq+9/wAMxfBzH/YFi/wpf+GA/wBir/o2L4Of+CaL/Cvryij6jQ/kX3Iw+rU/5V9yPkL/AIYD/Ys/6Ni+Df8A4Jov8KP+GA/2Kv4f2Y/g3j/sDRf4V9fZ9qM+1H1Gh/IvuQvq9P8AlX3H/9T+sv8A4YE/Yq5/4xi+Dn/gli/wpP8AhgP9iz/o2L4N/wDgmi/wr6+z7UZ47Zr+RFgKH8i+5H5v9Xp/yr7j5B/4YC/Yr/6Nk+Dv/gli/wAKX/hgP9iv/o2T4Of+CWL/AAr69z7UZp/UaH8i+5B9Xp/yr7j5C/4YD/Yq/wCjY/g5/wCCWL/CgfsB/sVf9Gx/Bxv+4NF/hX17mkzgc0fUaH8i+5B9Xp/yr7j5D/4YD/Yq/wCjY/g5n/sDRf4Uf8MB/sVDJP7Mfwc/8E0X+FfXufakyOMgZo+o0P5F9yD6vT/lX3H8W/8AwSk/Ze/Z58b/APBbf/gsz8O/Fvwd8Ca/4G0NtPGj6XcWKtb6ZuumB8lf4Mjjiv6T9U/YY0zwHv8AEP7I3xR8f/s1eNIf3ltbWV/JfaBdsOfKvNLnLI0R7+WY2HY1+OH/AASi+DvxY8Ff8Fuf+Cy/xE8YfDXx14Y8Ba++n/2HrOoaXNBZavi6Yn7NO6hJcDn5Sa/qIPqa+l4ihBYqNSnpJQhqtGrRWzWp346EfaJrey1Wj27n/9X+wL9mj9qnxT4v8Yal+z5+0Z4Y034a/tJWFo1/HFZzF9K8WWK4U32lSP8AMUBI3wtl4iecjmvuzepJZSG+lfmh+2l8I9R8c/DSP4l/D930j45eAJj4r8H6lCMSpcQDdLaMRy0NxEJIXQ8EODg4r7a+BvxQ0b41/CD4bfFfQto03xBo1rqyR55i82MMUPuCSv4V+ccBcT1cZCWFxTvUgk0/5o+fmno+90zwsnx0ql6VR3kuvdf5r/I9YZ0TcWwDjPXrXx9o/wC3f+y74h+KOh/CXQviFca14m1LX7vwrp93aaReS6Ve61bxPLPp8WprEbVrmNIZi0YkyPKkB5RgNv8Abd1n4naJ+x5+1FqnwTs7/UPjHB4A16Twvb2ilribUxYy+QIVHJk8zbtA5JxX5z/sy/tWfst/s6/sM+DfBvw68I+KvH8/wi+Ddp421ELojwJDqS2wjmge6uVUJqk00t0WUZcBpS3UBv0SK1d1tb9fyS/FHvNaK3X/AIH53P3EOMHPSjpk15J8E/iHrvxU+F3gnx34o8Aa98LPEmo6fDc3+galLHLPpk7IC8ZkjJV1BztbgsMEqpOB6yDn0A71pKLTsyU7q6He+OaTaPfrmlPUdRSf98+1IZ538WfFumeAvhd8R/G+t3E1po+kaFfalcyxymN0jigd2KuOVOF4I5Br4M/4Jz/G/VLz4Mfs+fCD4t6t8S9X+OGteBf+E/kvPEpaWe+tLi4DsPMZi4EJu4oVWQBiqj0r67/ae+EGo/H34DfEn4M6X4jj8KT+IrFdNlvmiMgit2lTzhtBBJaMSJ16sK8F1v8AZr8aeFvj146/aA8MeKTc6HF8OLfwro2h2kJW+tRax3T+RA5ygWaWWCQsMMGgRcY5oA//1v77oJoLmPzYJI5osldyMCMg4PI7gjH4UlzdW1jbzXV3cQWdrGpZ5JHCog9STwBX5/fsl+AfjX8Ov2TPhT4S8HNZ6d48QSXmunx7b3MjSXdwxuLnylhdHA8+aQZbrhjjBBr3+a1+KkvgL4hw/HSy+GPijS2091tbLw3pd3K0x2tlZIZ2k8w52YCj1oA9y0fWtG8RaVYa74f1XTNc0S7iWe1vbO4SeC5ibkPHIhKsp7EEg1+SfxG1zW/25/jJ4s+Htlrmq6R+yt4Lvm0/WBZStC/jHVk/1lu0g/5dozlSB978QR7T+xXD4o+Gv/BM74MaHr/hvxB4L8beF/hnDYX2nalZva3FheW9lhkKMB0Zeo444rz39gfRrLSf2VPhhf267rzVkutYvpTy09xNcyMzse56D8K+F4zx04QjQg7KV2/RW0PDzrEuKVNdT6g8JeDvCvgLRLPw34N0DS/DehwKEjtrOJY0UDjOB1Puea6cHoCeaazYAIpwIIzjAr87PnBvYjkfWlBycYwaMZBHP406gBM9Dgmk4PXgg0oxgY6U0jrnJoA//9f+170H9aQk4H8NPpuBjGOPSvwE+AFyCcZ5pBtY5HJpB95hS/7u3NADqZlf4cZpBgkH/OadwAD19KAH9qbkn+6RSHkjheK+TP2j/wBoXxN8NtT8KfCz4ReF9E8a/HHxBbT3tlb6rdvb6ZoWnQlVm1LUJI1aTyULqqxIN0z/ACgr8zLpQoSqSUIK7ZvhMJUxFWNGiryeiR9advmGcULjsCK/KGH4nftW6bpvgzxp4Y/a1/Zn+NN/rrv/AGJ4YufC66Rp/il0UvJb6fqMN5PNGwVX/eMlwFxll64+8/gJ8btC+PXgMeLdM0bWvCWt2t3NpOu6Dqe37ZoGpQkLNaz7CVJUkFXUlXRlYcMK6sZllWguaex6+c8MY3ARUsTGyezTTXpof//Q/teB4GBxSADHA596dgdMcU0DgfeFfgJ8AKORjGBikABzwVp3tgYpD2zxQAEd9uTSFc5Yg59K+av2jv2gZvgvp/hTw74O8JyfEf4z+KbqSw8K+HRcfZ4rmWNN8tzeXG1vs9lCuGkl2seVVVZnUH5WGsftnr4jl0WP9rT4D3HxVj05dbfwW3w9YaaLZpCgXzxe/axEXVo/tGScgny/4a78LlVWtHnjt5n0eT8J47H03Vw0LxWl20rvsr7n6gkDHTiomAkVkZQyEYIIyCPpXzr+zh8fT8cNC8Taf4k8Lj4e/FvwzfjSPFvhw3QuBp10Y1kSWCbavn2syMHim2ruG4EBkYD6N4XnnH8q46tKVOThJao8Gvh50pulUVpLRrsz4a+NHwE1n4catN+0Z+y9s8E/FTTD9r1TR7Vdlh4utFy0lvPAuB5hA+Vx3HqQR+jHwE+Mfhz4+/Cfwf8AFTwwSlhqdvvlgY/PZ3CkpLC/+0jqy/hXGEB1wQpU8EHvXyx/wTiY6Bqf7W/wvhZo9L0L4iXMljAD8tvFcIJNqjsO+PevquDsfOGI+r392SfyaPSyau1U9n0Z/9H+ub/grfIkX/BLv/goMznj/hT3isfnpk+K/N7/AINVI2i/4Iufs87xjdrPiJgfY6lLXsv/AAcV/tLeCP2dv+CTH7WFt4j8SabpXirxl4dm8G6BYvcKs+pXN4RE4iTq4WNpGbA4A5xXT/8ABvZ8Gdc+Bn/BIf8AY68KeJNOk0vWb3QpPEMkDqVZFvZ3uELA9CUkQ/jQB+02OOOKb8xzzgV/PL/wVt/4LtaJ/wAEj/2qv2Z/hz8VfhHc/EL4D+ONBvL3WNT0q58vVtBniuUjWaKFh5dxHtY7oiUbuG42n9Xf2Pv27v2Uv28vh1B8Tf2W/jF4W+J2glVa7t7aYJfaW5H+ru7VsSQv2wwwexNAH1+CCMijI9RTQMAjv9aTOOMf+PUAKAGGTyaXkA8cCm7h2Xmvzi0/41fEL4TfEnw38J9J8I+LPiXq/jT4peJhJca5dPp8ekaRGhuS9j565nt4g8ESlRsLOwUmgD9HTn2HrS4zyAuK+ddB/ad+Fvi7xjefDvwk/jm98cIbmOKC+8H61YWck0SsSpv57RbcKSpAfeQf4d2QDS/ZV+NPiX47/DjxD4r8X6Do3hnX9M8Z+J/CVxa6fcPPbk6Zq1zYCRXdVY7xbB+QOvSgD//S/rM/bMAH7TH7DmOn9uav/wCkDV9JE4zwTXzf+2WB/wANL/sNnOT/AG5rH/pA9fR5P+c81+QcUf79P5fkj4/NP94l8vyQvDZ4P+NHOeg/xpM8A53f1pScECvAOAQZ9xxS/P8A7NIT6lSKTlcDIFADmzjjrQeMkDmlyPUUDBwQKAGlsHGDThnHPJpo2E9s0uVA4IoA/9P+14ZzkgCkGQFxQMDJzuNLkDHOT0r8BPgBQcjrmjHXgU0YwvrSjb04NABzxwOtOpDjvik4wef1oAU57DNJluOABSgg9KTjoCPagAU9jnNGW+vH60cHrg+lBxz0OKAP/9T+2Cmhl4FKMdsUgA4OBX4CfACcsOgP4048AnHNGV9RSZByc47fSgBTyccZ+tct43Xf4N8YLzzpV2P/ACC9dQMY5YGvPfi1rdj4b+FnxK8QahNHbWNjoGoXU0jEAIiW7sST9BVU1qkVHc/mT/4NMiw/ZH/bFgB4T45at/6QWf8AhX9V+DkZH/1q/lq/4NM9Eu4f2DP2gvGs0ZWw8Q/GbW720YjiSNLSzjJB7jcrD8K/qVJ4GCCa9TPX/tlT1OnHfxpHxz8fT/xkl+wznj/irr7/ANJK/VsYr8o/j9/ycl+w3/2N1/8A+klfq4owOua+z4J/3ef+L9Eezkn8OXr+h//V/v0II53EClHqMflSkDIzXyl8e/22f2Yf2W/iB8GPhr+0D8WvDfwo8RfEGe8tPCsutM0FnqFzbiIvC10R5UTnz49okZd3IBJGKAPb/ih8R/Cnwd+HXjb4q+PL2403wX4e0yfV9VuYoHme3tYULyOI0BZsKpOACeK6LQNbsvE2h6N4i0w3I07ULSG9tvOhaKTy5EDrvRgGVsEZUgEdDXxd/wAFA/DXiv4wfsoeJ/hX8MYdX1y78c32k+GpL3SVE5stNur2Fbq73AMnlpbiYljlfXOa+oPhLoNn4W8Dad4Z0/SNd0XT9Ou7+xgj1O9lvLmaOO8mRbiSeVndzMFEwLMTiUdOlAHpedw2n71O993FJgdvUU70zjdQA3PPJycdK/Df9tX9qT9kX/gjL4U8S/E79oPS/wBoT4z+APi947vrweH9L0Wy1m30vU3RrqZhBK0GI22jDO0hGyNRgKK/cfA2saxtY8OaB4gjhh17RNJ1yKJt0aXlskyocdQHBAPvQB/GZ8Z/+Dl7/gkv8WPg98Qfgnonwc/bn+D/AIZ8UafLpmsXPhf4baPbXFxbSKUkj+a7ZPmViu7bkdiK9u8C/wDB2z/wTU8F+DvDHhGX4Zft8eIzpljDYi+u/AunrNdCNdoeQR3ypuwB0AFf1S/8Kv8AhsX3/wDCv/BW7GP+QVBjH/fNP/4Vl8N8n/i33gjOP+gVB/8AEUAf/9b174df8HPP7APh39qj9qn43a78Kv2wLzTfFsfh+y0ZIPCNo9za2tnayK6Tob0BMyTOygE5BzxX09/xFsf8E3yD/wAWd/bdx/2JNl/8n1+ufwH8BeBpP22P+ChFtJ4L8KPBFqfhYRRtp0JWMHSyTtG3Az7V9y/8K7+H/wD0Ifg4f9wuD/4mv52xeNy36xV5qMm+ed/f3fM7v4erPi5VaPNK8HvLr5vyP5pv+Itf/gnBjB+Dv7bpP/Yk2X/yfX6NeC/2o/hb+0T8HNM/bf8AEkOv6V8GLm2GoeHbPWdOZptB0/cImubm1iMgScv5jO4LCOMfeADGv1B/4V18Ps4/4QXwcf8AuFwf/E18UfHj4FaXa+HfiL4d1XwprfiT4A6+xvdYsNEne3n0lcq9xGY4WSR7OUxkukeTiSVSpRjXVlWNy9YiLVNw83K/6I/UPCnMcDRxs+e0KrjaDk9L9Vfo2uva662fOaP4o8DfFXwhD8ZPgxrtlcXtn9pfRfENnAyR3ZiJ3KHIH2i1cqVJGUcHKngGvyeH/B2Z/wAE8LILZ6x8G/2z49Vh/dXK23g6ykhWZeHEb/bhuXcDg4GRg4r9Nf2dPDfgjx18H9A8N/slaNqukfAXVkmurDX2jlh07T7OdyXTTIrjDlRlxHHGghj7YA21+imkfCf4Z6JpWmaNYeAvCEdjaW8dtCrabCxCIoVckrknAHNdee43L/bWnBzfk7ffo7nr+LeY4Op7CnJqdaN78r2Wlk363tHdavS+v823/EWx/wAE3/8Aojv7bv8A4RNl/wDJ9H/EWx/wTf8A+iO/tu/+ETZf/J9f0t/8K6+H3/QieDf/AAVwf/E0f8K5+H3/AEIng3/wVwf/ABNeL9by7/nxL/wP/wC1Pxj21D+R/f8A8A/mj/4i2f8Agm//ANEd/bd/8Imy/wDk+l/4i2P+Cb//AER39t3/AMImy/8Ak+v6Wv8AhXXw+/6ETwb/AOCuD/4ml/4Vz8Pv+hE8G/8Agrg/+JqfruW/8+Jf+B//AGoKrQ/kf3/8A//X++/+Itn/AIJvj/mjv7bv/hE2X/yfVrT/APg7J/4J0apfWmnWXwf/AG1BeXEiwRGXwZZqgdiANx+3HAyRk1/Sj/wrr4ff9CJ4N/8ABXB/8TUM3w1+HdxDLBL4D8HeW6lGxpkA4PH92v5weMy5f8uJf+B//anxkK2H5lzQdvX/AIB+dfiT4rfC/wCHngq1/aG+OnifStLvdYFowvtSdWa0S5kRYbS1Qn5I186MEJ/tO2eTVb44/G3wH+yf8MfFf7Y1tb6hqHg/w9ZJqfiOLw2kc7+INL3qrBIw6xzSqH3RsWB4I3YJrI/aq8AfCn4f/B3xFpH7V3h+11D9nXR5be9i8UfZxcR6bDFcRyQpcoqtLE4ZI4y6qVkXjI3EV9CeDvAlz8U08OWEXgd/CnwXtZYLzF/aJb/2ukbB4oYLM/MkO4KxaRVyAAFIOR9VPH4D6ir6x7df+H8z+tMxzzLnl1SaqQ+rOFkrrs7R5b7rS0d09dLXPxa/4i2P+Cb/AP0R39t3/wAImy/+T6P+Itj/AIJv/wDRHf23f/CJsv8A5Pr+lr/hXXw+/wChE8G/+CuD/wCJo/4V18Ps4/4QXwZn/sGQf/E18p9ey3/nxL/wP/7U/kr21D+R/f8A8A/mk/4i2f8Agm//ANEd/bd/8Imy/wDk+l/4i2P+Cb//AER39t3/AMImy/8Ak+v6Wv8AhXXw+/6ETwb/AOCuD/4ml/4Vz8Pv+hE8G/8Agrg/+Jo+u5b/AM+Jf+B//agqtD+R/f8A8A/mj/4i2f8Agm//ANEd/bd/8Imy/wDk+l/4i2f+Cb//AER39t3/AMImy/8Ak+v6Wf8AhXnw9/6EXwZ/4LIP/iad/wAK6+H/AP0Ing3/AMFkH/xNH17Lf+fEv/A//tQVah/I/v8A+Af/0Pvz/iLa/wCCb3/RHf23f/CJsv8A5Pqnc/8AB3P/AME0bNoFu/hV+2jaNK2yIS+D7FDI3oub/k+wr+mY/Dr4fZ/5ETwdj/sFwf8AxNfyy/8ABxf4S8Lad+0B/wAEaItH8M6BYxSfHSzW7W2so0Ekf2mz4kCryvXrxX8/5XHLMTWVH2Mle+vP2V/5fI+Sw0aNSfIov7/+Ad9/xFs/8E3v+iO/tu/+ETZf/J9H/EWz/wAE3yP+SOftukf9iTZf/J9f0tL8Pfh5J88XgfwXIPVdMgI/9Bo/4V34A7eA/Bo/7hcH/wATXn/Xst/58S/8D/8AtTCVSh1g/v8A+AfzS/8AEWz/AME3v+iO/tu/+ETZf/J9L/xFsf8ABN//AKI7+27/AOETZf8AyfX9LR+Hfw/HXwL4MH/cMg/+Jo/4V18Pv+hE8G/+CuD/AOJo+u5b/wA+Jf8Agf8A9qJV6PWD+/8A4B/NIP8Ag7Y/4JvDj/hTv7bv/hE2X/yfS/8AEW1/wTf/AOiO/tu/+ETZf/J9ftZ8If2mf2IPjx8dvjd+zR8KdT8E+KvjX8OTGPGWijwvNAdHLuUXM8tukMuWGP3TvX1b/wAK6+H3fwJ4MA/7BcH/AMTWtStgKbtUw8k/Off/ALdLc6K3g/v/AOAfzOX3/B2f/wAE3buyvLVvg3+22UlieMg+CrLBBBHP+n+9eVfsYf8AB1H/AME/fgP+zh8P/hP4z+E/7ZWpaxo8moQrJpng+zmtxbvfTywqrteqSVjkRSMcEEc4zX9WF58Ovh99muv+KE8GqfKb/mFwen+7Xzh/wTI8AeAbz9jvwFPd+B/CV5cnWPEQLy6bAzMBrN4ByVzwMCvp+B8Vg5Zhy0KbjLklq5XVuaGlrLrbU7cqnSde0I2dn180f//R+I/+Cnn/AAc8fFzXf20vhL+0F/wTa8cfHX4UfD6w8HRaN4k8KeO9Ihh0/XrpLyeXe9gJpo2GyVV85Skg5AOBX7df8E3/APg4m+AP/BXbw/qX7DP7WnwC+I3w2+Kfiuw/sye98IW1zfaJqYJX5jPDm4sH3BWDOpjGOZB0P2z/AMFLv+DeP4Pf8FPP2yvhN8ePiv8AES9+HfwZ8M+EotBufC/haxitr7WrgXc07M90QUijKyouQhbryOK/YH9kb9hH9kz9hn4e2Xwz/Zc+B/gr4U6BHEq3M9pbB7/UXAwZLu8fdNO59XY+2BxSa7hF21R9UaLpkOi6PpWjW095dQ2ltHbpLcSGSWUKoUNI55ZjjJY9Tk1rfd2il4A9qQYYZwDTAXrkUnOeQMU3jII6dsU/oPmxQAE9hjPvSdD/AAgUuRjPakxuIOeKAPlb9tb4K/Fj9on9l/4u/Bn4HfGrWP2dviprunra6P400/zftGhSiVHMsflOj5Koy8MPvV/Mp/w4E/4LPf8ASfz46n8dW/8Akyv7FMEEnjFGc4OcCgD/0vrD4g/8EG/+Cxui+BPGer6t/wAF5vjjremWul3VxcWjnVdt1GsTFozm7xhgCPxr5t/Z0/4Iuf8ABV7xv8Dvh34t8G/8FsfjT4F8OX2nmay0iFtTEdim9xsG27AxkE8DvX9w/wAaj/xZ74pkD/mX9Q/9EPXwT+wyf+MSvgaN2P8AiT/+1pK+G4szGpRqQULap7pPqu6PDzbEyhKKj+R/nw/Ekf8ABX79lz9tXwt+yZ+3P/wUx/ao/Zh8H+ILt7Tw98SpNe1HU9B1JS22Gfek6FImJCuT80RPzKBlq/fW2/4Ic/8ABXu8t4Lq0/4Ly/G+5tJUEkUkcmqMkikZDKRe4II5zX7X/wDBVv8A4J9eCP8Ago7+x78Sfgdrtlp1v4/htZNV8GazIg8zSNZiUtEwbqI3I8tx3Vz3Ar4S/wCDbH9sPxZ+0v8AsEy/Cr4q3d3L8Y/g/rsngLV1umLXD2saBrV5CeSwAlhJ/wCmFeVVzadTDe3pJJx0kuVddmtDllinKnzwsmt9F958hf8ADjD/AILBf9J4fjv+eq//ACZXwF/wUI/4Jyf8F9/2MfhHcfGv4Wf8FL/2j/2qfCWmRyT+I7PRfEGpWep6PAoz9oS2eZ/tEQAO4o29eDtIyR/eiMZJyKilghuYpbe5jimt3Uo6OoZWUjBBB4INeXR4grRknJRa7cq/yOSOYzTu0n8kfwW/8E4f2Tv+ChX/AAUv+AFj8bfgx/wXo/aA0jVLeX7B4l8MX8mp/wBoeGdQAyYJwL3DKRhkkX5XU9iGUfoIf+CGH/BYIYH/AA/e+OpP+/qn/wAmV45qngm1/wCCMH/BwR8LH+HxHh/9kP8AadRrG70mI7LXRdWllMYSNegCXZgkX0juXUdK/s2IAPJ56V3ZnmdWnKM6VuSSuvdj923Rm+JxM4tOnaz1WiP5K/8Ahxj/AMFg+M/8F4fjt+eq/wDyZS/8OMP+Cwf/AEng+O/56r/8mV/WnxgYwTijEf8As15iz7EeX/gMf8jn+vVPL7kf/9P6MH/BDH/gsEc4/wCC8Hx1/PVP/kyk/wCHGP8AwWC3f8p4Pjrn66p/8mV/WnwTygpfl/2etfjH9vYjy/8AAY/5Hxv16p5fcj+Sv/hxh/wWCT/nPB8dfz1T/wCTKd/w4v8A+CwXH/G+D47f99ap/wDJlf1oDb2wfwpcIOMAmj+3sR5f+Ax/yD69U8vuR/Jb/wAOMf8AgsFnb/w/g+Omfrqn/wAmV9UfsE/sv/tEfsD/ALQfxQ8GftjftT+Mf2w/GfjjwlZX3g/xZqouZLhIdPmujeaRAs8kjbx9qguAin597cfKa/opwFyB96vI/jP8Dvhv8ffCI8HfEjSLq7tIp0vNPvrC9lstR0a8T7l1ZXkLLLbzqejowPY5BIO+Gz+opfvEuXyST/BHs8P8STwWLhiZRuo7qy2asfgp+yD+zr8ZvhT+1un7RXj74L6la/C3x8utR+GPDqyPJN8DmmuPtD+dbljFEuoiMSTPHzDKFj+6xNeSfF7/AIJ6ftuft0/tD/Hn4/fsd/8ABR74hfse/Ba41qLRodO0F702niS9s7aOG51FDBPGjDzFMG8A7vIPPFbHwD/ao+BH7Uf/AAUd+Pv/AATQ8Tftc/tK+PfCHhWwjXSlm1fTbGDxndwEjUdOmvrGzgu5FiDRjCzKZQk24tjn+mvwr4W8NeCPD2jeD/BugaP4W8K6dbpaWGnafbpBbWcKjCxxxoAqqAOABXrZtnEqVlFe80t1pb0Z9XxbxlRrYeOFwsWk3zNyS/D+tNkfymf8OMf+Cwecf8P4fjrn66r/APJlKf8Aghl/wWDH/OeH46/nqn/yZX9aZKtg5GaRscEYrwln2I8v/AY/5H539eqeX3I//9T6MX/ghj/wWBPT/gvB8dR+Oqf/ACZS/wDDjD/gsDnb/wAP4Pjtn/e1T/5Mr+tEEA9eetB6gYFfjP8Ab9f+7/4DH/I+LWOqPt9yP5mP2T/2Lv2s/wBgj9qHRtW/bF/bv8aftY2vjzwrqPhHwb4h15bl4/Cms+dBcC1X7TNIoe5jhdlGVDm128kgH1yz/Z2/bET9uXVNXf8Aay1w2/8AwrO0iOvHwBYeTKBqkzfYsZ2bwDv3Z3Yb0r9zfiX8MfAPxi8F6z8PviV4b0/xZ4R1CMJcWlxkYYHKyRupDxyqwDLIhVkYAqQQDX89vgj4/fs9/EP/AIKQfEH/AIJWaR+2t+1ldJoXh9J/LHiHT1W/v1XzLjRotXWyGoM8UDxsXFx5uRIu/KGvcy3OXOEudaxWtkrWP0zhbjejQwv1fEwl7j5k49fJ6r7zl/2qv2Hf2u/2/P2u/iD4/wD2Mv25fG/7HPhXwh4Z0bwP4m1zQTdLH4z1yGa8uZISLeaIN9kjvIAWO4BrlkGChryZf+CGf/BYIn/lO98dfz1T/wCTK/qa+HPw38DfCTwXovw/+HPhzTvC3hDTovKtbO2B2rzlmZmJZ3YkszsSzMSSSTmu0yOc7cV49fP6zl7iVvNJv8j4bOM+qYrFTxFklJ3tZM/kw/4cX/8ABYJcn/h+/wDHTH11T/5MrwT9nH/giv8A8FcPiB8S/wBpXQPBv/Ban4reBrvQ/EEVjrGoRS6t5uvTtAGE8gW5HzBTt5JPvX9puBgZABr5X/YKx/wvL9u3uP8AhNbX/wBJFr1eHs4rTxcIStZ36JdPJGWXYucqyi7dei7H8sHxU/4M+f2vvjt4iXxf8bf+Cq198W/FSkFb7xJo2o6lMnsrz3bFR7DAr69sP+Dff/gshpOn2OmaZ/wXu+N2naZbRJb29vAmqJHBEoCqiKLzCqAAABwAK/sjAPc5pMAEHOK/TT6Y/9X8Nv8Ag4W/YY/a/wD2Lfix+zr4U/ar/bf8d/tyeI9e0O9udGvdXW7aTRokuFRoIhPLKTvYhsLjkV9Sf8EI/wDghZ/wVY+IfxW8E/tU+D/iP4+/4J6/DG1ZLiPxTOJYdX8Q2x5MNrphAWeFxwTcgRYOQHIxX+jd8Uf2Jf2XfjZ8ePht+0l8YPg74T+JfxZ8H6bNpnhi91qD7XDoySyrK8sNu+YhPuUYlKl1GQpGTX1RHFHDGkUSJFEoCqoAAUDsBQBj+HdO1LR9B0bStY1y68TapbWsUNzqM8SRyX0iqA0rIgCKzEFiFAAzwK3cf7K0h9v/ANVLhv736UAClcZHAr+U79uX/gtF8XvhF+1v4t+G0X/BFH9q39pJvhp4gubTwv470nR9QltLvcqh7mzkjs3Uo4C9GYEoPQV/Vjj5cbqQ8rkkUAfx9XH/AAcrftoX1vPbN/wQZ/bxngkRkcf2PqoypGCMjT8iuA+Fv/Bdz9sL4T6NqWi/BT/g3Q/bb8OeHLy/udUuobbSNbKzXs0jSTTHOnN87u7Ox6szEnk1/aEeAvY0/j+HaDQB/Cp8ff8AguD/AMFJfib8Xv2cPEaf8EN/2rvBmv6Dqd/Ppul6vY6tC+vSSWzI0cO/T0JKLlztDcDtXsA/4LVf8FnLzB07/ggd8dh6ee+pL/O0Ff0Q/tlnH7Sv7DYH/Qd1j/0gevpWvzHiDG0YYucZUlJ6atvsuzPl8wrQVaScE9u/Y//W+r1/4LFf8FyrgkWn/BBf4lxE9POvb8Y+uYBUg/4K1/8ABeucAW3/AAQv8QQenm6ndjH5gV/VWy7j1o5GOgWvxj+0qP8Az4j98v8AM+L+sR/kX4/5n8qg/wCCqP8AwcFT/wDHv/wRGt4PTztXuBj85BS/8PN/+Dii4/49v+CLvg+HPTztclGPzuRX9VYODjczGlwTj5j+FT/aVL/nzH/yb/Mf1mH8i/H/ADP5V/8Ah5J/wcdr+9k/4I2/DZ4P7i685b/0s/pXNa1/wX2/4KNfszCLxD+3j/wRx+MXwx+FsTBdQ8UeHZbqe0sVzy7u8TwgYzgNKuexr+swA87hXP8AivwroPjrwzr/AIL8V6VZa54a1Wzl0++s7iMPHcwSIUdGUggghj1qoZlQbtOjG3k2v1COJpt+9BfieJ/so/tVfBb9tH4F+Bv2ifgF4qi8V/DnXoGeCQrsns5lO2W2uIusU0bAqyH6jIIJ+i8fLx+HFfyb/wDBrnd6l8Prf/gpD+y013K/h7wH8W7uDTIHOfIQyTQtjsM/ZkOPWv6yNvXk4NcuZYRUK8qcdlt6PVGOKoqFRxWwEZHUg0DJHIpR6ZyaPy21xGJ//9f+1/uaQdf/AK1KPXOaM+nNfgJ8AeH/ALRH7R3wR/ZR+FniP41/tB/ETw78MfhvpSb7nUdQl2h3P3YoUGXllY8LGgLMegr+b/Xv+Do3wZ4717UNL/Y0/YB/a3/ar0uCQxjUdM0eWOOXH8QjhindQf8AaAPtXmf/AAUN8Np/wU//AOC9fwA/4J1/ES71K8/ZZ+F3hePxp4s0GGdootcvpIhcFZipyQUks4vUKZMY3Zr+sX4efDX4e/CTwrpfgb4YeCvDHgDwhYwrBaadpFlHawQIBgAIgA7depr2/ZUMPTi60eecle17JLp53O/kp04pzV29T+X1f+DgH/goxOoltP8Aggh+2zNbtyjf2TrHzD140yj/AIf8/wDBSM9P+CBP7bBP/YL1nn/yl1/VzkkcEgUu5j/eH41n/aGF/wCfC++X+Zn9Zpfyfiz+UUf8F9v+ClTEbf8AggT+2r+Omax/8q6D/wAF7/8Agpmw4/4IFftmD66drH/ysr+rvcR3Y0uT/eNL+0MN/wA+F98v8xfW6f8AIvvZ/KIP+C9H/BTz+D/ggZ+2Hj3sNX/+VtO/4fxf8FRm4X/ggZ+17+Nlq/8A8rq/q4BJxknNGemSTR/aGG/58L75f5i+tU/5F97P/9D6o/4fuf8ABU9vu/8ABA39rUfWz1b/AOV1Q3P/AAXa/wCCrkdvLLD/AMEDv2qgwBI3WerkfkNPyfwr+r/ijv7V+L/2hh/+fC++X+Z8T9ap/wAi+9n8jo/4Lu/8FgbkhbH/AIII/tGBj087T9cUfmbAV8e/twf8FAP+DiT9sb4L+MPgV4A/4JT/ABZ/Z28GeJLF9O1q+0/w3qF7qtxZyDEkEU84RIVdSVYiPfgkAjNf3SEn+I0hAPWtaOb0aclKFCN15t/qawxlOLuoL8T+I39gz9r7/gsX+wT+yv8ACv8AZY+F3/BCv4razoPh22l87Vb+6vIp9XvJZWlnupVEICs7uflBwoAHavsP/h7d/wAF6Jwfs/8AwQu8SRj/AKa6ldj+YFf1XEgAk80D8NvtSrZtSnJzlRjd+cv8xTxcZO7gvx/zP41fij/wU0/4Lla78XP2bda1z/gjhJ4b8U6XrtzceH9Ol1S4/wCJ3cNBtaLJYYwvzV9t/wDD2P8A4OO2GF/4IYaSmeBu1qf/AOPV+xPx9P8Axkl+w1n/AKG6/wD/AEkr9Wu/+z+lfccJYiNSjNwgo69L9l3bPcyiopQbStqfyO/8PRf+Dl18FP8AgiP4DXuA2vScf+TdfzMf8HHX7U//AAVC/aM8M/stwf8ABQ39hrw5+x9pml3+sv4YmsNRa5OtyyR2wnVszSbfLCREdPv96/1Su/I+tfkf/wAFMv8AgkB8Bf8Agql40/Zivv2j/EviuP4afDu81S/n8O6S4gbxHJci3CxzXX3ooVFudwQBm3ABlxz9Yesf/9H8zP8Ag3o/4KXf8FnPD/xc8Pfs9/szeAvGP7aPwHRkj1Xw14ieT+zvC1sWAM8essD/AGfgdEdmRugjJwR/qLaVNqFzpen3OrWMOmao8CPc2yS+asEhUFkD4G4A5G7AzjOK8e/Z/wD2bPgT+yt8OtJ+E37PXwu8H/CfwDZIFi0/R7NYVkIAG+Vh80jnHLuST617mCeg+b3oAUAjjgihhkYoxzu3cU3OVbJ5oAX5h1YUAcDk0E89VpeoAP40ANBzuxupctz90UhHXG6lz2PI7GgD8p/gOM/tv/8ABRH/ALCfhT/01GvuvsK+E/gKSf24P+CiGVx/xNPCn/pqNfdnYV/KuN/3qt/18qf+lyPz+fxS/wAUv/Smf//S/uI6A4r+cP8A4OY/+Cik/wCxT+wdrPwv+H+urpXxv+Kol8MaW8UgE+n6WQPt10oByD5TGFW7NNkcrX9HnSvwq/4LQ/8ABLz9kD9rH4G/Hr9qX45eD/Ffir4ueBvhfrs3he4TxDd29rpj21pPcxstrG4jJ80BmJB3YAOQK/mjh6dCOMpyxCbinsu/T5X3PiMHKCqrn2Pxy/4NE/8AgozJ46+GXjz/AIJ4/EzW0n8SeF2l8TeBJbiX95caVKyi5sVBPPkynzVHXbO46KK/thwV9SK/kG/4Nqf+CXP7H9z+yR+y5/wUGHg/xVZftSC51ctrdt4gu4oZVW7ntwj2gfymQxgKVK4PXrX9fXUe1dvGMqEsfUdC6119etvJ7muZODrNwEo9jiiivmTjD9aQZwM9aXrRQAUUHnuRRQJKx//Ti/4O+f8AgobHoPhf4a/8E7/h34hQ6zqnk+LvHa203zW9orH7FaS4PBd1aYqf4UjP8Qr9gv8Ag3A/4KG2n7c37AHhbwt4s19NR+OfwwEHhLxJDLKDcXFqqf6DeEHkrJEhQt/fhcV+Zf8Awc3/APBLn9j7wR+yh8e/+CgOj+DvFc/7UWqeKdGW81288Q3dxGyTTLCyLbO5iVBGqoqhcKAMV+wv/BHj/gl1+x/+x98IPhB+0d8A/B3ivwl8UPHHw40STxPNJ4hu7m11J5raG4dmtpHMYIkZmUgfLuIHBNfh+MrYH+xKcIp813Z2+1pzddrOy9D5io6X1VJb/qftzRRRX5+eQHOe2KKKKADgY9qwvEuv6f4U8Pa14l1V2XTbG2kuptoySqqSQB6nGK3sA/SsLxJ4e0/xZ4e1nw1qaSNpt/bSWs2w4YK6kEg+vNOCVzTCez9rH23wXV7b262+R+cX7QH7QbfD/wAF2HxS+N/ivxT4O+H91rGlaJa6H4bWUOk1/eRWsBup4v30h3zoW2FEUZGHIGfBvEnxh/4J661Pdz+MNf8Ahf4j1LQr+KOM6rZzXl0t1I8iRtZiVGknLyW8savBv3PGygllxXvH7QPwM8XeNfhpL8LfHUfiqHTbfVdJ1ex8VeGbRbt1l0/UIL23kmtGDMrb7WMOoVlILYIyMfnP4c/Z/wD2Tv2efjj4G+LHxM/bGNt4vkvorfStN8Vy2enPqc0c9xJDF++VZpZg1/KDjr+7wq45/Wcohl7pr2Nr/j/mf2FlVWjCnH+x1D2FlrG2nrre+3xe931P/9T96Ph7+09+y942+I0vw3+GfxRvvCni2ZrI6LdaRqU8cGsLcWYuomi+9ASU8wCOYbnMUgCnY2P0U+GPxJ8Rt4kg+HPj65tdV1eazlu9L1aGEQ/2lHEVEiTRj5UmUSI3y4VgSQFwRX5jfC/9mH4SeFIdJsPh54g+JPxCuLbxXp/iyP7DpolV7i1t3gSJphGsQiKyMTlgc9DX6b/DD4c+Ij4oX4j+ObWDSL2C0ls9I0tJBI1lHIVMss7r8plby0XauQqgjJLHH4dxN9Q9m/Z25+lv1sfr3iF9R/s+p/aaiq9vctbnv087d76W87H0bRR1yKTkj0NfAH8wQnc/ky/4I9f8p7f+C4n+/pv/AKVtX9ZxB5+Y5r+TH/gj1/yns/4Lin/b03/0rav6zjycg8V9BxP/AL1H/BD/ANJR2Y9++vRfkVr3/j0ueR/qm/ka+cv+CXnP7GngDj/mM+I//T1eV9HXv/Hpdf8AXJv5Gvm//gl2Sf2NfAQxj/ic+I8H1/4nN5Xp+HP/ACN/+4cv/SoGuTf7yv8AC/zifoR6460hGNxHWgqCc0cdsA/Sv30+wP/V/v3PTt+NJtA6cGg9OQTS888/pQA0rnvzTuw7UnHcc/Sl49vSgBDyOflpcck0YGc96TgnHGfpQAf3iW+WlwDnIoXGBg5pAPTGPpQB5d8axj4OfFI/9S9fj/yA9fBH7DJH/DJfwOB/6A//ALWkr74+NeP+FO/FPkf8i/f/APoh6+B/2GW/4xL+B3/YGz/5Gkr8445/i0/R/mj5zPPjj8z/1v7XWGFY47YFfyGf8G94/wCEZ/4Ke/8ABbzwDpBMPhmLxfb3kcC/cSUajfrkDtxIw/Cv69X+4a/kN/4N+h9p/wCCqH/BcS/yTjxdbR5/7iV//wDE1+M5b/u1f0X/AKUj43DS/d1PRfmf14BgegJp9N/759qU+xxXjnEfyD/8HVaDSNX/AOCYHjXTv9G8R2PxUC206cOg3Wz4B7fMimv69S+/DE7CeTj86/kL/wCDrImW6/4JhWg5WT4rf+zWw/rX9e7gIzKMZBxXtY//AHPD/wDb35nfX/gw+f5jeCMFWx9KNxxkKad178Ug4Az1rxTgEyu08cU7v04pDkYwM06gD//X/tdJBwD/APqo6AD5vwpxOMnrRX4CfAAAB04r8TP+C8X/AAUkh/4J0fsSeKta8H6tDa/H3xok/hrwTGGHm2kzxkTX4TuLdHDjtvMYPWv2T8VeKPDvgfw14g8Z+LtYsPD/AIV0qym1HUb+6kCQ2dtEheSV2PAVVUkn2r8Vf2hvjD8W/wBrPwPcfET4KfsNfBrxz4H0uymufD3iT4t6Guq6jrcDYOdG0FTHJtl2qVae5tww2krjBr08owrqVlJx5orfp+J6+SZRiMXV5aFNztq7fqz/AC4vgj+0L8U/2ffj14F/aN+HviK/s/iloGuR69bXzuWe4uBJvcTHq6yZZWB6hj61/sSfsGftk/Dr9vb9lj4UftOfDgxWena/YK2oab5oeTRdRT5bi0kI/iRwwBPVSp71/Lx+w9/wTD8d/EX4v/F3/goD4L1z9mCPx7reoan4S1r4a+PvgrEPD2k3NpcRxyi2tbXVJBbtutwA6s5wzcc1++/wL/aIh/Zt1Pwt8Ffj/wDsy/Dr9luz13U0stK8TeASkvgzWNUlwqRSMIoZbG4lICos6FWOFErHAP1nFK9vFOENY9U/vVj6XP8AhrGxorESpPlXVNPTzsfqnx9DRnpjpRkZx3pDxjk9a+APgwK+mB+FKOnXNAz3xRz7UAf/0P6Gv+Cvv/BQvwr/AME3f2LfiH8aru8tJPibqMT6D4H0xmG/UNZmjby229THCA0znoAgHVhX+TB4D/aM+MHw+/aE0D9qLw34z1a1+M+n+I18Ux6z5pM0l/53ms7n+IMSwYHghiDwa/2dfjJ+zX8Af2iYdEt/jr8Hvh78W7bTWd9Pj1/TIr1bNnADNGJAQpIAyR6V/Hv8Ef2Rf2X9V/4Oe/2qvgRqPwD+FF98GtP+H9veWPhiXRIG020nOmac5kjtyuxW3SO2QOrH1r824YzGhSpVIuN3Zt+aWlvxPm8rxMIQkra2uf1Pf8E8P20PA/7f37I/wn/aa8ETWkbaxZCDWrCJwx0jVogFubVx1UqxDAHko6HoRX2z7nIryX4QfAj4Mfs/aBf+Ffgh8MPBPwq8NXV0b65sNB0+O0t5rgqqGVo4wAXKooJxnCj0r1ztxXx2IlBzbpq0eh41Vxcny7EfQDOBXyz+wSAfjj+3Z6f8Jra/+ki19THd+v6V8sfsE/8AJcf27ecf8Vrbf+ki17PDH+/Q+f5HblX+8R+f5H6gkZGKQA5BIWnU3A4PGMelfsJ9gL29KaQOgXNPpu0ZJPNAH//R/v3GcDPWo9p/uin5XHXg00MgAG39KAHkcdSKZ9cZ7UoPfH40FtvAGKAA4GMjjpQBg5+ag9s7d1IOTngevNAH5r/tl8/tL/sOf9hzWP8A0hevpLOAAcg+9fNv7Zn/ACcx+w5/2HNY/wDSF6+kcHGCNwr8g4o/36fy/JHxuafx5fL8kL2+bFL0HygGj/exikXoOSfevAOI/9L+1zjg5GPpTue23FGBz+tGcELivwE+ADn/AGc96VCSR9fypFUDkE0AsSvpkUAfybf8G8H7n9uH/gthaLu2L8WbhgP+3++r+sksRjpmv5OP+DfEeX+37/wW6hznHxXnP/lQvq/rIPIOK9fPv95fpH/0lHXjv4n3fkHJ9MUh65AyaD0IGKdXkHIA9uRSZOecV+aNt8Tf2zPi58a/2kvCnwk8dfATwF4I8EeJrfw3axa54XvNRu7wtptrdtK8sV3Co+a6KhQvRRzWj4cuP28vF+my6v4W/aU/ZD1/Soru5sZLi08DX8kaXEErQzRFhqOA6SRujDsykdq+exvFWW4epKjWrJSjurS0v6RaPfw3C+NqwVSEdHqtUf/T+jf2ePl/4Ovv20gOh+GFjknt/wAS7Ta/rRO5fT3r8S/Dn/BPf9oDwl+1n42/bf0Xx3+yzZftJeItMj0bVtePhDWGFzarHFGsfkHVPKX5YIhkKDx719GaHq/7dniTxB4x8K6H+0l+yZqHiDw/cQWusWqfD/Uw1jLLCs0atnUACWjdW4zwa/mbMPELJqzjKFdWjFJ+7Ppp/LsedieFcdNq0OiW6/zP0rphAOMKMV+aeifF/wDa4+Gv7VP7N/wX+Nni74IeP/B3j+z8St5vh/w5d6ZdabNp1pDcK26W6mV1fzipGBjGc1+luDyQa9DBY2jiKUa+HlzQls9eja6pPdHzmPwFXDVPZVlaQhJznAx9aXvxj3pBxgYNOrrOQQgHrRzgcZNKM8etJjnOTQAnfjHvSnPpmkPGTk4pTnHHBoA//9T+18d+MClopvbkkkV+AnwA6m457AfzoO7dxyKU59e9AHx18fv+TlP2Gf8Asbr7/wBJK/VodvpX5R/H7/k5L9hv/sbr/wD9JK/VwDgZr9J4I/3ef+L9EfSZJ/Dl6/oKWUjHNJ8vzD7tJ1JJwPrTgMj5utfantik9eMHFBwcH8aCTtJ6GkHJPcfyoA//1f79RjjC0fKcsOTRluo+YUpAOcigBAc46dOaUkf0o2gEEUuB1AFACHA55/wpBjG09aUAbcDpSbR95etAH5UfAY/8Zv8A/BRD/sJ+FP8A01GvursK+FPgNkftv/8ABRHkY/tTwp/6ajX3X2Ffyrjf96rf9fKn/pcj8/n8Uv8AFL/0pnif7Qvx8+Hf7Mfwh8YfGf4o6idO8J6PCrvHGyefezO4SK3gV2UPLI7oirkctyQMmvkn9qv4g+KPih/wTR/bO8W+K/hnq/wqvZPhn4vjg0681O0v2uLYaZP5dyk1q7xlJFOQM5HcV51/wU80XQPjJ4w/YC/ZV1Xw3oHi2Pxn8ZNP1rWLW9so7oRaDotrcaldvtdSFV3gtLdj3WcqeGIryX4vfHL4i/FH9mX/AILAeAn8I/CTwr8CPAHhXxT4K8Ipot84vZfsvh9WuGe08lI0hEtw8Q2sArQsoDDDn2MvwacYVLe9e/8A26nFaerbXyO2FJWjL0f4uy/C5//W/Zj/AINm/wDlDl+zB/121v8A9OdxX74HqK/BD/g2c/5Q4/sv/wDXXW//AE53FfvdnODX8z8Q/wC/Vv8AE/zPh8d/Fl6gRkY6g1w/hH4i+FPHd342sfDN7d3tz4f1eTQ9UElpLCsV4kaSMiNIqiQBZU+dMrkkZyCB1Op3BtNNv7oQzXPlwSSeWgy0mFJ2rjnJxivwr/Ym/a68Q/B39mv9lPwrqnw2/aC+OXxb+JfxB8SJqtrN55uvDaPJeanLGsmpOjTQ2dqLe3CoxQMjKGBXaebC4J1YScd1ZfffX0smRCm5RbW//AbP3j53ego4/Gv5ZP2r/wDgrd+39+0d+2t8R/2AP+CO3wd8B+KvFPgXEHjn4heLWQ6bpF1nbJGgZgipG2Y9x3s7q4VCFyctfgv/AMHZMoEjftSfsGWzNyU+whtvt/yD69JcOTUVKtUhBvW0nrbpokzdYKSScpJX7s/bbxt/wUp/ZJ+E37Ts37Ivxp+IcPwV+Ls1rb6hoY8TAWen+JbWXhZLK8J8piHDRlHKsGU8EYNfeVvcW15bw3VrNFc20ih45I2DLIpGQQRwQR3r/Lc/4OB/Bv8AwU/0vx3+zX4V/wCCi/xQ/Zv+L/xXuLS+bwrZfD+xP9qW1m0iBvtG22iJieQfIuWO5XIA5r9lv+Df34Df8F+/Beh6dd+LvGt18MP2NEtWkttA+KiT6hf3EJQlV0a0Li4tAflwXaOLnIRq9rH8JUaeDjio1op22vo/8L3+Vvmdssti4KSklf7j95v+CxfwA1f/AIKRfskfED9jf4Qazb6LrtxrenXd14kvos6ZYyWs/mPBkMHlkP3TsBCnIJzxX3B+yX4gX4cfCf4Kfs8eNtOutC8YeHvC+neH4bossllrT2lqkTPbSqeCfLLeW4VgDxnBNfGXxv8Aix8YPg98Bvhz4g/Z9+GT/Gbx1NItu3hmOF1fUAYpGnkN0Pkt2idWkJk/1hUxj53Wvd7XWtX8RfDD4WeIbyf7T42u7nRbq3dLCWyY3rzRFsW8v7yLgyAo3IXcD3rqxGQxWXqPN8N5Lbeyv8tD+iMT4ZZT9VlgopqpBN87b3tu1flt3Vrpder/AP/X/uI6004JxTga+NP+CgPjT4tfD/8AY7+OHij4HRa1/wALNh06GCwuNOgaa50+Oa6hhnu4kUEs8EEk0wwCcx9DX8pUqbnNQXU+Cpw55KPc+yc5OPShcHp07V/Pn8O/2hfEn7LPxK+Mnj/4Z+DP2jPj1+zvrOs/D74Y+DbO/wBZu7uXxb4ruGl/tTVrJtUl3CJYpIxI6bUllt26AM47z/grp/wVo8YfsPad8BPgl+zd8HL34y/ttfFeTy/B/g67UbtOi4Uz3ccbHJ3t5YQOFJSQltqE160MjrTqRp09b/Lom79rJmzwj51GOt/6/DZ+aP3Px0CrxXwJ8Hv+CmH7I3xh+NfxA/Zn/wCFjWvw2/aP8MatPo2reCfFWzTtSaaNiA9qGYx3UUi4kR4mbcjKcDOK/EDTvh3/AMHafjCyt/EV78aP2FPhfcXSiY6JLbxyyWGefLZo7WdSR0OJGHua/jt/4K3+AP2/vEX/AAUbm8DftE+Ivh38cP21I9M0yG7k+E1jKXVwpNvHIIYImN0qFCWC5ClMnjj6HJuEKWInKnOtF6X913tburJW+Z24bLYzbTkvkf6489xBawzXVxNHDaxoZHkY4VFAyST2AGa/n8/4Kgfsq+NP+CiXxM/Yf+IHwi1Cz8O+Hvg/8QYfGtxc6yvlR+Jo45YH8m02kugPkECV1CncCARzXyP/AMEo/hT/AMFx/AP7K/ix/wBvv4k6YfgeNISPTvDnilm1LxrDYmRBKXvUf9zGIPN/dztJLgAYSv0n/aS+N3xs+GHxF+EXhf4Q+Bl8V+AtYjhXxHqsej3F0ng2zNzHGt/+6IFyrBvJFunzpnzj8iNWuRZGqeKny1FJw0Tjtqn38j9W8OuEMBVoVMdjP3nLLlUU3bZO7s09b6a2Vnfy/UT4ffFvRvHk9/pE+nal4X8W2sazXWl3xQyiMnAlidCVljzxuU8HAIBOK9WypKg4r4ut/l+MPwgNgANZ+03yybfvfYfsr+bu/wBjeLb23bO+K+0A2RnHzDtmvlM5wEaFd04u6PkPEXhmhleOjSwz9ycVJJ6tatNX67XXXXruf//Q/uHooor+Tz4E/kx/4I9f8p7f+C4n+/pv/pW1f1nnvX8mH/BHr/lPZ/wXFP8At6b/AOlbV/Wd9SCa+g4n/wB6j/gh/wCko6ce/fXovyKt5/x63Xb9238jXzl/wS7BP7GngDPT+2fEf/p6vK+jrv8A487z/rk38jXzj/wS9z/wxn4B6/8AIZ8R/wDp5vK9Tw5d83/7hy/9Kga5N/vK/wAL/OJ+g+OMAUg7ZJzRxxnBNJ0wfmav3w+wFIGMHgUnHK84pT93HJpcj3/KgD//0f79+Pwpo4LHBIpSwHB5pT7YzQAYz15pvf1+Wl5z0wKXnGcc0AIcHI4zQBj3NLnkjHSjtnGDQB5b8aRn4O/FLuP+Ee1D/wBEPXwT+wzx+yX8Dj/1B/8A2tJX3t8aT/xZ34o5/wChe1D/ANEPXwR+w0cfsl/A7r/yB/8A2tJX5zxz/Ep+j/NHzmefHH5n1gxIjY98V/Ir/wAG7g+0f8FHv+C4t91P/Cd2kef+4jqn/wATX9dT/dr+Rb/g2zQ3/wC2d/wWw8Rbs+f8SbWLd64vdVb+tfP4H/da/pH/ANKR5+Gf7qfy/M//0v7XScDvx+tKxAzk4A60LnJ7DNfHf7eHjPxF4N/Zp8Vr4W1O70HV9c1XRPCQ1OBykmlRanqdtYy3KsPuskdzIVbs201+C0aTnNQW7dj4XD0HUqRpx3k0vvP51v8Ag4O0nxD+2R49/Yi0L9lrwp4v+Pj/AA4+Irar41n8MaXNe22iwpJBvDXCL5cki+W4MaMzAjBANf1J/Cb48fCz48abq2rfDHxXba8tlP8AZ9Rs5IpLa90yUjIjurWULLCxHIDqMjpmvxQ/bm/aL8bf8E+Lf4M+HPgXF4Dm8DaxpL+F7fwzd2zRx+D3VwF8S3EkMbObCDzB9pD4B+Vg2S1dR+2T8VPFP7Dngj4N/tl+DPCWv/tI/Guyt4/CmuWfhm1SC6+JVnPaySIHjhUhkhnjFxGQrmJGm2/fbP2eMyjnw8KcHrG9vO+9z9czrw9hSwcvZVG6lJXd0rO+un6H69/H39ob4M/su/C/xF8ZPjz8QPD/AMNvh1pkZe61DUJggZsHEcS/eklbB2ooLHsKyP2W/wBpH4cfte/AL4a/tIfCSbVZfhx4ss5L7Smv4RDcGJJ5IT5kYZtjboW4yccV/kpf8FMP+ClP7Xv/AAUQ+Ml74h/aW1fWPDWk6VcSxaL4FiE1tp/hhCeVW2fGZyAA0rje2OwwK/fz/gkZ/wAFBf8Agt74N/Yf+G/w1/Yq/YY8J/tC/ATwtd6hpllr83MzytcvcSQtm5j5RrggYXoRWGK4SlTw6nzLnb6tJJevc/NqmUONPmb1/A/0MD0GaM8ntX8eif8ABwN/wUZ/ZI8UeENR/wCCnn/BM/xF8IPghqd/Fp9x4n0VZkOns5xvG9pIpSOvll0YgHBzgV/Wz8O/iB4Q+K3gXwl8S/h/rtl4m8E67p8GqaXf27Zju7aVA6OPTII46jpXzeMy6rQSc1o9mmmvwPOrYWcNXsdlTQTtHGaUZwM9aauFGTXGYH//0/6aP+CmXw+8bfEr9jj4jaF4G0a88Uaha6joWu6jottnzdf0mx1W1u76xQD7zTW1vOgT+MkL/FWr4W+Ovw38a/B5vjN8Mrq7+IXgqLT3vIbbQYPtF4/lrk2q2wIYXCkFDC2GDDBxX3DkMcYyK+FviN/wT1+BPi/xxrHxU+H2pfEr9m74pak4l1bW/hxrb6L/AG5KOkmo2ahrS8k/6aTRM+P4q/HMpzSFGPs6i0vfQOB+NY5Vz06sLxl1W6/zPgT9ge5+I+u/DH9oL4YN4N+N37O3jTUfG/iPxTpWva94WRIorS81AyQtEszGOSXYcmNugJ9K9A/bBvR4f/Zc1L9mrxl48f46ftDeN4zonhazFjDbajq9+8ytFcraQcRRWpCzPNgKgiyTnFfhDqP/AAUO8b/Br/gpb+2f+y1+2l/wUW+L/wAM/wBk34b6G15pWq6dYabBr+r3ha12WqSw2bSSysJ5cCNQflzkAE1e+DP/AAcC/sD/AAH8SeI/GP7NP/BPv9tb47eI9RHkaj8SfEtwupa/rMQPCSX073Eqw55EKusY4+QV9hXlUfvUoOV0mui183+R95j+P6ccM6WFpylJq13ZLXyu7s/tZ8L2OoaX4a8PaZq1yb7VLexgguZ/+e0qxqrP+JBNbwr+Rz/iK/8ABq8S/wDBNX9sRfwh4/8AIdPX/g7D+HQx5v8AwTl/bIj+kcB/9lr46XDmM/k/Ff5n4lLL638p/VF8QPih8O/hTpul6z8SvGvhnwLo97fw6Va3eq3aW0Mt3KSI4RI5ChmIIAJ5NdzG8c0cckMiTIwDKynIIPTBr/O8/wCC0/8AwXq8Ef8ABQH9ivUv2fPDn7Jf7RHwX1SbxDp2qDWfEkcS2iCBmPl5Xne27j6V8Kf8Exv+Djb9s39hT/hG/hf8RLnUP2n/ANnyB47ePQtbu3fVNKhzgJp9625lUD7sLhk6Aba9CnwjiZ0PaLSV9n/mbxymo4cy37H+n98QPiH4G+FXhXU/G/xG8U6L4O8J2gHn3t9MI41JOFQZ5Z2JAVRlmJAAJr+Sv4Na1L4Y/wCDij9oT9uDxl4Z8deBv2TvE/gu30XRvHWt6Fd2Ol3N2LCwh2vLLGvlKXglAeQKp25zgiv1X+C/x98Ff8FGvi34T+MEfhD4ieFfAnhbwZp+veGvDnjPQ5LG407W9Qe4R9Re0mGyWWGO3MUM67lAllKN8xryf9lb9rb4yftD/tN+PP2XPHerfDW+8NfD1dTtvEGu2toslv8AFaNpPKh/s+Fxtjjtg/lXYUuBOmwHaTXRlWU+zpyc95Kz8k/10P0DhngSFfCxrV6ji6t1FJJ7Pr933eZ//9T+1i0u7e/tbe9srm3vLKWNZYZoXDpKhGQysOCCCCCOtTfL/WvzM/Zd8ZeLPhR4Z/aZ+DngTwLq3xQ0r4feOY9M8LaZDqdvZi00m7021v1svtFy6xqtu9xOiKTwjQrwBkff3gDxRrfjLwxYa/r/AID8SfDXUJy2dI1ea1lu7dRwDKbWWaEE9cLI3GOe1fhWKwzpTcX0Pjsyy+eFrzw894tr7jsxnqc5r5Z/YKAPxx/bszx/xWtt/wCki19TdcgcHNfLX7BH/Jcv27f+x1tv/SRa9Thj/fofP8h5V/vEfn+R+oBAIwabgHuRml7ngmgE56EDFfsJ9gKAB0xSDoO1GcjgZpDtG7j60APpu1fSgbs88ikGP7h/KgD/1f79Tz/+zSHODlaMDkk8+1BUDJJJFAAxb/gNGHz1pHwe+aOpABz9aAPzY/bM/wCTmP2HP+w5rH/pC9fSZYZbrxXzX+2X/wAnMfsOc5/4nmsf+kL19Jkg9+tfkHFH+/T+X5I+OzX/AHiXy/IT72MrxS8MeCaNw9/pSYAyTjrXgHCBIOM9OtLuHTkmk49ecUp+7xzQB//W/tezxkgihDgjOccUnynrjnig/KM8k1+AnwB/Jx/wb+fJ/wAFFP8AguDD0C/FSc/+VC+r+snj7x4Nfya/8EBjs/4KT/8ABchOgHxQmP8A5Ub+v6yuO4Ga9jPv95fpH/0lHVj1+8b8l+Q3n1AP86cCD0pu7pkYNKDnB4xXjnKfnv8AssqrfHj/AIKCq6gofifagj1H9gaZX5v2f7PXjn4XfCLxj8KNN/ZM8XXEt18XfEGp+I9Yt9KTWEvtGur/AFG6sLvT7RL6EXWBPbROshUw7yxRtuR+i3ij9h74mP8AFn4vfE74R/trfGr4IWfjTVYda1TQ9N8P6DfWsd5HaQ2u+KS8s5ZQGS2jJUsRnOKg/wCGN/2qMZH/AAU2/aG/8Ivwn/8AK6vzXMeDsfPG1cRQqQ5aji7NzT91WW0bdX3Wx+mYDirB08PTpzveKtsfn74L/ZF+PPxS+G3ws0n4/eCfibcXmifB7xppVva3GuSW0lnrj6pnSd4t7kg3S2yRmN97+X/fzzXR6P8Asm/GrVLD4lfGfxV4H8bN+0Qb34cXWi339rNHIr2tjp8eqSJGkwi3bhdxyswy6pt+YAVwfhP47ar4y/4KGeO/+Ccmi/8ABVH9oa4+MOgeHItZnuv+EP8ACP2We7PzS6fGf7OyZ4omilb/AHmHVTX6N/8ADG/7U55H/BTb9oY/9yX4T/8AldUV+Cs6g/enSjzapXnazlzWXu7O1n3R2S4owMbc3N93kf/X/o5+Po/42B/8E3v+vf4if+mm1r9Jc9GGFwa/Pn4d/sP+OtC+PPwt+PXxc/bA+Mfx+1TwbaavbaDper6Jomn2lu+oQpDPK5sbSGRzsiUAFsA81+gzEAfLtFfzJw1lE8BgKWEqSUpRvdq9tZN9Uns+x87xJmNPFYp1qW1luG7PZqXp6mm8ZxxgilLAcE5Ne6eEOpOvOMGkBHyjvijIJA/GgB1NXp0xQDwedxoABA70AAwDjJ+lLww9RTWxjseaXK/L+lAH/9D+175f60bhnHekGM56DFLjGAGxX4CfAHxz8fv+Tkv2Gv8Asbr/AP8ASSv1bHTJ/nX5SfH7/k5L9hv/ALG6/wD/AEkr9XB90fSv0ngn/d5/4v0R9Jkn8OXr+gvzY77qaFJHcf1pdu1lxSjnnufevtT2xhLAYPSpNpzncaG298U0j+EHA/nQAfPxnmg55++aduU8ZpOMYz+tAH//0f79PnHoaMkAdBzzQMc849Oaceq9qADnOc/LTf3lLnA4IPPrSBeT29s0AflT8Bsj9uD/AIKI85/4mfhT/wBNRr7q7CvhL4Cg/wDDcH/BRHJJ/wCJr4Vxnt/xKjX3ZkHABr+Vcb/vVb/r5U/9Lkfn8/il/il/6UzgW+GHw9f4iR/F2Xwjok3xMj01tHi1uSEPdQWRcO0EbnOxGZVLBcbioznAr43/AG6vhX8NfA/7Ev8AwUG8VeDfAXhHwt4m8Q/DTxTfa5f2GnxQXGr3H9lTjzbiRVBkfgctk1+gpwOpP518c/8ABRMBv2Bv21QGGf8AhVfin/013FbYKtL20Ffql8rmlOT5l8j82P8Ag2cz/wAOcP2XumPO1v8A9OdxX73N+Vfgj/wbOqU/4I5fsvqRtbztbyD/ANhO4r97flyMYrs4hf8At1b/ABP8zTHfxZep/9L+4dfUjFfPvxi/Zz+Hfxk8U/DDx94pufGum+JPBlxd3ejzaLrlxp28ToqzQz+QymSJxGm5MjIBBOGYH6D71WuiPst1gg/u2/lX8o0qjhJSi7M+BjJp6H8mX/BsLbQX/wASf+CvHiq7jFx4gufjK1vNdOMySRq90wUt1xl2OPev61e9fyZf8Gvfy+Kf+Cte4Fc/G2TGe/8Ax81/WZkZ6ivouLP9/n8v/SUdWY/xX/XQ+Wk/Yv8A2aJf2hPEH7U2ufCrw94x+Pd/bwWMfiLW0+3T6TaRKFSCwWXclrHnLHygrMzMWJr8/v2hrL4g/t6fGP41/DP4Dsmg6Z8F3TSjr8ni/XdIj1bxXNbx3L6dGNKurcCOKF4Y5ZpfN2mZ0EfUn9pxxyQAa/Ln4Xfsn/tR/Afx/wDGLTvg/wDFH4R6d8H/AB18Y7n4p6zqeoWNzN4gt7W7khlvdKhi2m2O9oDElwxykUhG3equODA4n3nOcveivdvfTVL8FfQKU9G29Vtft/Vl8z5d/wCCtXxr+IH/AATJ/YX179q/4Oaf4cl8Y2V5omn6n4X1JprzR5rm4dYppoXLidGyCchsOfmZdxLH9Ev2R/Dt78SvhN8Dv2iviLqv9veMde8Kabr1tZQwCCw0WW7tI5ZBBFlmd/3hUSSMzADA25Of5e/+Cwni34lePf8Agjd+294p+JPxb+JXjzWrj42RKmga3oH2GDwnbR61Lb2trbSmCMyA21rbysis6qZd3Bk5/qv/AGFtv/DFP7JHOD/wrXw3/wCm6CvczOVSGXQi5Xbk099rRdvRXPq8Vxhmn1L6rKs3HWPS7iul7Xt031Wj00PqvIPQ5pCM8EAijIyORS18cfFH/9P+w747/s7/AA+/aJ0fwVo/j2fxfpg8Pa9D4j0m60HWJ9Mura8jilhyJ4CrhWjuZkIBBw+QVYAj+Zn9oeyt9S/4Ovv2NtM1FXvrDSPg0JNPSdjJ9nYLqfzAsSd3zE7jySc1/W8SCSMg/jX8lnx5BP8AwdmfstnBKf8AClyM9vu6lX86cOVXerG+ipz/ACPjcFN+8n0TP60yCcjkD1FfLPwQ/Yt/Zr/Z88WePfiL8N/hfocPxT8Varc654l8W6gDe61rV7PIZJHlvJdzqmWIWJCsaKAqqoAFfU2RnqKXgYAr52nWlGLjF2T38zgjNpNIrXFvDd29xaXMcdzbyoY5Y3XKupGCCO4IOK/nm/4Kk/tS/Fb/AIJ2/FD9hX4ffBQ+Gdf8H/Fv4jQ+DLu08Q20lwfDkEksCl7SSN0ZgPPbCSFsYABxgD+iEZIyRj1r+VL/AIOQFx+0F/wRXAH/ADXizz/4E2VfQ8KTaxkYp6NP8E2fS8N8Q4zA1GsLUcVLddHbbR9fPfdbNn9Knw8+E2m+BLvUdfvtVvPFfjK8RYbnVLlFQrCDkQwRL8sUQPOByx5YsQMes8gA55oIORkbsd6X73XpXz1WrKcnKbu2eTmeaYjGV5YjEycpvq/6skuiWi6CAg9KUkDrRj8qKg4z/9T64/4I88f8F7P+C4n+/pv/AKVtX9aB61/Jf/wR5A/4f2f8FxB/t6b/AOlbV/WgetfzdxN/vMf8EP8A0lHxeP8AjXovyKt7/wAed1/1zb+VfOP/AAS94/Y08AnOf+Jz4jP/AJWbyvo28z9juu37ps/lXzh/wS8Uj9jXwGd7tnWfEfHHH/E5vOlep4c/8jf/ALhy/wDSoGmTf7yv8L/OJ+hP3h1OKYuRuHenk47E0tfvp9gNLYxmgk47D60YBXA6UuOMDigBuTjjqP1o5PQn1pdq53UYbGN36UAf/9X+/U54xkdqMnc3BIpxxg56UgBB4+7QAnLA5+UULnHfpTuvBoXoOc0AeWfGzP8Awp74pZ/6F+//APRD18GfsM/8mk/A3/sD/wDtaSvvX42/8ke+KXr/AMI/f/8Aoh6+B/2Gv+TSfgb1/wCQOf8A0dJX5xxz/Fp+j/NHzeefHH0Z9YNyG61/JB/wbAr9t+K3/BXrXxkm5+LUUe71Al1Bv/Zq/rbnfyopmHGEJ/Sv5Lv+DVKM3/h//gpp4nwWF58ZGTPrtjnb/wBqV4OA/wB0rv8Aw/mcGH/gz+X5n9axYdzzXmnxj+F3hv42/DDxv8KvFf2lNE1uxe0kmgOJbV8ho54m7SRyKkinsyCvSsndgUnzZOBxXjxk07rdHJGTi7rc/9b9LP2jv2svA/7DHxt+E/ww/bM/Z8m+L/xw+KVmnw+8KeKvCItbxPGVmJ1iW0vLW9lhFgzyXSGRAWiZnJ3kDA/Rn9nz9mnx7L4s+GvjX4oeD/D/AMH/AIeeBbKW18A/DzTtQ/tF9Mlkj8o3moXYJRpUhLRRQxFkjDud7krt/DD/AIL9J/af/BU3/gh3oeN2fH8Nxj1xqtl/8RX9dTZDsTwtfk+ZZhU+rU7bzTv99jnzXjTH18JGhUkrSWrSSb16v8+5+c/7Zv8AwSh/YK/b00a8sv2hf2f/AAnqniZ0It/E+kxnTdas3PRlvINrvjrsk3oe6mvhX/gnf8FNF/4JTH4+fsR/Avwt8Z/2jNKtZ5PiLpYuktba5hspJIrSS2DEqskjSo3lHCiQJIx2hcn+gIDnJVa+fPD/AMOPCXgb44/Fr4yav40sZvEXiix0rTo7S6kjh/suys0kCxoS2WDyTSyE4HJx2zXj0cdP2bpTbce3n+h8hCu+Vwk7rseAf8FHfhL8Pv2q/wBhj9rv4EeIV0fVNWfwFfX76e8kclzpF6ltJc2czICTGwlt1Ktxna2CRX53/wDBsJ8XNZ+Kv/BJH4RWOvXb3t/4T8Qa14TRn5YW8M6zRKT/ALKXQX6KK+xPBvgqL4e6R/wUR8eeNF0mw1fxfc69f2N4/iGO5jl0eHTFtrKCOPzWK/LDLMQQu1p2UDAr8r/+DTDXtD0z/gmL4tg1PW9K06c/FTXXWOe5SNtv2ax5AY5xwa71G2CqRWtpRa+adzdRXsJJdGj+pzIHRT+VG7HUEVjWfiPw/qM621hrujX9ywJEcN1G7n6AHNbPGOOSK8I4Dy74z+KPG/g7wBrOv/D7TfBmp+JINrL/AMJDqT2OnwR5+eSaWNJH4GQFVSWYgcda5TwZ8am/sL4eS/F2w0j4b+L/ABTfnT9E0dLiS5kuZBC0uCTGpUlYpXwyrtXaGwxIrZ+Mvwf8N/F7RLez8TeKvG3hW3shLPa3ei6sbB7G5K4S7DAFWki5KCQNGCclTxj5X1H9n7wzpfxV/Zo8TaP4v8MePND8Fwau15qPivXFu9Uur+7e3/055ePMlCJdYA2KrSLgBRgdNGEHGz3/AOAawjFqzP/X+XtL/Ym+GH7dv/B0X+0t4D+MOj2/ir4XeGp5vFuraTKx8nVfs1tapHbzY6xmWZCy9wpB4Nf3l+G/hx8O/B2k2Xh/wj4E8H+F9Et4xFBaafpkNvBCg4CpGihQB6AV/H3/AME+Nd0XTf8Ag6G/4KO3N1q2mR2N34W1Aw3D3CCNv3mnHhycHoa/sV/4TDwlxnxR4d/8Dov/AIqvyLiCrNypwvoox/I+RzGcrxXSyLh8PaAw/wCQJpBH/Xsn+FMPhvw62C+gaKe3/Hqn+FaFreWl/bpdWN1bXlq2dssLh1b6MDjrVrk8HHvXz1zz7s/GD/gtr/wT0+JH/BQr9j/Sf2dfgNbeBPDfiq78ZaXqF5qepYt4NPsYhKZZW2KXcjcuEUEkkdOo8Q/4Jpf8G5/7E37BP9jfELxvo8H7S37Q0Kq6+IfEVsGsdJk7/YNPJMSHPSWQPIMcFea/oPXGBjpXml38Z/g7ZXM9ne/Fv4ZWt5E7RyRS6/aK8Tg4KspkyCCCCDXo0szxKo+wg7R8jeOJqcns09D5u/aO+Avj/U/F+kfGz4Fw6Ff+NINEl8M6/wCGdQu3srXxVozNvWKO7jBNrdwuXMUuNuJHVsAhl/JbwJ+1R8AtX/al8IfsEfA/9jPxl4T/AGzvhhptxf6Xo2uX1pp+meErW4gR5pptTtrqc3UMqXSSMirK0jOGZQ4yP31/4Xh8E/8AosXwq9f+Ris//jlfyZfAX4h/D+D/AIOtP2uvFk/jrwbD4Wm+HFtHDqb6pAtpK40vTAVWctsJyCMA9QfSvWybGVfZzg1pGLa+9f5n1/D/ABdjsJQlQptOKTaur21W35+p+wX7V/wP1b4afsN+IPgdPdf8LD+Nvxg8fabY6xqFsgga81bUbuBLme23n9ylvZ2xSM5yiW6nrmv0D+EXx58PeIdL+Mmn+ILDTPAWm/DjV28PapeyX4k09Ehs4ZzItw4TCokyq24DBU1h/EW5/Z7+Jfjv4J+ONb+PPw+t28Eavda5ZWEXiGw8i+u5bSS2VpsuT+7WaRl2kfMea+TfH37PXw21z4S/GX4W/Cn4teHPi/rfjrxcmt3ekf8ACR2gnFpPqa3V+kQjkHmOYWkjDN/BHGv8JJ8tyVRWqbt3v6v/ACR8tVrSqtyqO8m7t+b3P//Q/tYhnguoYLm2lWe3kUSRupyHUjIII6gg18vfsFf8lw/bt5x/xWtr/wCki19E+FrFtO8NeH7BtLh0RobSKP7HHMZVtMKP3YkP3tvTPfFfO37BIz8cf27R/wBTra/+ki1+P8M/7/D5/kfHZV/vEfn+R+oByBwMmk5wNtAHAz1o4z15+tfsB9iKOnUmm9AOefen00DGegFACAdGJ5p9IRwcUDoKAGsQO5Bpfu7ey0fKMdKaQABjtQB//9H+/Q8En8qOcMv3qMjt9M0cgliKAPzX/bL/AOTmP2HP+w5rH/pC9fSJODglq+bv2y/+TmP2HP8AsOax/wCkL19IjLdVwK/IeJ/9+n8vyR8bmf8AHl8vyQ7I9RSDp3H1oJOQO1VLy+sdPQS395aWUfHzTShB+uK+fOItZU4UdKd0ya5pvGHhBeG8V+G1PcG/i4/8epg8a+DVJH/CXeFwP+wjD/8AFU7AdPx25zxSYzuGf/rVyx8d+CMZPjLwoB76jD/8VTT4+8Cj5j438Ir651OD/wCKo5WOzP/S+mv+CFCvoH/BVr/gud4Pu8LqS/ENr0qeCEbULzBx/wADFf1i465JNfxmftKfG3QP+CO//BdnV/2xvEs1tqP7Gn7QmjxaV4s1PSpUuRoOrIkYad0QnJWWBJiOpjnlIyVwf6qPhl+1x+y78ZvD9j4q+F/7QXwg8baDcRLLFPZeILZ/lIz8y79ynnoQDX45ndKc5RrpXjKK+9KzR8fjoSk1Uto0j6Lznk54pQSMZ9K8+T4sfC2RgsfxL+H7ueNo1m2P5fPVn/hZnw47/ELwMP8AuLW//wAXXi+z8jhsdxxkDg/0r83f+Cq//BQDwX/wTe/Y2+Jfx+1yW1vvG7Wz6R4N0pnAbVdbmRhbgjqYo2/eyEfwI3civuiT4pfDKJC8nxG8CRqOpbWLfA/8fr5L/aW+On/BN7/hG0uv2rviF+yP4i8P6ZvuYYvF9xpepLaMV5aKGbzCGIGPlXJ6V0YOH7xOcW12XUqjD3ldXR/kafDn9sD46fDH9q3Qv2ztA8YXs3xysvE58VvqUzlvtl00peVJRn5opAzoy9NrEV/sF/sSftZ/D79uP9mH4S/tL/DaZBoniPTI7i4tPNDyaXeqNs9rIR/FHIGXtkYPev5L/wDgl98KP2XP2/f+C3P7b/7SPwh/Zy+Ffiv9hPR/DsGh6THe+DLNdDl1ERW0KyW1m8Xkq7m3uJchQ21txxur+z74cfCf4XfBvw+3hL4RfDfwJ8LvChna5OmeHtJg061MzYDSeTAqpvOBlsZOBX0/FmNpVHCHLaaS+5rZ+h6ub1ou0bWa/qx3xOMcE0p98Yo65waOQPU18eeOf//T/tgopuDjG406vwE+AEAwBSAejEinUmCRycH2oACAetIOB0xS4BOe9ISRnpn60AKOgz1owM570fTmjAznHNAC009QcEmnUUAf/9T+pj4/f8nJfsN/9jdf/wDpJX6tjtzx0r8pPj9/ycl+w32/4q6//wDSSv1bUcd+lfFcE/7vP/F+iPEyT+HL1/QcSMZ49qOPn6+9HB2jp7UDHIyTxX2p7YpOCQRxTSAe5GaG7jv/ADpcc8E8UAIAASNxzS9+cZ/lTABtY5pzL6DigBRjdk5DUAYUdAfWmg9yNv5Upx8rdqAP/9X+/Thc/Mc0K3+8TQcL/eA9qYxJPNAH5V/AUH/ht/8A4KI4/wCgp4V6/wDYKNfdfYV8J/AYKP23/wDgogMf8xTwp/6ajX3ZX8q43/eq3/Xyp/6XI/Pp/FL/ABS/9KZw3ji3ju08K2c6tJazaxCkqbiA67JDg47ZAqj4q8IfC+y8NeIrzxrp3hy28Hw2U0mqSam4FoloEJlM5kOzywgbdu4xnPFXPHFzb2Q8JXd5PHbW0esws7ucKg8uQZJ7da80/aE0/wCEfxZ+Bvxa+G3xA8faV4a8E63oF5p+qait7FGdPtmiO+fc/wAoCD5vm+XA54zXzsIYaWLmq9r3jva9rLa5+w8Gc31BOK6v9DB+AWu/sj/EXQdX0f8AZv1H4Za74b0K5Fpe2fhsLHb6XPIvmiN4owqxuysHxgEhg3Qg17BZaVYaP48gttMt/slvJpMruisSGYTIASCeuCfzr5u/ZN0f4V+EtF+I/wASdD/aE8KfHXVPHWuR63qXiHTzawWU8kFlb2EcdvFbu0ahY7NAx3Eli2T0A+kbXV9N1fx/BJpeoWmoRR6RKHaFwwQmaPAOPoavMaeDhWUaFuZSXa/n0T/A7s955ZfVlNdP1Rt+NQR4U10K23/R2HBwRVeTwf4NtrKS6u9Msre1jiMkskkhVUQDJZiTgADJzU3jdgnhPXnZsILZiT6CsHxbe+A/GXgzxJ4O1bxLpsWl6rplxpty0V0qusU0TRsVPY4Y4NGIjh3in7e2y3t3e1zxOA4yeGm4r7X6I//W/rH+AWufsReONa8baL+zdefBjV9chaLUPEEHhdYkkJl3eXPceUBv34YrIc7hkgkV77q3h7RtF1XwncaXYpZTtqHlsyM3zKYnyDk9OBXxb+x14U+CPgbxD8QfE/hr9qHwb8e/FUWjaH4Gu5tPeygj0Ww0tbgWttNFbOw+0Zurgu74LcAKoXFfaer+INC1TVPCUGm6tp9/ONRDFIpVYgeVJzgdua/jjN6GCpz5adrpx7X3XkmfdYxVJYSpKa+zL8md9KcRzduD/KvFbl/h94L+FFz8SPGsCW2iaZoj6xql2d7skMcRkkcgHJOFY4HWvapuYpcc/Kf5V4zJceE/FPwluPA2p+JrDS01HQpNLuHEqeZb+ZCY2O0n7w3E4PpTxkaH1mH1i3LZ727x7nxnAKbjW5d/d/8Abj541L46/wDBPn4qeGbnTfGHjf4KeIPD40FPFl9p3iCSN4rLT1tUvPPuYp8pGY4JY5mVvmRGVyACDVO3/a7/AGEdD1v4c+CdO+Inw6stD1uxvf7Evre7WPTQbO4tbZ7QMGAjkDXtvhcBQpByAVz8N/Er/gn18HPDthrni740/t1/2V4Jl0a88OvLrS29tZ2cV5osWksieZceQgxCsqgIGLMyszjGPp/xL+xn8DfidcfFDxjp/wAboGsfG+kapaxTWaW0kVrBqB0hzLbuGwwA0WEjswmPoK9qWFyZRUrrlfWytf7j7hPEv3eU+/fEfhjQ9Js7S907TktbtL61CurtkZmUEdfQ16N9cYrzrxP4n8P6lZWlrY61pt5dvfWu2OOZWZv3yngD8a9FBrxMv9l7aao2taO1rdex+fcfRaqUubez/M818MeGND1bTr2/1LT0u7t9Qvg0juxJAuZFA69gAPwrwPXPEP7DmnfHvRPCmvaj8FI/2j3dNJsUmET6yrtGZVtUlwZFcxs0nlbgdrbsYOa998K+JvD+nabeWd9rWm2d2mo32+OSZVZc3MhGQfYg18ID4dfADSP2qbU3P7VHg+DWbjxfN8ULT4eySWS6hJq0mnGxedZt/ntbGONm8vbwwPzbRtCyShgZ0V7Tluo3e19tb6P5/mfokfaKnDlXb8j/1/7SPEfg7w3a+HdcuYNKiinjs5nR1dgVYISCOeoNdlpTFtN09iST5EZz6/KK43xN4v8AC8/hzXoItf0mWV7KZVVZ1JYlDgAV2OlDGl6dnj/R4/8A0EV/G2D9j7eXsbWstrd32Ovj6LVOlfvL9C+Otfyof8HIP/Jwf/BFb/svNn/6UWVf1XjrX8qH/ByD/wAnB/8ABFb/ALLzZ/8ApRZV9zwt/v0fSX/pLPz/AAH8ZfP8j+q0gZIwCaWlPU46UlfPnMFFFFBHs0fyY/8ABHr/AJT2/wDBcQf7em/+lbV/Wfiv5L/+CPP/ACns/wCC4hzkb9N/9K2r+tDNfQ8T/wC8L/BD/wBJR2ZhbnXovyP/0P7gLz/jxuv+ubfyNfN//BLwf8YaeAck/wDIZ8R/h/xOryvo69/487r/AK5t/KvnH/gl2B/wxp4A/wCwz4j/APT1eV+B+HKtm/8A3Dl/6VA+Pyb/AHlf4X+cT9BiBg4ApTyP880hzjI61806P+0x4X1j9p/xx+yzF4Z8WWnifQPCun+K7vW5hbDS3ivLiWGC2RhMZvtLG3nbY0ajbGTk5FfvqXQ+w6XPpY8c4o45HJpDwnBNIeQQOn04pAP9MAZpAeBnmgDaOTxR82euBQA6mnAIJzSfxdRn6U7v3oA//9H+/cYAOOlNXBLGl68cH1oDAjrg0AeX/Gr/AJI78U8DI/4R7UP/AEQ9fBH7DP8AyaV8Dc/9Af8A9rSV97/GrP8Awp34pcD/AJF/UP8A0Q9fA/7DP/JpPwP/AOwOf/R0lfnPHP8AEp+j/NHzmefHH5n1JqMgh03UJWGAsEjE+mFNfyjf8GlsBk/Zp/bb1zgi9+NF6Q397baRH/2ev6pPGM5tfCHiu5Bx5emXUn5RMf6V/Lr/AMGjlv5/7AHx71//AJ/vjFqzZ9cWFif/AGevn8HK2Cresf1OChL/AGefyP6q88nA5oHI6Yo67hmkxyxPSvHOI/ki/wCC2w/tL/gtl/wQ10RmVh/wkSXOPpqcX/xFf1wKcjNfyRf8FcANV/4OD/8AgiVoeNwiJu8fS+nbP/kKv62sPwMivYzRfuKC/uv82duK/hwXl+p//9L+18HkjGK/K2x/Z3+A/wAbf2zv2wtR+L/we+G/xN1Cxi8KW9lNrujwXr2sR052ZI2kUlVJ5IHev1RAJ+Y8e1fnp8OvFvhTw7+2L+2jB4g8T+HdCmkHhN40vb2OBnX+zW5UOQSPev5R47q1IZTVdJtS93bf4l2PL4NSeOSfZny5+0Z4I/Yg+CvxJ+G/wN8Df8E7vht8f/jj4psbzVrHw7onhzSbVbXTbYqJbu7urkJFDFudUXJJZjgCuF/Yb0r9j79sHW/jv4c1P/glr8PP2ebzwBrx8NatFr2kaJcPJqYUO8Kx2wLKBG8UgkYbXVxtJwa9h/ah0fx/oX7Tnwl/bI/Zi1T4E/FrxXpHhe/8D+IvB+veNIdFOo6ZPOlxHcWd9smRJ4pk5SRQHRiNwIFdJ+wh4DufhAn7RHxe+PfxG+C9h8bvit4zbxdrmkaB4iiutO8PQx20Vra2UVy/lmdkihBeXYoZ3OBgAn8hliuXLudTbqNKz55c3Nzarl5rJcuza+d3p+pqF6lraeita3+Y34n/ALNX7PfwV/aD/YX8SfCL4K/DH4Z+ILn4j3NjcXuh6LBZzTW50LUmMTPGoJQsinaeMqPSv1mBJzkV+cn7Q3jLwh4h+OX7BtjoHivw1rl4vxOuJGhs76Kd1T+wNTG4qjE4yRz71+jR5XANfrPh/VqzyqnKq23eW++/mfl/GySxiS7I+JP+CkFvBdfsLftOWdzEk1rN4ZlhljYZWRGkjVlI7ggkEehrCb9hz9hrS/D51jVf2WP2ebPT7ayNzczyeFrMLFGke5nY7OgAJJrof+CjrpF+w5+0tLNIsUa+G3ZnY4CgSxkknsK9D17xV8JPGvgLWvBmsfEbwSml6to82l3WzWLcMIpoDG+Dv64Y1854pYmvCOGVKUopud7X/udj3eA4xdKpddUfgD42/aJ/Y88O/Djwd8fvB/8AwRetPHHwE8Ta/b+H/D3i4aX4dsF1N7icwW9w1vMfNhtpZAAksgAO5c4yK/abSv2Ef2LrzT9Murz9kD4CaVeywxyzWsnhaxZrZ2UFo2KpglSSpI4JFfkbpHwF/ai8S/An4L/8E9viFqf7K2k/s3+D9d0pr74mWXj5Jr/xBoGnXa3Ftbw6N5KmC8k8qFJHaUxqAxUsSBX7+f8AC0/ho8rO3xG8Cbmbcf8Aib2/XP8Av18NxNipU1COFk73lqpyleOnI3q0m9bqya6pbH2mGgnrNdui36n/0/6Q/wDgl5oej+Fv2fPiX4Y8N6XZaH4d034zfEmw0+xtYhHBZW0fie/WOKJBwiKoACjgAV+jYA+9jBr88v8AgmdeWt98Dvi9eWN1Be2cvxt+JskUsTh0lQ+Kb8hlYcEEdxX6GYfAGa/nujd04uW9l+R8bmf+8VLd3+Y1m/dv2O01+EX7D37MH7LWp/sGeG/iz4v/AGXfgx8SPGy2niLVrq4vvDNlc3mqTR6hesA8sibmZtgXJP41+7jAhHyedpr8YP8Agnf+0B+zlpH7EPw68B+Nvjj8JvDmqY12x1CwvfEVpb3Fvv1O8DK8buGVsN3FfDeJE60cvi6N/wCJG9r7Wl2PreBFH21Tm7fqeU+CtW/Yfm/ZW+EH7SXxN/4JyfDDwbH4k0iz1Ce2/wCEW0GGCLzLVJ2khluJUDxHeQgJ8xsfdrpvFPiD/gmNotl4d1Xwx+w94K+J2j3/AIDj+Jb3fh/4c6dPFY+Hmdla5mLhfmXy3JjXLkA4BxWXd/BH9lG48K/Bjwuf+CgPwvv7fwDpd74d8Nvqsvh7UEs9Hnhhh8nyZcx/aY0t0CXePMALg5DEV6f4A8CfsP8Aw98FR+CdP/bA+HWoWa/Cf/hUvnz+J9LEhsN87/asKwHn5uW4xt4HFfm9atSXvqVRu70XOla7tv2Vv6R+hpS8vwPnix8SfsK6H8avj34c8YfsQfB3V/g3oF54SGneJ9H+HNi9ppVnq9lFIk2oTMASGmnUARqSiHLDHNfTfx6/Zm/Z4+DXx8/YA8S/Cj4G/Cn4aeJJfi4LOa+0PQLaynkgOhaqWiaSJFYoSikrnBKj0rnZvhp+wzN8PfjX8O2/bD+Hn9neN7fwxb3048T6Z5loNHt7eCExfNjMgtULZzyxxiu//aO+OnwY+J3x1/4J/wCi/D34s/Dfxzra/F8XUlnpGtW93MkQ0LVQXKROzBQWUZ6cj1r0sgrzlmmH9jz2vaV+aztBa693zX+R5+bRSwdXmts+x+sxyMcZ/Gvlr9gj/kuP7dv/AGOtr/6SLX1IVPt1zXy1+wTx8cf27P8AsdrX/wBJFr+kuGP9+h8/yPyDK/8AeI/P8j//1P79+FHoKQY4H5UYHQgH8KB6Zy1ACjoOMUndvlpRjGARTQM85yPpQA7AAx2puE9vzpcqFHpRsX0oABtPQCmNkbc807b3BOe1GCf4mFAARjgDP9Kb0/vbqVsbud2KGX7qigD/1f6zP2yyf+GmP2HB/wBRzWP/AEgevpHAIXnnFfNv7Zf/ACcz+w5/2HNY/wDSB6+lOMDpivx/ij/fp/L8kfHZp/Hl8vyR/N//AMFwP25f2ofCvxg/ZT/4JqfsK+KYfAP7SHxiuJH1HxVwZfC+j+YIhJEcExs2LhzIBuVYfl5YEfNenf8ABqr4M8eAeIv2nv8AgoH+1h8YviBcfvNQvVvgsckx+8VNw0r4znqaT4zqPFX/AAdl/s56ZqBEttonwlae1U/8s2+xXkmf++pCa/rUDc45z3rStjamEo04UHy80bt21d33LnXlSjGNPS6uz+Utf+DSL9h7aDJ+0N+1dI3c/wBsW3P/AJCqUf8ABpD+wp1Px9/atf8A7jNsP/aNf1Y9/u/jSZ56Z/pXL/b+M/5+Mx+vVv5j+VEf8GkX7Bv8Xx3/AGrD/wBxu25/8g09f+DSL9gjv8cv2rSP+w5bf/Ga/qt4645pox9feo/1hxn/AD8YfXq38x/Khcf8Giv/AAT4vFEd78Yf2o7xFOQJdZtWAPtmClt/+DRT/gnfaDFt8XP2ooFPaPWrVR+kFf1XYGfeg5yABxV/6wYz/n4w+v1f5mf/1vrtf+DSH/gnuOf+FxftVfhr1t/8Yqb/AIhJf+Ce4HPxi/asP/cft/8A4xX9T4wBjFH4V+K/27jP+fjPi/r1b+Y/lkj/AODSb/gnkHG/4tftUzLnlT4hgwf/ACBXrnw9/wCDWD/glP4P1K01PxR4U+L/AMV/KcObbXfFk6wS4PR1t/KYj23V/R9huCpxS5P405Z7jHp7RieYVv5meT/Bb4GfB79nP4f6N8KvgX8N/CPwr+HunLttNJ0WzW3gjPdiBy7nqXYlmPJJr1jvRzSfMMdN1eVKTk7vc5m76sWmjOAD1obt938aU5xx1oEAzjnrScjk5+lLzwfzpM5OAcGgD//X/te/i6/hTqaM/wCzj2pfSvwE+ADtzSdef60uc9CKMjGcjFACEkds0uOc5NA9CRmkz/tLQApGfUUtFMww+7gigD46+Poz+0n+w0P+puv/AP0kr9XARjOK/KP4/f8AJyX7Df8A2N1//wCklfq2AOMEYr9J4J/3ef8Ai/RH0mSfw5ev6H//0P79gQeMEH+VCkYAFNyyhaXr3Bz7UAGcE8HP86CQpxjikxxnIx9KCM85BI/SgAwNq8E0M3UCj+6xbFKMfNg5agBjyJGjPIRtA3EnsK+bf2dP2sfgb+1H4FufiR8G/Fya14Nj1S90pbq5ia2E8tvM0TPGJMFo2KFlbupBwM19GXcP2m3uLfzGj3xsm4dVyCM1+Fnxe1f/AIJ9f8EqP2Qf2cPhh/wUq+Jvgz4hfD7TJ28N+ENW1rwNJd75orcM2YbWOdhO6I8kkzn53ZsYBCgA/Ub4zftVfCv4GeJfgz4U8YN4mvdU8d+JbTwroj6Xpct3ax3k7FYzc3SjyYEypHzsGJ+6rc4+lDtYEEAgiv5B/jr/AMFpP+CInjnRf2f/AAf8EP23PAXwH8E+BPG9h47h022+EmuXMN5eWsryRxBUihEUbNLIXwCxOMEc5+9Yf+Dnz/giekaCb9sVXkCgMy+CtdAY98D7IcD2pRvrfe/4WX63FJ66bW/G7/Sx/9H+pr4CBR+2/wD8FEQFHGp+FP8A01GvuwnAPFfy4fCP/g4R/wCCSvhb9qb9sz4ja1+0+1l4R8V3/h+fQrseEtYb7alvp5imOwWpZNr8fMBnqM19Xj/g5e/4Iw/9HcNn/sTtc/8AkSv5txmQ414itJUZWc5te69nOTT2Ph5YOrzS917y6ebP3gYI4+dVcehGa+NP+CiMMI/YF/bUYRxhh8K/FJBCj/oF3FfnT/xEv/8ABGE4/wCMuD/4R2t//IlfMf7Zv/BxP/wSN+LH7In7UPwu8B/tPXOteNvEXgDX9E0i0/4RHWYhdXlxYTRRR73tQq7ndRuYgDOSRRg+H8Yq0ZSoy0a+y+/oXRwtZSXuv7mfRv8AwbPRRSf8Ecv2YGaON2M+t5JUE/8AITuK/e5Y44zlI0TPHAAr+Kr/AIIdf8Fxv+CZv7Hf/BNf4E/AD9oT9oO48DfFXRJdVbUdMXwzqt2IBLfTSx/vYLd42yjqeGOM1+t4/wCDmT/gjDnP/DWdyf8AuS9c/wDkSuzPcixk8bVnCjJpyevK+/oa4vDVXVk1F2ufvSdrAggEHrmqV3Bb/ZLr9zFny2/hHpX4Qj/g5n/4Iw/9HZ3X/hGa3/8AIlMuP+DmP/gjG9vOqftYXJLIwA/4QvW+uP8Ar0ry/wDV7G31oy/8Bf8Akc8cLW25X9zPiD/g1+SOTxT/AMFaPMjRyPjZJjIzjm66V/WT5ManekUat7KK/gF/4IP/APBYH/gn3+xnr/8AwUNvf2ifjfP4Dt/HXxQfxH4ZZfD2o3f9oaeTPiX9xA/l/fX5X2tz0r+hQf8ABzN/wRi6f8NZ3X/hF63/APIlfQcUZJi6mNnOFKTWmqi+yO3HYarKo7J2P//S/uIyfrUHkwf88Yev90V+Df8AxE0f8EYf+jsbn/wjNc/+RKU/8HM//BGE9P2srof9yZrf/wAiV/Mr4ex//PmX/gL/AMj4ZYWt0i/uZ5p/wdOxRx/8Ef8A4tOiJEw8U+HcEDH/AC+Cv2F/YWiiH7FX7JDeVET/AMK18N5O0f8AQNgr+SL/AIL5/wDBbH/gm3+2x/wTh+IXwC/Zu+P1x4/+Kl/r+i3tppreGtUsxJDBch5W824t0jGF5wWye1fpJ+yn/wAHF/8AwSH+Gf7L/wCzr8O/GX7UV3pHi7QvBGiaRqlr/wAIfrMn2a7gsoo5Y96WpVtrowyCQccGvcr5HinltOmqUuZTk7crvtHyO6WHq+xirO92f0xCCFSCIIwR0+UcVYyPWvwT/wCImb/gjEP+bs7r/wAIvW//AJEpw/4OZ/8AgjAf+bs7j/wi9c/+RK8JcO47pRl/4C/8jgeFrP7L+5n7xmC3JJMMR752iv5IPjvHF/xFpfswLsjVD8HQSuOM+VqPavufV/8Ag50/4Iy6Xp11fRftPa5rEsalltrTwTrLTTHHCoGtlXJ6ckD1Ir81/wDgmdrnj7/grJ/wWv8AiX/wVd0b4Y+Mfh7+yd4I8JjwZ4Nv9ZtDbvrk/lsiiPPDtia4lfaSIw8ak5Ne3lOV18NGtWxEHCPJJXatduySV9zuw1GpBSlNWVj+yDyYf+eMQ/4CKlAAHHApOucHmlr45LseXGbZ/9P+4gjOa/lR/wCDkHA/aD/4Isc/814s/wD0psq/qtY4Ffzc/wDByp+zN8bvil+zT8Bv2nP2ePC2oeN/iV8DfHdr49Ok2cDXE91ZIVMjJCgLybGihdlUE7Ax7V/N3C84xx0OZ2vdfeml+J8TgGvbK5/SPk5PGKDX84vwb/4Olv8AglD498DaLrHxS+KvjH4E/EEwJ/a3hzWPCmpXD6fc4+dEntoJI5FDZwcg46gV6o//AAcwf8EY0A/4y1mbP93wZrZx/wCSlYT4dx8W06Mv/AWTLA1U/hZ+8hwMZpQe1fgr/wARM3/BGL/o7S7/APCL1v8A+RK4n4gf8HR//BHvwb4W1bW/D/x38X/EjW4YXe10fSvB2qJPeyAfKgkuII4kyeNzMAKmPDuPbt7GX/gLFHBVb6Rf3Hyf/wAEef8AlPX/AMFw+cnfpvP/AG9tX9aBHB9K/wA/L/gip/wV3/YP+EP7S3/BRr9uH9sb452vwg+Ivxg8TW39ieGY9C1LUJNP0uGSeYvJLbW8iZYzQIBuz+5Y4GRX9FX/ABEv/wDBGLgf8NcMf+5O1v8A+RK97iTJMXPE+5Sk0oxV1FtXUUn07nZjsLUc9It6L8j92r3P2W6/65P/ACr5t/4JekN+xr4Bxgj+2fEef/B1eV+WN1/wctf8EZJLeeNf2t2JZGA/4o3W/T/r0rxL9g//AIOKf+CQfwY/Zp8JeAPiL+1OfDvi221LWbi4tf8AhEdZl2pNqdzNEd8dqynMcqN14zg816XAeUYujmftKtOUY8kldppX5oafgzTKcPUjXUpRaVn+aP/U/v2IGMY4r+YP9rL9lfxz46+Lv7RH7Uuv/AT4u+NvFt18dfA3hvw0NMs5ri80jwnpKRTz6np1qGVQ01x58JuOCI5CCyoXz+PP/BSX/g66v/hN+2n8KPHH/BOT4q+F/wBor9l9/B8Vr4w8L67oN5ZW0upreTkvBLPHDPDP5LRDem5CAoZTjj93f+CZP/Byb+wZ/wAFENT8OfCzU9Zvf2cf2jr8LHB4V8TPi21SfgFLHUABFKxJ4jYpIeynmiKtJT7f5r/hvmF9Ldz9Sv2Zf2m/iD8e/i3+094S1T4SQ+Dvhp4F8TDwtpGvf2xDcTazdxW0ElykluhJjKPcbcgkAoyk5Br7Z9ccmvGPhx8APg78IvFHxI8ZfDjwLpfhXxN4v1OTWPEV1BJKx1G9clnlKuxVCzEswQKGYliCSTXs4HXJOajou9vx6je7tsLk/LjJFA55YD8qPlbHOaUY9TjpViFyOeelJwe+QPWlPoCAaCe2eaADrnHWm/NyflWgcDGT+VLgEYzmgD//1f7pvjYf+LP/ABSGD/yL2of+iHr4J/Ya/wCTSfgbzj/iT/8AtaSvvb42/wDJHvijxk/8I9qH4f6O9fBH7DeR+yV8DSMf8gf/ANrSV+ccc/xafo/zR83nnxx9GfQPj+0ur7wF440+xG69n0e9hhGM5kaBwox9SK/mD/4NEtV02L/gnX8ZPAskyR+LdG+LmqjVbMn95beZY2IQsvbJhkH/AAA1/VSQGHJwCMV/Er+1B4C/au/4N3v2zfjP+3Z+zt4J074xf8E+PidqKz+MvC4vVgfQr2SVnACdY2V5JfKmVWXbIyOBwa8PKoqrSqYVO0pWa82un46HDhVzwlSW7tb5H9t3NM5J9PWv5JvDf/B4N+wNqGmW9x4k+CP7Rmgaqygy20VrZ3Ko2OQJBMuR74FdrF/wdw/8E9ZEWQ/CX9qUIwyCNBtyCPUHz6yfD+NX/Ltk/UK38rOQ/wCClSDU/wDg5b/4I66eo3G38OzXJHpibUmz/wCOV/WkflHUYz+Vfxf+Ov8Ag4H/AOCK3xL/AGlPhd+1145/Zf8A2lda/aI8FWbaf4a8RPpbLJptu3nZRYVuxEw/0mblkJ+b2GPqf/iLb/4J0Yx/wrP9qQf9y7B/8frvxuWYqrCnGNJ+6rfi3+p01sPVkopReiP6myM9ga8L8f8A7MP7N/xV19/FnxO+APwZ+IfihoUgbUtc8M2V7dNGv3UMssbNtGTgZwK/nbX/AIO3P+Ccn8Xw4/aiU/8AYuQf/H6rSf8AB3L/AME4Y5CjfDj9pzaO50C3Gfw8+uBZJjU7qmznjgq6d1Fn/9b+mD47/sV/saaL8DvjRrFn+yn+zlYXlp4S1i6hmi8F6cjwulnKyurCHKsCAQR0Ir8G/wDg1x/Zp/Z9+L//AATU1bxp8Xvgd8JPif4mk+JOuW0WpeIfDtpqF0LdIbTbGJZ42bYCWIXOBk+teOftof8AB1v+yT8R/wBmL44fDP8AZ9+Enx11P4m+JfDN/wCH9Nn1mwhtbOxe6haBp5GWV2OxZGYKByQBkda+Yv8AgjJ/wXt/YM/4J1fsFfDr9mz4l+F/j/q/xFttT1PWdam0rw/FLam4uZywWN2mUsFjSIZIHINfmFLK8YsJOLi+ZyVu9lc+Zhh66otWd20f2y+Cf2U/2Yvht4js/F3w7/Z4+CPgbxZahlttT0fwrY2d1BuXa2yaKNWXIJBweQTXv3Yc4Ffy5D/g7Z/4JqdD4I/agB/7FmH/AOSKX/iLZ/4JodT4O/aeQ/8AYrQ//JFeNLJMc9XTZxSwVd7xZ/Tj4j8NeHfGOg6v4V8XaFo/ijwxqFu9rfadqNslxbXkLDDRyxOCroRwVYEGvnL/AIYZ/Ysxn/hkj9mtVx/0JGm8f+Qa/Bm4/wCDtr/gmqsMhtPA37T95cgfJEPDMK7z6Z+0cV8mfFL/AILFf8FJv+Cs9rq/7NP/AASp/ZA+Jfwd8K64jadrXxS8T7rYaRZSDbI0V1gQWrlS3KtJNj7i7sEaUcjxa+JckerbsioYOst/dX3HH/8ABL34LfA/9ov/AIL4/wDBS7VNA+DXwq8efsu+GdKuNJtbSXQrWfRdOvvtVtEgtrZkMSOTb3fKqOFc96/rI/4YX/Yt6/8ADJX7Nf8A4Q+m/wDxmvmH/gkr/wAEzfA//BML9miP4UaZrCeNfiprl5/bnjfxMUIfWtSK4wpb5hDGCVQHnlmPLGv1GwF3Gss2x/PW/dP3Ukl52W/zDF4pyn7r0WhyHgfwB4G+GXhux8G/Djwf4X8A+EbZne30vRrCKytLcuxdykMSqilmZmOBySTXY007sdfxo2+7fnXlNt6s4m76s//X/tfAIzlsivDbv9mL9my/u7q/vv2ffgffX08jTTzS+E7B5JpGOWdmMWWYkkknkk17ng8c0whjyCAcV+BRk1sz4GLtseDH9lb9mLg/8M4/Af8A8JDT/wD41X8pnw7+D/wf1/8A4OsPjh4Bg+F3w3uPh1pHwwgd9CGiWp06Gc6VZkv9l2eUH3TZztzk5r+y9cnOD3r+Pz/gk1dL+0B/wcKf8Fbv2kNOxf8Ah7SIv+EYtroHcm5HtrRQp6fd05jXtZTUkoVpt6KP5tHfg5u05N9D+ocfsrfsw9/2cvgMR/2KGn//ABqt7wz8APgR4L1qz8R+Dvgp8I/CfiG33eRf6b4cs7W5gyCDsljjVlyCQcHoa9eOMHik4xxgCvG52cPtX3A18r/sE5Hxy/bs/wCx1tf/AEkWvqYjJB5FfLP7BP8AyXL9uzkj/itbX/0kWvb4Y/36Hz/I7cq/3iPz/I/UDv0/GjjA9O1J2JP60nBw2SK/YT7A/9D+/cHgY5o4PuKQL1ySRSAADOWAoAXI5/Om7/8AZpw6DB4pcj1FADeSAc7aTH1HSndM9AtJwFzgGgAPIPA3UbRwRkULj7w3UYUDIGaAPzU/bK/5OY/Ye/7Dmr/+kDV9K8A46V8yft7z/wDCI/Ej9jz4r6gAvhbSfGUmmahKR8tt9st2hSRj0Chj3r6aDA/MvPGQfWvyHii6x07+X5I+QzVf7RL5fkf/0fov4iYg/wCDtv4PN0EnwfYf+U65H9K/rV4BA43fSv5KPi9m3/4O0f2f3/56/CFh9f8AQbwf0r+tfn5funvX4znO1H/Av1PjMbtD0QmBjHQUdsZ+lN3EfeFLwd2OvSvHOMXOMcjNHHGMUmT3H5UckEZGaAFAx2xSA5yDj6Uvpx/9ag9RQAhOBnqKX05xSHIHqaUcAA0Af//S/tez9CPWnU3JzyABTq/AT4AKTpk0AgjIpaACkz1oGMcdKOo7igBaTjHtSE5yF5alGMDHSgBaTueOaWm4ycg4oA//0/7YKTg+hpaT1x1r8BPgBaTgYFLSA59RQAtFNJ69/anUAfG/x8H/ABkn+wyB/wBDdf8A/pJX6t9toyRX5QfECQeNv25f2TvAmmBrubw/bav4s1NV6WsXlpDEzemWJGPcV+sAK4GT2r9L4JT+rzf979EfS5In7NvzDpjPJ+tIBxjac07JLLkYpMjGNw/Kvsz2j//U/v2xjJGSaTbgg8mlyp6kGj0bdQAm0febrRhc42mkPBxkY6044wT8poAbjG7acmvDfjh+zP8As+/tM6Novh/9oL4O/D34yaHpty15p9p4i0yK9is52TYZI1kBCsVJGR2Ne5Fedw+tOA4GeooA/PY/8Enf+CaO7I/Ya/ZmJ6f8inaf/EUv/Dp7/gmjn/kxn9mXd/2KVp/8RX6DblxjkUpAwWGaAP5x/h1/wTh/YG0z9vf9r74S+JP2QP2frnTX0Lwv4s8N2M3hm2MVvYyR3FtOYEK4UefbndjuRX2dP/wS7/4JwW8E9w/7Ev7NojjVnb/ilLXoBn+5Xd/tw+EvEHw38a/C/wDbX8A6BqOv6n4Nhm0bxrYWMRe51TwpOweUxoOXktpQLhV7jzB3r6D8OeOvCXxL+Hll478Ca/p/ifwlqmmteWF9aSB47iJkJBBH5EHkHg1/OvEccThMwq0ZTdm3KOr1jJ309G2vl6HxeK9pTrSptve69Hr+G3yP/9XlP+CD3/BOj9kH/goD+0H+33+2Z8ffgX4C8TeGNM+It34e8EeC1sxBomhWyyOysLSPakjLEIIxuBHDtjc2R/Viv/BKj/gm2AoH7Ef7NoA4H/FLWv8A8TX4a/8ABpnz+z1+26Mf81jv/wD0WK/rM/Gv594szGvHHzpxm0o2SV32R8fmFWaqtJ6I/P8A/wCHVH/BNz/oyT9m3/wlbX/4mk/4dU/8E3f+jIv2bv8AwlbX/wCJr9Afwo/CvnVmOI/nf3s4vbz7s/P7/h1R/wAE3T/zZJ+zaP8AuVbX/wCJpf8Ah1R/wTb/AOjJP2bf/CVtf/ia/QDn0o59KP7Tr/zv72Htp92fn9/w6o/4Juf9GR/s2f8AhK2v/wATR/w6o/4Juf8ARkf7Nn/hK2v/AMTX6A5pA3XnNH9oYj+d/eyvbT7n5/8A/Dqj/gm9nH/DEf7Nn/hK2v8A8TS/8OqP+Cb3/Rkf7Nn/AIStr/8AE19/4z6GlwaPr+I/5+P72R7Wfdn/1v6h/wDh1T/wTd/6Mj/Zs/8ACVtf/iaP+HVH/BN7/oyT9mz/AMJW1/8Aia/QLn0pPwr+W/7RxH87+9nwKrT7s/P3/h1T/wAE3cAj9iP9m3/wlbX/AOJpf+HVH/BN7/oyT9mz/wAJW1/+Jr9AfwpAc9BR/aOI/nf3sft592fBunf8Eu/+CdOk3cN/p/7FH7NsF3GwZH/4RS0O0g5BwUxX2r4b8MeHPBui2Hhrwj4f0Xwx4etUEdrYafapb29unokaAKo+gre/Cg89T/8AXrGtiak/jk36smUpPdiUUUViMKQ4YFWAZTwQe9LRQB//1/6wvHH/AAT2/YY+JeuXXibx9+yP+z74p8QTsXnvLvwtaNLMx6lm2ZY+5ripP+CV/wDwTglAD/sSfs2Ef9ira/8AxNfffOORR+FfyysfXSspu3qz4L20+7Pz+/4dUf8ABN3/AKMk/Zs/8JW1/wDialj/AOCV3/BOCPPl/sS/s2DP/Uq2v/xNffmaPwqv7QxH87+9kqvPuz4H/wCHWn/BOP8A6Ml/Zt/8JS1/+Ipf+HWn/BOP/oyT9m3/AMJS1/8AiK+9/wAKPwqP7RxH/Px/eylXn/Mz87vEH/BMf/gm1omg61rV3+xV+zVb2dpaTXErnwraAIiIWJzt9BXkn/BPf/gl3+wD4o/Y/wDg54u8dfsZ/s9694g1q2u9Z+033ha2kma2uLyeW3BZkzgQvEB7AV7h+1z4z8Q/FTVtJ/Yl+DE7XHxK8YIn/CU6hCcp4O8MFh9qu5mH3ZZUDQRIeWaTPRc1+n/gzwvovgfwn4b8GeHLNbDQdKsYdOsYQMCKGJAiL+Siv0Pw5p4itiKmKnJuEVy6t2cm039yS++x6+Sqc6spt6JW+bt+VvxP5Df+Cm//AAa7eHP24/20PhH4j+A6/CD9jr9lnSfCMVl4kfw7osYv9T1H7ZO7eRZpsjL+UYh5sjYGRw2MV+337AH/AARb/wCCfn/BOLRNMHwH+C2lar8RoolW68a+JFXUdbu5OMuJ3XbACR9yFUUelfq+W4yATTq/ZD6Y/9D+/bJHXbQOCRk/4UHA7dOelO68igBCQBk0g7jn8aCRjpn8KOCvoKAF5z7UfmaBjHGMUDp0xQAHGcnrQOQD0paQgEYNAHl3xr/5I/8AFEf9S/qP/oh6+Cv2Gf8Ak0n4G/8AYH/9rSV96/Gsf8Wf+KXU/wDFPah1/wCuD18E/sM/8mlfA0/9Qf8A9rSV+c8c/wASn6P80fN558cfRn//0f7X+Oc9ua/ia+JHw91v/gvH/wAFtfjT+zt8VvF/imH9hT9n+ER3vh3TLx4IdavFlETb3QjEk03mKX+8sUOFKkk1/bExwhPJABNfyO/8GwNtH4j+LX/BXX4pXaLJrWqfFaK0llIySiy38uM/WU1+OZPP2dKrXj8UUkvK7tf7j43BvljOot0vzP3z+Gn/AAS7/wCCeHwg0i10PwF+xx8ANLsoVVFafw5BdytgYBaWdXdjx1JJNfSVt+z78BLK1trG0+CXwjt7OJBHFEnhyzCxoBgBR5fAApPix+0L8CvgPpT618afjF8NPhXpgUsJde1q3st4HXasjAt+ANfkv8YP+DjX/gkh8IWu7eX9pq1+ImoREqbfwtpVzqG4jsJAgjP13Yrjp0sVXd4qUvvZjGFWe12cp+17+178Ov2a/wDgp5+wn+wZpn7H/wCzz4m8IfF23aW/8R3elQx3ekMJp4wsMKxbH/1KnLH+I1+yZ/Z5+AJz/wAWQ+EP/hN2fH/kOv8APj/4Km/8Fo/gl+1z+2x+wX+1x+xd8Lfjf4n8X/B7Uzd3ttrehrbQ6zbi6jnWKJoJJZBuxMhLKMBweeRX7NWv/B0j418Q2dv/AMIZ/wAEkP2wNe1F1B2CaQxMf9l0smJHvivbxeRYr2dOVONnbXW2t/N9jvqYCpyxsrO2p/UD/wAM7fs+n5f+FGfCA/8ActWf/wAbrMuP2X/2aLuRprr9nr4JXMp6s/hWxYn8TFX8zQ/4OQP20r395pP/AARD/anniP3TJdXwz+WmVGf+DiD/AIKG3HGn/wDBDj9o1vTzLvUP/lcK4f7Exn9SX+Zz/VK39Nf5n9NFt+zD+zXZkm0/Z++CdqD18vwrYrn64iq5/wAM2/s7H/mg3wbP18MWX/xuv5gz/wAHAH/BUu6z/Z3/AAQ0+NuT08y81H/5BFQ/8P4v+Cwtx/yD/wDghf8AE4/9dbzU/wD5EFH9i4zuv/Al/mH1Ot3/ABX+Z//S/ry/4Zr/AGczwfgF8GT9fC9l/wDG6Z/wzP8As3nP/Fgfguf+5Wsf/jVfzB/8Pv8A/gtvdf8AHj/wQ38VxZ/563epn/2gtRn/AILNf8F6Lrmw/wCCJMkOenmzakcfqtfj39jYr+Zf+BL/ADPjfqVXuvvX+Z/UHD+zR+zlBLHNF8A/gxFMpyrL4XsgVPqD5XFevaZpOlaHYwaboum2Gk6dGMRwWsKxRoP9lVAAr+R3/h7v/wAHDt3j7B/wRc8OxDt5w1E/+3K1C/8AwVY/4OTLk4s/+COXw5t89DJb6i2P/KgKiWSYh7yj/wCBL/MHg6j3a+9H9ewwBjFO/Cv5A/8Ah5j/AMHN13j7P/wSV+Ednnpvtb3j89SpB/wUJ/4Ojrsg2/8AwS/+Btn/AL9pccfnqdH9gVP54/8AgSB4Cfdfej+vzjHbFG4fWv5Bx+3H/wAHVN1/qP8AgnN+z5aD/atGGPz1OkP7YH/B1zdgiP8AYO/Z1sv+3ZBj89RNT/Ycv54f+BIf9ny/mX3n9fW7jPakDHPXA6V/ICf2mv8Ag7JvM+X+x/8As3WH+9b23H53xpP+F2f8HbGoDyof2eP2aNJLfLvMVh8nv810af8AYsv+fkP/AAJB/Z0v5l95/9P+mL/gpt+3D8PP+Cfn7HfxZ/aA8b65Y6frUGnzaf4WsHkAm1nWpY2FtbwoeWO752x91EZjwK/Lr/g2Z/ZA8afAn9inxN+0Z8WrK4tPi58bfEEvja6S4TbPDppLLah88gybpbj/AHZkr5X+Cv8AwRG/bv8A25vjz4M/aT/4LafH3S/HPhrw7cC80L4X+HLhWsxMGDBbnYiwRQHHzJHvdwAC6jg/1t6ZpmnaJpdjo2kWVtpuk2kCW1rbwIEjt4kUKqKo4CgAAAdhX41ialPD0Hh6cuaUneTW2myXfufG1ZRp0/Zxd29/8jQpi4AzjFKfTrmg9D0BxXjHENbHynjrXy1+wR/yXL9u3/sdbb/0kWvqY84PBWvlj9gnB+OH7dhAGP8AhNbYZ/7dVr3eF/8AfofP8jvyr/eI/P8AI/UDAwep/ClHTrmlpoGD0OcV+wn2A6m8jtn8aUDHqaOD6GgD/9T+/bjGOPpTqQDBJpuB7/lQAd8Y+tKPu+n0oGD0IzScE8DDe9ADCxYing9V6fSkJPB569KG2/N/eoA8m+Nfwi8KfHf4ZeKfhd4ztmk0XU4NnmLxJayqQ0c0Z7OjqrD6V+avhH43eLv2btSg+C37W1vd6KLRha+H/HgiZ9L8QWw4jM0gB8icD7wbAOO3f9glOCCDkGsHxD4Z8O+LdMutE8U6Fo/iPRplKS2l9bJPDKp6hkcEGvns8yCnjEpX5Zrr+jPPx2Xxra7NH8TXxf8AEWga5/wdbfsq63oOt6TrWk3HwmYJc2lwksTn7HfdGUkZr+ugXEBHMsGf98V/Hv8AtVfspfA1f+Dpz9lD4N6T4P8A+ER+HmsfDNr66stGupbPE/kX/wA0bI2U5jXhcDjpX9Rh/wCCXH7Kw5Fv8VCP+xuv/wD4uvncy4WxVRU1Br3Ypbvz8jzcTlNWSio20Vj/1f7V/PgHPnwD/gQoWeAj/XxA9/mFeB/8Otv2Vx/y7/FQn/sb77/4ug/8Etv2VuP9H+Kv/hX33/xdflf+pmN7x+9/5Hyf9jV+6+//AIB799ogwf38Of8AeFJ9pgJP7+ID/eFeA/8ADrb9lb/n3+KmP+xvvv8A4unf8Otv2Vs/8e/xU/8ACuvv/i6P9TMb3j97/wAg/sav3X3/APAPfPPg4PnQr/wIUC4gwP8ASIh/wIV4F/w62/ZWx/x7/FX/AMK++/8Ai6Uf8Etv2VuM2/xU/wDCuvv/AIuj/UzG94/e/wDIP7Gr919//APfftEPP76D/vsUn2iH/n4h/wC+xXgX/Drb9lbp9m+KmB/1N99/8XTv+HW/7K3/AD7/ABU/8K6+/wDi6P8AUzG94/e/8g/sav3X3/8AAPffPtunnw4/3hTTc2/eaLPX74rwP/h1t+yt2t/iof8Aub77/wCLpD/wS2/ZWH/Lv8VT/wBzfff/ABdH+pmN7x+9/wCQf2NX7r7/APgH/9b+1j7RD/z8Q/8AfYp32i3/AOe8P/fQrwE/8Et/2Vs/8e/xUA/7G6+/+Lo/4dbfsrH/AJd/irz/ANTfff8Axdflf+pmN7x+9/5Hyf8AY1fuvv8A+Ae++fb4x58OP94UfaIP+fiH/vsV4F/w63/ZW6/Z/inj/sbr7/4uj/h1v+yt2t/ipn/sbr7/AOLo/wBTMb3j97/yD+xq3dff/wAA98+0Q5/10P8A32KXz7fvcQn/AIEK8C/4dbfsr/8APv8AFT/wrr7/AOLo/wCHW/7K+P8Aj3+Kmf8Asbr7/wCLo/1MxveP3v8AyD+xq/dff/wD337RB18+HP8AvCgXEHe4h/76FeBf8Ot/2Vu1v8VP/Cuvv/i6P+HW37Kwz/o/xU/8K6+/+Lo/1MxveP3v/IP7Gr919/8AwD33z4P+fmL/AL6FJ58ByPtEOf8AeFeCf8Ot/wBlb/n3+Kn/AIV19/8AF0n/AA63/ZXxn7P8U/8Awrr7/wCLp/6mY3vH73/kH9jV+6+//gH/1/7WvtFvj/Xw5/3xTTPb8fvof++hXgf/AA63/ZX5Jt/ipkf9Tfff/F0f8Otv2VT/AMu/xV/8K6+/+Lr8r/1MxveP3v8AyPlP7Fr+X3/8A93uNS0+1jee8v7G2gHV5JVVR+JNfL3xN/a1+H/hS4HhD4dCX40/Fu5/dad4c8Pn7S8kpOAZ5UBSGME8sx6Z4rsrT/gl3+yVDPHNfaB4+1yFTn7Pe+Kb+SJ/qokGa+tfhn8DfhF8GdOGm/C34deE/BNuVCyPZWaJNP8A9dJsF5D7sxrbDcFV5SvVkkvLV/kjWlklRv32kj5+/ZH/AGefFPw7bxZ8Y/jLcWGp/Hvxb5b6o0OHi0a0XmKwt2/uJkbiOGYDrgGvtw4IODS4xySRRjuOn1r9DweCp4emqVNaI+ho0Y04qEdkLg9d1NGc/wB6lznHX1pcFlw3FdZqNPc9f+A0DthaP4sbmpflOBQB/9D+/fnjgAU3GBjAJ+lHPC9fWjnkn8qAAj5xxkUDhRyFpuD3OM07PAON1AAFPdjQenQD60cgkknFIep7n+VAEDRJIGDAOrDBB6EV+YPxB/Y1+J/wYvfGXxB/Yl8caB4M0/UHn1HVvhx4ggeTw7qFw4Jkls3j/eafK5JJ8tWRj1Uda/UJQM+p71l62c6Lq+D/AMu0oPt8hrw85yDC4+mqeJje2zWjXo1/wz6nJi8FTrxtUW2z6o/hL/4Nivjp47+FnwK/bB060/Zp+N3xchufiteTXV94Qt7W7gsrjywGgYSzxPkckHbgiv6dv+GyvHhH/Ji/7aA/7gFj/wDJlfi5/wAGhJz+zt+3V/2We/8A/RYr+vjbyPmr5fN/D2nisTLEOtJc3S0f8jz8TkvtJufO1fyR/9H+p7/hsrx3/wBGL/tnf+CGx/8Akyj/AIbK8dgbv+GF/wBtA/8AcAsf/kyv1X5OCuMUhUdc4FflH/ELKf8A0ES+6P8AkfO/6v8A/Tx/cj8qf+Gy/HeP+TGP20Cf+wBY/wDyXSn9srx2cf8AGC/7Z4/7gNj/APJlfqsOw+YUnUHgnmj/AIhZT/6CJfdH/IX+r7/5+P7kflSP2yvHg/5sX/bPz/2AbH/5MpW/bL8ecZ/YX/bQz/2ALH/5Mr9VsD1JIo+9ja2KP+IWU/8AoIl90f8AIf8Aq/8A9PH9yPyq/wCGy/He7H/DCv7Z/wD4T9j/APJlJ/w2T48xgfsLftn/APggsf8A5Mr9VvYZB60fLkc0f8Qsp/8AQRL7o/5B/q//ANPH9yPyp/4bK8d5wP2Fv20On/QAsf8A5MpP+GyvHe4j/hhf9tD/AMEFj/8AJlfqx3PQ+lN9Bz/k0f8AELKf/QRL7o/5B/q//wBPH9yP/9L+p4ftk+PAcj9hf9tDP/YAsf8A5LpR+2X48Jz/AMML/toEf9gCx/8Akyv1WGSGzyPam8E5BH41+Uf8Qsp/9BEvuj/kfO/6v/8ATx/cj8qx+2V47xkfsLftoH/uAWP/AMmUg/bM8dHaR+wv+2hj/sX7H/5Mr9VwOhOc0AjAJOKP+IWU/wDoIl90f8g/1f8A+nj+5H5Vf8NleOgOf2Fv2z8f9gCx/wDkymn9snx4Of8Ahhf9tD/wQWP/AMmV+rHHOMUgG3qeKP8AiFlP/oIl90f8g/1f/wCnj+5H5Un9svx4Bk/sK/tnf+CCx/8Akyl/4bL8d7sf8MK/tn/+E/Y//Jlfqr1BIJNGRyMNmj/iFlP/AKCJfdH/ACD/AFf/AOnj+5H5Vf8ADZXj3/oxX9s7/wAEFj/8mUv/AA2T47xj/hhb9tD/AMJ+x/8Akyv1U+VgcGl2jOeaf/ELaX/QRL7o/wCQf6v/APTx/cj/0/6nv+GyvHnT/hhf9tDd/wBgGx/+TKX/AIbJ8d5I/wCGFv20P/BDY/8AyZX6r7QMZHNAHJJGK/KP+IWU/wDoIl90f8j53/V//p4/uR+Up/bJ+ILfJb/sJ/tmyzHhVbRNPQE+7G8wPqaqQ6v/AMFBPj/cvo/h34ReH/2NfAcnyTeIPE2oQ63rzxnqbXTrUmCJ8dDLKcdcHpX6xHG0nGDSK2/grgVtQ8LcKpJ1q05Ltor+rSv9zRUMghf35trtov8Ag/dY+Z/2bv2X/h1+zZoWuW/hVtc8S+Mtbuft/iPxRrUxuNW8Q3eMebcztyQBwqDCoMAAV9MhQoI7U5l/ujmkGfmJb/61fo2DwVLD0lRoRUYR2SPcpUo04qEFZId19RRyTzladTDkDIYj611mgfdxyfejOQflY0p+UcCj1XbQB//U/v3z68Uc568Un3sggikxgADnmgBc4x2FIrdj1pSOvApGXO0AcUAL68txQcbckZFHUEYP40LkDnAoA5bxroC+KvB/ijwy7tGmoafcWRYcbd8bLn9a/LP9gbxDIfgj/wAKq1mM2PjTwNq194a1SzcbZIfLuJDGxHoyMMHvg1+ueAV25r84P2kP2dfiZ4U+Jl1+1D+zDb6fe+OZrZLfxX4UnYRQeK7ZMYeNui3ahQFY43YxnsfjuLMpnXpxq0leUb6d0zyM3wkqkVKO6PoeT5kdRxkEV/nCf8Ep/wBkH/gph+1n8RP29vAX7HH7asX7HfwPtfiXdJ40vbOScanf3Ty3Aj+ziBVkIEavn99GMnvX94fw9/bT+Cvi6U6B4u1ef4OfEKE+VeaB4qT+z7iGUdQjSYSQZzgqc+wr+bv/AINh/FnhfQ/iD/wVktNT8R6Hp0T/ABYimgaa6jQTIXvxuTJ+YcDkV8lk+IlSw1ecbXXLur9ezPHwVRxpza30P//V+ufhf/waifsiya23jb9rT9oL9oX9rHx/MwkvbzUdSNjHdP1Jb5pZiCSespr9afg1/wAEYv8Agl18BxZT+BP2J/gbPqtvjy9R1zSU1e6BH8Qku/MIPuMV+hf/AAsTwCOnjfwkB/2EIv8A4qkb4ieAQefG3hPH/YRh/wDiq/EK+c4mp8c3b1svuR8RPF1ZbyZd8L+DfB3gmwj0zwZ4T8M+EdMUALb6ZYxWsS49EjVRXUmSQA/O5HpmuJPxE8AcbfG3hQH/ALCEP/xVA+IngHPHjbwl2/5iEX/xVcD8znR2nXk9aQ+uDXGr8RPAA/5nbwkR/wBhCL/4qm/8LD8A8f8AFbeEx/3EIf8A4qgDsz0ODn8aMD39a4w/EPwBz/xXHhMD/sIxf/FUp+IfgDOR428JD/uIRf8AxVAHaU0c9c/jXGD4h+AMjPjbwn/4MIf/AIqk/wCFh+ATx/wm3hPH/YQh/wDiqAP/1v7Xx69+9HtgYrjB8Q/AHT/hNvCY9/7Qh/8AiqD8RPAOePG3hPH/AGEYv/iq/AT4A7LH+z39adXFj4ieABn/AIrbwl/4MIv/AIqk/wCFieAf+h28KZ9f7Qh/+KoA7TuOce1LXGf8LE8AHr438J/+DCL/AOKpP+Fh+AOMeNvCf/gxi/8AiqAOz4HoKT36nFcaPiJ4Ax/yO3hPP/YRi4/8epD8RPAOP+R28JH/ALiEX/xVAHa0nOeMba40/EP4fD/mdfCbf9xCH/4qk/4WJ4Az/wAjt4T2/wDYRi/+KoA//9f+1/j8qaRnGTg1yB+IfgAZx438Kk/9hCH/AOKrzzxv+0v8Avh7ZzXniv4s+CbDYP8AURXyT3Eh7KkMZZ2PsBX8/c6PgLnseq6pp+haXqGtapcxWemWsL3E80hwscaglmJ9MCvnn/gmho+o6l4A+MPxp1G1ezg8eeNr/WtPDqQz2KN5UTnPY7XI9q8ThsvjB+31cx+G9B8PeKvg1+yp5yPq+sanC1rqfi6ENn7PaRdUhb+Jz1HHqtfr34Y8N6J4O8O6J4V8O6fBpegadbR2dnbxLhYYkUKqgfQV91whlVT2n1qorJLTzv19D3Mnwkub2stuhvc5PzdKVTntgUEbselL0wME1+jn0YZ6cGmjI3Zyadk+hpu7nbtoAQgYP3h7U7d/u/nSN95c9KTb/sfrQB//0P79R93uopc56BqQnrxkd6McksQaAEJIwxB/wpecZySaOOcDPGaQqCeQfrQB8Tft9/tHeNf2UP2c/Fnx38FQ+ANTuPD7297faXrryq+r2YmQTW1l5bKRdNG0hRm3KCoypGcdb8Of2idKg1T4Z/DP41+OvhZpXx08ZWd5r2g6Dok8hWbTEZdu1pGYyOiyIrP8quwbYuAQPYvGnwh+GHxH1HTdV8feBvDfi/ULO1u7K1l1C2WYwQXMZinRQ3AEiEoeOVJHc14j4e/ZYsfA37QHgn4rfDzWtD8IfDvRvAx8EReEYtGDJBELx7kS2tx5g8ncWjV1KPuEMeCvzZAP5of21o/7M/4Ozf8Agnbfp+6+1/DJ4iRxuwNTWv7FiT8rY4r8hPj7/wAEpdO+OP8AwVJ/Zi/4KWv8ZL3w1qPw30B9CXwqmjrKmrAtcHzDdGUGP/j6Ixsb7vvX694BGOSRQAMWX+Km789Qpp/GCO/Sg8Db7UAf/9H+/Tc/939KP7uPlWlGRjOTSfL947qAFzgEZJak+fbjFO+VuKbgLz3oAcCDwDmk6A+vSjPA4J/CjGeVOBQAA4AG1qdxntmgDAxTeM/M2TQAdF5FLgMB1pflPJxTT1ByDzQB/9L+/UYPPH/1qTPv9eKUbV70uPujdigAwDjPX3pOfTn0zR971VqXv1J/pQAncfNz7incc4xmk4GOntSbu2dpoACeueP60ue+Rtp1Jx0oAZ3yCSD6UvBODkD0o5AAz+QpcA9OuaAP/9P+/QDnnp70c8nK4/nS+h6dqUAYOOKAEO3nJ60ny59j1o/+Kpf95srQAmR8mDS5Ayck0nG5QD2pzHqME0AH97jNJz6gcUevdsUhwM5Jz/OgB2QFyBkUvr2FM2gnOee9LgYGGwKAP//U/v1+bIztoB7kkc0MAccgGnLjHHSgBvY8jrQc5GOCaOox0B6UhUHpgGgD8OP+Cnv7U/xL+CHxt/Zv0b4J/tZ6T8PdZ1TXrTwx4q8KzaTZX9h4e02/EkI8Q6q7RtPb+VNcWAhDyRQu4wQ4dtv2F8IP2ofhXaeL9P8A2NW+MPj/AOPfxg07wJZ+Jbvxhe6KDBrltdrO1vNNd2cEdmkssdvLIoRUQqvy88V9f658KPhp4mn8ST+Jvh94H8RS6zFbwaq97pcMzalHCcxJOXU+YqHJUNkKemK8Q8NfswXfhP8AaM+MHx50r4o69Fo/jHSNN0y88Krp9sLSGSzt3ggkFxsM3lossjLCpVQ8jklsgAhZaP8ArT/MHqfze/8ABoP/AMm6/t1/9lov/wD0WK/r5JwBjmvx3/4I9/8ABK69/wCCV/w6+PHgK9+Mdp8Yj408a3Hi4XMWjtp/2ASLt8gqZZN+P72R9K/YbAXBOc0ABzsO7rQcnPp70owActmggBSMnFAH/9X+/bnK9QKM5HJ2mkHUZOT9KOR1OOaAADPc8GjIx97mgqDyTkUgGDjPHWgB3oeT2ppzj+opegLZ3UEfKeSwoATGCMDt6Uu3leWNOyOueKaGGQACKAAA9csB70HGOG5+tB6emfWl446e1AH/1v79RkY3NSnb940cHGSCaT6Y9s0AL0+9t20bVyO1Dbf4qQLnBJOaAEGckABTS/dBx07UZHIJwaQAHoTn1oAXPB5ORS546nOKReeeN1LkHOSMUAKB8uDQSecA5oJ9xmjHOaAP/9f+/bv6/wBKU9Dzik9sdvWkOBzjOaAFP3ep/Cl65HIpOOxUUh5zjkdxQAoIxnGBSrjAx0oB3DjNNz7DrigBzYxzTfm3+1O4Ue1Jnp0xQAvH3RkUmcAE5o+Un3pOMA/e9KAP/9D+/YY28ZxR12kdKXvSDHXj60AOpuNy8g5o4IwDzQCAByBQBFLIkMbyyuqRKNzMTgKBySa+NvFP7dnwD8Ha6+gazc+ODqBs7TVYlt9CuJ/tOm3F4tlFfRqilmgM7ogONxDB1Up81fWevJrMuhazF4cbTE19rWVbE3yM9uLgodnnKpDFN2NwBBxnFfn/APCn9nv9pjSfj34r+OvxD8RfCaHxF4g8M+GfDd2+iQziPRbfT2kkvI7SGYMCLqWaRk3N+5UqD5hTLAH2N49+EPwm+Kto9t8QPh/4O8Zrsz/xMLCOV0B6ckblr+K7/g2w/Zg+AfxU/aQ/4LDeFfiV8NNB8TQeHvifBBpcEzSotjC1zqS7ECOvH7tRznpX9Sv7HXhfx5a/Gf8AbN8ffEC/+Kst/r/jd4NJs9bthHZ2WjWNvDaWvkN5SBjIyXU4KFhsmTPzZJ/HH/ggZ+x3+05+zD+29/wWE8X/ABu+DXjP4b/D3xv45ttU8IalqUISDXYFvtSdpICCSQEuITnA4YVhPC0pJqcU776EypRe6P3LP/BO/wDYt/6IB4RH/be5/wDjtL/w7u/Yt/6IB4S/7/XP/wAdr7SJAxk4oGMY3A1yf2Xhf+fcfuX+Rh9Up/yr7kf/0f7Ij/wTu/Yt/wCiAeER/wBt7n/47R/w7v8A2Lc4/wCFAeEf+/8Ac/8Ax2vtPj2B/lQOQM15v9l4X/n3H7l/kc31Sn/KvuR8Wf8ADu/9i3p/woDwjn/rvc//AB2l/wCHd37Fv/Rv/hEf9trn/wCO19p9Rw1J8o7gUf2Xhf8An3H7l/kH1Sn/ACr7kfFh/wCCd37F3H/FgPCP/f65/wDjtB/4J3/sW4/5ID4RB/67XP8A8dr7S+U5HYU78aP7Lwv/AD7j9y/yH9Up/wAq+5HxWf8Agnd+xb/0QDwiP+21z/8AHaP+Hd37FuR/xj/4S/7/AFz/APHa+0sA9z70uAcHNH9l4X/n3H7l/kH1Sl/KvuR8W/8ADu79i3/o3/wj/wB/rn/47TT/AME7v2LgCf8AhQHhL/v9c/8Ax2vtTgHJNJlRuOfrR/ZeF/59x+5f5B9Upfyr7kf/0v7Iv+Hd/wCxb1HwA8In2865/wDjtL/w7u/YtP8AzQDwj/3/ALn/AOO19pY7cYpfTnivO/svC/8APuP3L/I5/qlP+Vfcj4r/AOHd/wCxb/0b/wCEv+/11/8AHaP+Hd/7Fpz/AMWA8I5/673P/wAdr7UwOtIcYPej+y8L/wA+4/cv8g+qUv5V9yPiz/h3f+xb1/4UB4Rx/wBd7n/47S/8O7/2Lf8Ao3/wj/3/ALn/AOO19o8Y64PWl+XGMjFH9l4X/n3H7l/kL6pS/lX3I+LP+Hd37Fv/AEQDwl/3/uf/AI7QP+Cd/wCxaP8AmgPhA/8Abe5/+O19qYBxzmk7fzo/svC/8+4/cv8AIf1Sl/KvuR8WH/gnd+xb/wBG/eEh/wBt7n/47Xongr9kb9mb4c3EN74N+CHw90W/jJMdx9gWaVffzJNzfrX0gc4pPbr/AEq4Zdh4u8acU/RFLD01qor7j//T/vwjjSGNYokWKNRhVUABR2AqYYO7BINHyjAOKCPlIAoACAT3zRnBC5OaXA24PIpBjPT9KAAsMeooBznrS4HTHFGRjOeKAAjI4OKTn/a/Skyo4GMUceifnQA71x1pvoVWgqMbh1oyBliNtAH/1P79SPUDpQRgdWxSHnoORRjOeqigBQBjdjmjoPv03+L5KAV75PpQA7G3nljQOOeefejqDjOfrSEZX+KgBwx82OKD3OWFJgdfm4oHTuT9aADpkfzNBxycZIoGOoLf40mOduWFAH//1f79dg96NnXnJpAudp60/wB8GgBOMY6ZpAAMFRmkwRwDn+lHfOHoAUZ2DHWnem7GaaSQvTBoByN2OaADscnJxSgD13UN6bSaTq3sKAFx6HvmmjHVsc0uQcgA/hS8fxbc0Af/1v79dgzknNKNoBI6U3J7ZA/OnAYyMHFAAFAJNHyrikzy3BxT6AGnGDyAaOnCrQeefmWg/iD0zQA3GTjkYpVUjOaAM93p2eM4NAB06Cm4Y4PANIccDDZHpSjPQZ/GgD//1/79Tkg8c/zowMnIP+NKRkEDik4ZSdtAAQBgjFAAPBABobjBOc+tIOgHPX0oADznjI+tOwd27IxSd9wBNL97qrUAJ0DH3pep5xSE8c560h+8d2dtAC46detO4wOcim5GM7eKP7uNxoA//9D+/Ujg8Z70EDODnmk+7u4alJ6YB/KgBPu4zz/SlG0j0JpenADUny/3SfwoAQDk4HA4+tO44zjHak3K3y80o28LnNADQMYyxHtTsAH8PWm4A4wxFL0JGGxQAmATgAEfWlAIH3VNHXsB6UcbTkEj3oA//9H+/UqCST0pCMEbjkUpwe54o6YHOPpQAi7R1+9QE4OcUrH5c8ilf7poATvjB6Up3cYNIMHLDIoY42/3qAEVRzuoI4U4zSjb/DmkHORyM0AKqjhuc0duWOTRjOMZ9aTPCn5qAP/S/v1x97r+dAHQDgelOB5IpnHU7se9ACgKMHp9aCoI4ApMYG5d1A4yeSfpQAoAIz7Yp3OOwpGOF680h5GPy96AFyD6+1HBGT0pSM9gaQHpwaAFz1xyaQgnP6UvXI5FITk4BwaAP//T/v3ByM0meq9PpS5OOnPpSdhkbjQAvbik444PWlOcHHJpvXKgbaAFxwNppcc5yaRsbfmoH+7igAyCO+KQEYznnvS4A59KOMLzgUAG4ADqabnDtnigryOQfrSrnJ4wKAP/1P798j+lID1z64oBOCSOaCDjgAHFADqap46YFGB/cpQSeoxQAtNz1x/+ujGM8/8A1qPn/wBmgAPy7fSg9cAD3pDgEDbmnDGT1oAMDnjrSKQRwMUhPOQpPvTgMDigD//V/v2BycEYNHQH1xSkA9aQ7V5xQAoOQO1NyoGeo6UuTkdMGk4IHoTQApyRkHafekAz1Lg0HOMBSPSneg5FAC0zA5x680ADLDFLzk8ZWgAJyODjvS5AODxSZ68Y/rSN03L1oA//1v79VAABpQdwOODSdR0P0pTg7uOaAA9RjGaDjg+9JjpkcdsdqDnuP/rUAOBXoCKOnJppGSpFKucfN1oAP/Hufyp1M5IH8NOJIHAzQAg64zn1zS4AGMACjkD1NNPY8nmgD//X/v3H1zSDHy884pM9MrgUv93HSgAGFwoFHABxTqZg7cd6AHDA4FIEHcc0dzxj3pMH0SgBMkMRxTicED1ph+/Tm6rQApJyRnHGaaGwcYHXFKep+lM/j/GgD//Q/vyzkqKUMec800dRQOhoAXJOBnrTgTgdBzimjqlKOg+tADsYU5OaBg7eBTj0NMX+D8aAEwMZ7/8A1qCNqkdeaX+H/PpQ/Q/WgBwXrnnNIvCk0+mD7lAH/9H+/TOOevy08gHrUZ6fhUtAEY/g/GnYz375pq/wfjUlADW6HjNA+81D/dNA+81AB/s+1IfvKO1L/H+FIfvigA3HdjtRt9STSf8ALSpKAP/S/v1Uhu3SnY5B70yPvUlADGA4+tKORjoAaG/h+tC/xfWgAxxgnIoHO7PrTqav8X1oAaMEkY4zTyMgimD7x+tSUAMKjk849KQDnHTIp56GmjqPpQB//9P+/faOM84pv8LdvpT6YOj/AI0ADYHzYyacAAAKa/Sn0AMCjLDGaQjHHqacPvNSN1WgBWGVNNPfHGOlPPQ0xv4/woACcAcelPByM1G3QfhT0+6KAP/U/v0IAKYp5GRimt1Wn0AMZRgnvSgAgdu9D/dNKOgoAQjg8n1pF7jjApx6Gmr1agBQuO+e1GxfSnUUAMGCW49qcVBOSKavVqfQB//V/v1Kjj16UoAIJxgmlPUUifdFAAV69eab02cDJqSoz/yzoAOu8cU5uASAM00f8tKc/wB00AA4UH2pMck+4pf4Pwo9fqKAGtgE8Z704AYWmv1P0p46CgD/1v79T1z74pG+8o4pT1/Gkb760AKvK/pTm6HBxTU6U49DQA3oSB6Uo547YpD1P0pV/oKAFAAzijAPUZpaKAI8/dOPWn45zkmo+y/jUtAH/9f+/cjIxRnofWlpo6JQAN/D9aDxnHYUN/D9aD/F9KAHUxuWUdKfTD98UABJGMe9L329sU09vxp38f4UAL1yO3SmnKjrmnDqaa/SgD//0P79gd36UEZxyaRep/Cn0AFR/wAW3+GpKj/5aUAAOWA6Cn46cmox94f57VLQBHuyCcCngY7k1EPutU1ADTyC3tRnkr2xR/B+FIPvmgD/0f79jhQSAKAMqOTQ/wB00o6CgBaTHIPpS0UAJg5JzxTW6rT6Y3VaAHY5zk0tFFADOqgn607HXk00fcp9AH//0v79j8oAH0pcAgA5NI38P1p1ADCcY/GndRnoTTD2/GnjoKAGOcFSOtOB5YU2TtTh95qAEPDE+1CgfeHFB6n6UqfdFADqYOq9emafTB1H0oA//9P+/iow2QRjjFSVCv8AF9KAHjJXrzSFyCRxTk+6KiPU0Af/2Q==)

# #### 1. Vanilla RNN
# 
# Клетка Vanilla RNN принимает на вход очередной элемент $x_t$ и предыдущее скрытое состояние $h_{t-1}$ и выдаёт новое скрытое состояние $h_t$. 
# 
# Преобразование происходит по формуле:
# 
# $$h_t = \sigma(U_hx_t + V_hh_{t-1} + b_h),$$
# 
# где
# * $x_{t}$ &mdash; вектор размерности `[emb_dim, 1]`;
# 
# * $h_t, b_h$ &mdash; векторы размерности `[hid_dim, 1]`;
# 
# * $U_h$ &mdash; матрица размера `[emb_dim, hid_dim]`; 
# 
# * $V_h$ &mdash; матрица `[hid_dim, hid_dim]`, 
# 
# $U_h, V_h, b_h, W_y, b_y$ &mdash; обучаемые параметры RNN-клетки, а `hid_dim`, `emb_dim` &mdash; гиперпараметры.
# 
# Если же мы хотим решать задачу классификации, то мы можем применить линейный слой с функций softmax к скрытому состоянию и получить предсказание вероятности:
# $$o_t = \sigma(W_o h_t + b_o).$$
# 
# В модуле `torch.nn` клетка Vanilla RNN представлена классом `torch.nn.RNNCell`. Его можноинициализировать следующим образом:
# 
# `torch.nn.RNNCell(input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = 'tanh')`. 
# 
# Класс `RNNCell` возвращает скрытое состояние $h_t$.
# 
# Самый простой пример применения клетки RNN:

# In[ ]:


rnn = nn.RNNCell(input_size=10, hidden_size=20)
input = torch.randn(3, 10)
hx = torch.randn(3, 20)

hx = rnn(input, hx)
print(hx.shape)


# Если на вход не подается hx (например, для первого элемента последовательности), вектор скрытых состояний считается нулевым

# In[ ]:


hx = rnn(input)
print(hx.shape)


# Напишите функцию для последовательного применения клетки RNNCell к последовательности векторов.

# In[ ]:


def aply_rnncell_to_seq(rnn_cell, input, hx=None):
    """
    rnn_cell - клетка rnn 
    input - входной тензор размерности (seq_len, batch_size, input_size)
    hx - начальный вектор скрытого состяния

    возвращает: последнее скрытое состояние для каждого элемента батча
    """
  
    for i in range(len(input)):
        if hx is None:
          hx = rnn_cell(input[i])
        else:
          hx = rnn_cell(input[i], hx)
    return hx


# In[ ]:


input = torch.randn(15, 3, 10)
rnn = nn.RNNCell(input_size=10, hidden_size=20)
hx = torch.randn(3, 20)

aply_rnncell_to_seq(rnn, input, hx).shape


# In[ ]:


aply_rnncell_to_seq(rnn, input).shape


# Каждый раз писать цикл, чтобы прогнать все элементы через `RNNCell`, нужно писать цикл, что совсем неудобно. А если мы хотим использовать ещё многослойные `RNN`, то неудобство возрастает. Для удобства существует класс `RNN`. Его параметры:
# 
# 1. `input_size` — размер эмбеддинга;
# 
# 2. `hidden_size` — размер скрытого состояния;
# 
# 3. `num_layers` — число рекуррентных слоёв;
# 
# 4. `nonlinearity` — функция активации — `'tanh'` или `'relu'`, по умолчанию: `'tanh'`;
# 
# 5. `bias` – если установлен в `False`, то $b\_h$ устанавливаются равными 0 и не обучается, по умолчанию: `True`;
# 
# 6. `batch_first` – если `True`, то входные и выходные тензоры имеют размерность `(batch, seq_len, feature)`, иначе `(seq_len, batch, feature)`, по умолчанию: `False`;
# 
# 7. `dropout` – вероятность отключения каждого нейрона при dropout, по умолчанию: `0`;
# 
# 8. `bidirectional` – использовать ли двунаправленную сеть.
# 
# \\
# 
# `RNN` возвращает `h_n` и `output`.
# 
# * `h_n` – скрытые состояния на последний момент времени со всех слоев и со всех направлений (forward и backward).
# В случае, если слой один и RNN unidirectional, то это просто последнее скрытое состояние. Размерность `h_n`:  `(batch, num_layers * num_directions, hidden_size)`.
# 
# * `output` – скрытые состояния последнего слоя для всех моментов времени t. В случае, `bidirectional`=True, то то же самое и для обратного прохода.   Размерность `output`: `(batch, seq_len,num_directions * hidden_size)`.

# Сначала для однослойной RNN

# In[ ]:


seq_len = 5
batch = 3
input_size = 10
layers_num = 1
hidden_size = 20

rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=layers_num)
input = torch.randn(seq_len, batch, input_size)
h0 = torch.randn(layers_num, batch, hidden_size)
output, hn = rnn(input, h0)
print(output.shape, hn.shape)


# Теперь для многослойной.

# In[ ]:


seq_len = 5
batch = 3
input_size = 10
layers_num = 7
hidden_size = 20

rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=layers_num)
input = torch.randn(seq_len, batch, input_size)
h0 = torch.randn(layers_num, batch, hidden_size)
output, hn = rnn(input, h0)
print(output.shape, hn.shape)


# Объясните, какие размерности чему соответствуют.
# input_size соответствует размеру энбендинга, layers_num - число рекурентных слоёв, hidden_size - размер скрытого состояния.

# ---
# 
# #### 2. LSTM
# 
# Как выглядит LSTM и какие у неё обучаемые параметры, было рассказано на лекции. Поэтому сосредоточимся на работе с классами `pytorch`.
# Устройство LSTM в `pytorch` аналогично Vanilla RNN. Основное отличие в параметрах &mdash; скрытое состояние представляет собой кортеж из двух векторов $(h_t, c_t)$. 
# 
# Для начала посмотрим на одну клетку.

# In[ ]:


seq_len = 6

# инициализируем клетку lstm
lstm = nn.LSTMCell(10, 20)
# сгенерируем случайную последовательность чисел
input = torch.randn(seq_len, 3, 10)
hx, cx = torch.randn(3, 20), torch.randn(3, 20)

# пропустим сгенерированную последовательность через LSTM
for i in range(seq_len):
    hx, cx = lstm(input[i], (hx, cx))
print(hx.shape, cx.shape)


# Аналогично происходит работа с полной LSTM. LSTM возвращает `output`, `h_n`, `c_n`, где `output` полностью аналогичен тому, что был у RNN, то есть является последовательностью из `h_t` для последнего слоя.

# In[ ]:


seq_len = 5
batch = 3
input_size = 10
layers_num = 2
hidden_size = 20

lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
input = torch.randn(seq_len, batch, input_size)
h0 = torch.randn(layers_num, batch, hidden_size)
c0 = torch.randn(layers_num, batch, hidden_size)
output, (hn, cn) = lstm(input, (h0, c0))
print(output.shape, hn.shape, cn.shape)


# #### 3. GRU
# 
# Наборы параметров классов `GRUCell` и `GRU` и размерности возвращаемых тензоров совпадают соответсвенно с параметрами `RNNCell` и `RNN`. 
# 

# ## **Задание** 2.3

# **Обучение нейросети (сверточной и рекуррентной):**
# 
# Предположим мы хотим обучать нейросеть, используя $\tau$ предыдущих измерений. Тогда каждую эпоху мы будем передавать сети батчи из последовательных отрезков временного ряда с индексами от $t - \tau + 1$ до $t$ и для каждого такого отрезка будем просить предсказать значение с индектом $t + 1$, где $t$ &mdash; момент времени, $t > \tau - 1$.
# 
# Иначе говоря, нейронная сеть будет приближать функцию
# $$X_{t-\tau+1}, ..., X_t \mapsto X_{t+1}.$$
# 
# **Вычисление предсказаний:**
# 
# * Возьмем $\tau$ последних обучающих данных и на их основе вычислим прогноз в первый момент времени тестового отрезка ряда. 
# * Последующие прогнозы будем делать на основе предыдущих прогнозов, не используя значения из тестового отрезка ряда.
# 
# Вспомогательные функции для обучения.

# In[ ]:


from IPython.display import clear_output
from collections import defaultdict
import time

def plot_learning_curves(history):
    '''
    Функция для обучения модели и вывода лосса и метрики во время обучения.

    :param history: (dict)
        accuracy и loss на обучении и валидации.
    '''

    fig = plt.figure(figsize=(10, 5))
    plt.plot(history['loss'])
    plt.ylabel('Лосс')
    plt.xlabel('Эпоха')
    plt.show()


def train_ts_model(
    model,
    criterion,
    optimizer,
    train_batch_gen,
    num_epochs=50,
):
    '''
    Функция для обучения модели и вывода лосса во время обучения.

    :param model: обучаемая модель
    :param criterion: функция потерь
    :param optimizer: метод оптимизации
    :param train_batch_gen: генератор батчей для обучения
    :param num_epochs: количество эпох

    :return: обученная модель
    :return: (dict) loss на обучении ('история' обучения)
    '''

    history = defaultdict(list)

    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        start_time = time.time()

        # Устанавливаем поведение dropout / batch_norm  в обучение
        model.train(True)

        # На каждой 'эпохе' делаем полный проход по данным
        for X_batch, y_batch in train_batch_gen:
            X_batch = X_batch.type('torch.FloatTensor').to(device)
            y_batch = y_batch.type('torch.FloatTensor').to(device)

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += np.sum(loss.detach().cpu().numpy())

        # Подсчитываем лосс и сохраням в 'историю'
        train_loss /= len(train_batch_gen)
        history['loss'].append(train_loss)

        clear_output()

        # Печатаем результаты после каждой эпохи
        print('Эпоха {} из {} выполнена за {:.3f} сек.'.format(
            epoch + 1, num_epochs, time.time() - start_time))
        print('  лосс: \t{:.6f}'.format(train_loss))
        plot_learning_curves(history)

    return model, history


# Функция для рекуррентного предсказания сетью.

# In[ ]:


def evaluate_ts_model(model, start_seq, test_data, return_all=False):
    '''
    Функция для проверки качества модели на обучающем отрезке ряда.

    :param model: обучаемая модель,
    :param start_seq: обучающие данные для первого предсказания,
    :param test_data: тестовые данные.
    :param return_all: возвращать все предсказания или только для 1-го магазина

    :return: результаты предсказания.
    '''
    result = []

    input_tensor = torch.FloatTensor(start_seq).to(device).unsqueeze(0)
    with torch.no_grad():
        for i in range(len(test_data)):
            logits = model(input_tensor[:, i:, :]).unsqueeze(0)
            input_tensor = torch.cat((input_tensor, logits), 1)
            result.append(logits.cpu().numpy().squeeze())
    if return_all:
      return np.array(result)
    return np.array(result)[:, 0]


# Вернемся к данным. Нейронной сетью мы будем приближать функцию 
# 
# $$X_{t-\tau+1}, ..., X_t \mapsto X_{t+1}.$$
# 
# где $ X_t$ - некоторый вектора. В предыдущей части мы использовали данные по товару 1 только для 1 магазина. Однако обучать нейронку на векторах размера 1 не очень интересно, поэтому будем учить нейронную сеть предсказывать продажи товара 1 сразу для всех магазинов. 
# 
# На этапе тестирования (для магазина 1) для этой нейронки нужно брать только одну компоненту вектора (см. функцию выше).
# 
# Подготовим данные для обучения.

# In[ ]:


data = pd.read_csv('train.csv', parse_dates=['date'])
data.head()

# Задаем продукт
item = 1

# Выделяем только те данные, которые относятся к данному продукту
data = data[data['item'] == item]

# ВНИМАНИЕ: Дату уставнавливаем как индекс
data = data.set_index('date')


# Соберем в один dataframe продажи для всех магазинов

# In[ ]:


processed_data = pd.DataFrame()
for store in data['store'].unique():
  processed_data[f'store_{store}_sales'] = data[(data['store'] == store) & (data['item'] == item)]['sales']


# In[ ]:


processed_data.head()


# Проверим, что количество дат совпадает с прошлым разом.

# In[ ]:


processed_data.shape


# Разделим на трейн и тест, как и раньше.

# In[ ]:


processed_data_train = processed_data.iloc[:-test_size]
processed_data_test = processed_data.iloc[-test_size:]


# Для обучения нужно создать датасет, который будет возвращать последовательность из $\tau$ элементов временного ряда для обучения и следующий за ними $(\tau+1)$-элемент как таргет. Для этого мы создадим класс-наследник от класса `Dataset` из [`torch.utils.data`](https://pytorch.org/docs/stable/data.html).
# 
# При наследовании от `Dataset` необходимо переопределить `__len__` и `__getitem__`.

# In[ ]:


class TSDataset(torch.utils.data.Dataset):
    def __init__(self, data, n_past):
        self.data = data
        self.n_past = n_past  # tau -- длина отрезка временого ряда

    def __len__(self):
        return self.data.shape[0] - self.n_past

    def __getitem__(self, index):
        return self.data[index: self.n_past + index], self.data[self.n_past + index]


# ### **Задание** 2.3.1. Сверточная сеть

# Определим простую сверточную сеть для предсказания следующего элемента.

# In[ ]:


class Conv1dModel(nn.Module):
    def __init__(self, in_channels, output_size):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels=32, kernel_size=30, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=8, stride=2)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU()
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        # вместо global pooling можно использовать flatten, но тогда линейный слой будет сильно завязан на размер последовательности
        # self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=128, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=output_size)

    def forward(self, input_seq):
        # input_seq.shape = [bs, seq_len, 10]

        # input_seq.shape = [bs, 10, seq_len]
        input_seq = input_seq.transpose(1, 2)

        # x.shape = [bs, 32, seq_len_1]
        x = self.relu1(self.bn1(self.conv1(input_seq)))

        # x.shape = [bs, 64, seq_len_2]
        x = self.relu2(self.bn2(self.conv2(x)))

        # x.shape = [bs, 128, seq_len_3]
        x = self.relu3(self.bn3(self.conv3(x)))

        # x.shape = [bs, 128]
        x = self.pool(x).squeeze(2)

        # x.shape = [bs, 128]
        x = self.relu(self.fc1(x))

        # x.shape = [bs, 2]
        x = self.fc2(x)
        return x


# ### **Задание** 2.3.1. Сверточная сеть

# Определим простую сверточную сеть для предсказания следующего элемента.

# In[ ]:


class Conv1dModel(nn.Module):
    def __init__(self, in_channels, output_size):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels=32, kernel_size=30, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=8, stride=2)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU()
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        # вместо global pooling можно использовать flatten, но тогда линейный слой будет сильно завязан на размер последовательности
        # self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=128, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=output_size)

    def forward(self, input_seq):
        # input_seq.shape = [bs, seq_len, 10]

        # input_seq.shape = [bs, 10, seq_len]
        input_seq = input_seq.transpose(1, 2)

        # x.shape = [bs, 32, seq_len_1]
        x = self.relu1(self.bn1(self.conv1(input_seq)))

        # x.shape = [bs, 64, seq_len_2]
        x = self.relu2(self.bn2(self.conv2(x)))

        # x.shape = [bs, 128, seq_len_3]
        x = self.relu3(self.bn3(self.conv3(x)))

        # x.shape = [bs, 128]
        x = self.pool(x).squeeze(2)

        # x.shape = [bs, 128]
        x = self.relu(self.fc1(x))

        # x.shape = [bs, 2]
        x = self.fc2(x)
        return x


# Заметим, что у нас в данных есть явно два вида колебаний: с большим периодом около года и более быстрые, поэтому брать tau слишком маленьким (в районе недели) бессмысленно. Поставим сначала большую длину отрезка временного ряда.

# In[ ]:


n_past = 120  # tau -- длина отрезка временого ряда
batch_size = 64  # размер батча

train_dataset = TSDataset(processed_data_train.to_numpy(), n_past)
train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)


# In[ ]:


device = 'cpu'


# In[ ]:


model = Conv1dModel(10, 10)

criterion = nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)


# In[ ]:


model, _ = train_ts_model(
    model,
    criterion,
    optimizer,
    train_batch_gen,
    num_epochs=50,
)


# In[ ]:


model.eval()
start_seq = processed_data_train[-n_past:].to_numpy()
result = evaluate_ts_model(model, start_seq, processed_data_test)


# In[ ]:


plot_results(data_train, data_test, result)


# In[ ]:


add_results_in_comparison_table('Conv1d+data_from_other_stores+120_window', data_test, result)


# Теперь возьмем окно в 2 раза меньше, то есть 2 месяца.

# In[ ]:


n_past = 60  # tau -- длина отрезка временого ряда
batch_size = 64  # размер батча

train_dataset = TSDataset(processed_data_train.to_numpy(), n_past)
train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)


# In[ ]:


model = Conv1dModel(10, 10)

criterion = nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)


# In[ ]:


model, _ = train_ts_model(
    model,
    criterion,
    optimizer,
    train_batch_gen,
    num_epochs=50,
)


# In[ ]:


model.eval()
start_seq = processed_data_train[-n_past:].to_numpy()
result = evaluate_ts_model(model, start_seq, processed_data_test)


# In[ ]:


plot_results(data_train, data_test, result)


# In[ ]:


add_results_in_comparison_table('Conv1d+data_from_other_stores+60_window', data_test, result)


# И наконец, возьмем годовое окно.

# In[ ]:


n_past = 365  # tau -- длина отрезка временого ряда
batch_size = 64  # размер батча

train_dataset = TSDataset(processed_data_train.to_numpy(), n_past)
train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)


# In[ ]:


model = Conv1dModel(10, 10)

criterion = nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)


# In[ ]:


model, _ = train_ts_model(
    model,
    criterion,
    optimizer,
    train_batch_gen,
    num_epochs=50,
)


# In[ ]:


model.eval()
start_seq = processed_data_train[-n_past:].to_numpy()
result = evaluate_ts_model(model, start_seq, processed_data_test)


# In[ ]:


plot_results(data_train, data_test, result)


# In[ ]:


add_results_in_comparison_table('Conv1d+data_from_other_stores+365_window', data_test, result)


# Сделайте вывод, какая из моделей как ведет себя на различных временных промежутках и на различных колебаниях.
# 
# **ваш ответ** +- хорошо свёртка работает на 2 месяцах, но нигде не предсказан пик
# 
# На основании данных графиков сделайте вывод о преимуществах и недостатках рекурсивной стратегии, а также об идее иметь несколько моделей на различные горизонты прогноза.
# 
# **ваш ответ** Из преимуществ: быстро учится и может предсказать общий паттерн, недостатки - нужно иметь много разных моделей.
# 
# ---

# ### **Задание** 2.3.2. Рекуррентная сеть
# 
# Попробуем обучить рекуррентную сеть.
# 
# Для рекуррентной сети будем использовать 2 стратегии
# 
# * предсказание будем делать только по последнему выходу RNN для всей последовательности
# * предсказание будем делать по усредненному выходу RNN для всей последовательности

# In[ ]:


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100, use_pool=False):
        super().__init__()
        self.hidden_size = hidden_size

        # input_seq.shape = [bs, tau, 10]
        # lstm_out.shape = [bs, tau, 100]
        self.bn = nn.BatchNorm1d(10)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size, batch_first=True)

        # hidden_state = [1, bs, 100]
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=output_size)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.use_pool = use_pool

    def forward(self, input_seq):
        input_seq = input_seq.transpose(1, 2)
        input_seq = self.bn(input_seq)
        input_seq = input_seq.transpose(1, 2)

        # lstm_out.shape = [bs, tau, 100] - последовательность скрытых состояний для всех моментов времени
        # hidden_state = [1, bs, 100] - последнее скрытое состояние для последовательности
        lstm_out, (hidden_state, _) = self.lstm(input_seq)

        if self.use_pool:
          # берем среднее от векторов для всей последовательности
          seq_vec = self.pool(lstm_out.transpose(1, 2)).squeeze(2)
        else:
          # берем последний вектор
          seq_vec = hidden_state.squeeze(0)

        predictions = self.linear(seq_vec)
        return predictions


# Сначала только по последнему элементу.

# In[ ]:


n_past = 90  # tau -- длина отрезка временого ряда
batch_size = 32  # размер батча

train_dataset = TSDataset(processed_data_train.to_numpy(), n_past)
train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)


# In[ ]:


lstm_model = LSTM(10, 10, use_pool=False)

criterion = nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.Adam(params=lstm_model.parameters(), lr=lr)


# In[ ]:


lstm_model, _ = train_ts_model(
    lstm_model,
    criterion,
    optimizer,
    train_batch_gen,
    num_epochs=50,
)


# Сравните скорость обучения RNN со скоростю обучения сверточной сети.
# 
# **ваш ответ** RNN обучается дольше, чем свёрточная сеть (по ощущениям раза в 2).

# In[ ]:


lstm_model.eval()
start_seq = processed_data_train[-n_past:].to_numpy()
result = evaluate_ts_model(lstm_model, start_seq, processed_data_test)


# In[ ]:


plot_results(data_train, data_test, result)


# In[ ]:


add_results_in_comparison_table('LSTM+data_from_other_stores_last+hidden_state', data_test, result)


# На самом деле, внутри RNN используются функции активации типа сигмоиды, которые чувствительны с масштабу и центрированию данных. Попробуем добавить нормализацию данных.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
processed_data_train_normalized = scaler.fit_transform(processed_data_train)

processed_data_train_normalized


# In[ ]:


n_past = 90  # tau -- длина отрезка временого ряда
batch_size = 32  # размер батча

train_dataset = TSDataset(processed_data_train_normalized, n_past)
train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)


# In[ ]:


lstm_model = LSTM(10, 10, use_pool=False)

criterion = nn.MSELoss()
lr=0.5e-4
optimizer = torch.optim.Adam(params=lstm_model.parameters(), lr=lr)


# In[ ]:


lstm_model, _ = train_ts_model(
    lstm_model,
    criterion,
    optimizer,
    train_batch_gen,
    num_epochs=50,
)


# In[ ]:


lstm_model.eval()
start_seq = processed_data_train_normalized[-n_past:]
result = evaluate_ts_model(lstm_model, start_seq, processed_data_test, return_all=True)


# In[ ]:


result.shape


# In[ ]:


result = scaler.inverse_transform(result)[:, 0]


# In[ ]:


plot_results(data_train, data_test, result)


# In[ ]:


add_results_in_comparison_table('LSTM+data_from_other_stores_last+hidden_state+nolmalized', data_test, result)


# В начале...

# Теперь добавим усреднение всех скрытых состояний.

# In[ ]:


n_past = 90  # tau -- длина отрезка временого ряда
batch_size = 32  # размер батча

train_dataset = TSDataset(processed_data_train_normalized, n_past)
train_batch_gen = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)


# In[ ]:


lstm_model = LSTM(10, 10, use_pool=True)

criterion = nn.MSELoss()
lr=0.5e-4
optimizer = torch.optim.Adam(params=lstm_model.parameters(), lr=lr)


# In[ ]:


lstm_model, _ = train_ts_model(
    lstm_model,
    criterion,
    optimizer,
    train_batch_gen,
    num_epochs=50,
)


# In[ ]:


lstm_model.eval()
start_seq = processed_data_train_normalized[-n_past:]
result = evaluate_ts_model(lstm_model, start_seq, processed_data_test, return_all=True)
result = scaler.inverse_transform(result)[:, 0]


# In[ ]:


plot_results(data_train, data_test, result)


# In[ ]:


add_results_in_comparison_table('LSTM+data_from_other_stores+mean_hidden_state+normalized', data_test, result)


# В итоге сеть хотя бы как-то обучилась, но при этом все равно не очень. Такой результат мог получиться сразу по нескольким причинам
# 
# 1. Вектор для нейросети мог быть выбран неудачно. На самом деле, если посмотреть на корреляции между продажами отдельных магазинов можно увидеть цифры около 0.4, что не так уж и много
# 2. RNN обычно плохо предсказывают такие шумные данные, как у нас, лучше работают с более плавными зависимостями
# 3. Последовательность длины 90 для RNN это довольно много, поскольку они подвержены забыванию, усреднение всех скрытых состояний помогает решить эту проблему хотя бы немного
# 4. Мы использовали RNN для задачи вида $[x_1, \dots, x_t] \to x_{t+1}$, но чаще всего их используют для задачи вида $[x_1, \dots, x_t] \to [x_2, \dots, x_{t+1}]$. В таком случае линейное преобразование применяется к каждому скрытому состоянию и лосс считается по всем векторам вместе (некоторый аналог Inception, но с одними и теми же весами). 
# 5. Если потюнить параметры, возможно можно получить результат получше
# 
# В целом данная задача показывает, что нейронные сети далеко не всегда работают лучше, чем классические модели. По бейзлайну мы видим, что здесь оказались очень важными годовая и недельная периодичность, которые уловить нашим сетями оказалось довольно сложно.

# **Теперь сохраним модель**

# In[ ]:


torch.save(model.state_dict(), '/content/drive/MyDrive/MyModel.pt')


# **Загрузим модель**

# In[ ]:


LoadModel = torch.load('/content/drive/MyDrive/MyModel.pt')


# **Посмотрим, что сохранилось**

# In[ ]:


print(LoadModel)


# Сохранение модели таким образом сохранит весь модуль, используя модуль pickle в Python. 
# 
# Недостатком этого подхода является то, что сериализованные данные привязаны к конкретным классам и точной структуре каталогов, используемой при сохранении модели. Причина этого заключается в том, что pickle не сохраняет сам класс модели. Скорее всего, он сохраняет путь к файлу, содержащему класс, который используется во время загрузки. Из-за этого наш код может по-разному ломаться при использовании в других проектах или после рефакторинга.

# In[ ]:


import time
def save(file_name):
    def decor(func):
        def wrapper(*args, **kwargs):
            start=time.time()
            with open(file_name, 'a') as f:
              f.write( 'start at ' + time.ctime() + '\n')
            res=func(*args, **kwargs)
            res_time=time.time()-start
            with open(file_name, 'a') as f:
              f.write(str(res) + '\n' + str(res_time) + '\n')

            return res, res_time
        return wrapper
    return decor

@save('Decor.txt')
def sum(a, b, c):
    s = 0
    for i in range (c):
        s += (a+b)*c
    return s


# In[ ]:


sum(2, 3, 10000)


# In[ ]:




