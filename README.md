# Мастерская №8 Обработка естественных языков на ДНК и белках.

В папке `src/` лежат алгоритмы, которые мы разобрали во время обучения, в папке `notebooks/` - jupyter ноутбуки с кодом для интерактивного выполнения. 
Целью было разобрать классические алгоритмы машинного обучения и затем попробовать их на [белках](https://ru.wikipedia.org/wiki/%D0%9F%D0%B5%D1%80%D0%B2%D0%B8%D1%87%D0%BD%D0%B0%D1%8F_%D1%81%D1%82%D1%80%D1%83%D0%BA%D1%82%D1%83%D1%80%D0%B0).
У проекта было 2 части: научиться предсказывать вторичную структуру белка по его последовательности и научиться классифицировать белки по принадлежности к разным таксонам по его последовательности.
Они лежат в файлах в папке `notebooks/`. Мы не успели упаковать их в консольный интерфейс, вы можете запустить файлы по ссылке.

Все данные уже лежат в папке `data/`. Вторичные структуры взяты [отсюда](https://github.com/zyxue/pdb-secondary-structure).

# Сборка и запуск

Зависимости: Python и библиотеки `pytorch, torchtext, sklearn, numpy, pandas, matplotlib, seaborn, tqdm`.

Или можно открыть блокноты в Google Colab. Запустите блокнот и затем нажмите `Среда выполнения -> выполнить все`.

- Разобранные алгоритмы
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nkgrush/SSYP2022-ws8/blob/master/notebooks/models.ipynb)
- Предсказание таксона
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nkgrush/SSYP2022-ws8/blob/master/notebooks/TaxonClassification.ipynb)
- Визуализация датасета со вторичными структурами
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nkgrush/SSYP2022-ws8/blob/master/notebooks/Visualization.ipynb)
- Предсказание вторичной структуры 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nkgrush/SSYP2022-ws8/blob/master/notebooks/SecondaryStructure.ipynb)


# Состав мастерской
## Ученики
- Закиров Тимур
- Плюснин Антон
- Матюшин Никита
- Васильченко Михаил
- Притупов Вадим


## Руководитель
- Грушецкий Николай Артёмович
