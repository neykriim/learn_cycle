#to_datetime(regexp_substr( archive,'\\d{8}T\\d{6}'),'yyyyMMddTHHmmss')
from datetime import datetime
from qgis.core import QgsProject, QgsField, QgsFeature
from qgis.PyQt.QtCore import QVariant, QDateTime, QDate

# Получаем слои
layer1 = QgsProject.instance().mapLayersByName('extents')[0]
layer2 = QgsProject.instance().mapLayersByName('карта — temp_fixed')[0]

# Поля с датами
date_field_layer1 = 'datetime'  # Замените на имя поля с датой и временем в первом слое
date_field_layer2 = 'date'      # Замените на имя поля с датой и временем во втором слое
new_field_layer1 = 'near'     # Имя нового поля для записи ближайшей даты (без времени)

# Добавляем новое поле в первый слой, если его еще нет
if layer1.fields().indexFromName(new_field_layer1) == -1:
    layer1.dataProvider().addAttributes([QgsField(new_field_layer1, QVariant.Date)])  # Тип QVariant.Date для даты без времени
    layer1.updateFields()

# Индексы полей
date_idx_layer1 = layer1.fields().indexFromName(date_field_layer1)
date_idx_layer2 = layer2.fields().indexFromName(date_field_layer2)
new_field_idx = layer1.fields().indexFromName(new_field_layer1)

# Создаем список дат и времени из второго слоя
datetimes_layer2 = []
for feat in layer2.getFeatures():
    datetime_val = feat[date_idx_layer2]
    if isinstance(datetime_val, QDateTime):
        datetime_val = datetime_val.toPyDateTime()  # Преобразуем QDateTime в datetime
    elif isinstance(datetime_val, QDate):
        datetime_val = QDateTime(datetime_val).toPyDateTime()  # Преобразуем QDate в datetime
    if datetime_val:
        datetimes_layer2.append(datetime_val)

# Находим ближайшую дату и время для каждого объекта в первом слое
for feat in layer1.getFeatures():
    current_datetime = feat[date_idx_layer1]
    if isinstance(current_datetime, QDateTime):
        current_datetime = current_datetime.toPyDateTime()  # Преобразуем QDateTime в datetime
    elif isinstance(current_datetime, QDate):
        current_datetime = QDateTime(current_datetime).toPyDateTime()  # Преобразуем QDate в datetime

    if current_datetime:
        nearest_datetime = None
        min_diff = None
        for datetime_val in datetimes_layer2:
            diff = abs((current_datetime - datetime_val).total_seconds())  # Вычисляем разницу в секундах
            if min_diff is None or diff < min_diff:
                min_diff = diff
                nearest_datetime = datetime_val

        if nearest_datetime:
            # Преобразуем nearest_datetime в QDate (без времени)
            nearest_date = nearest_datetime.date()
            layer1.startEditing()
            feat.setAttribute(new_field_idx, QDate(nearest_date))  # Записываем только дату
            layer1.updateFeature(feat)
            layer1.commitChanges()

print("Готово!")