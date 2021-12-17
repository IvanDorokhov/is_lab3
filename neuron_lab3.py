#region импорт зависимостей
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
#endregion

#region обучение нейронной сети

# Метод генерации выборки
def gen_selection(length, min, max):
    input = np.zeros((length, 2), dtype=int)
    output = np.zeros((length, 1), dtype=int)
    for i in range(0, length):
        a = random.randint(min, max)
        b = random.randint(min, max)
        input[i][0] = a
        input[i][1] = b
        output[i][0] = a+b
    return (input, output)

# обучающая выборка
(train_input, train_output) = gen_selection(10000, 0, 10000)

# тестовая выборка
(test_input, test_output) = gen_selection(100, 0, 10000)


model = keras.Sequential([
    # выравнивание входного массива в вектор
    keras.layers.Flatten(input_shape=(2,)),
    # 2 уровень сети состоит из 2 узлов
    # функция активации relu - выпрямленная линейная единица
    keras.layers.Dense(2, activation=tf.nn.relu),
    # т.к. ожидается 1 выходное значение (прогнозируемое значение, т.к. это регресионная модель), следовательно только один выходной узел
    keras.layers.Dense(1)
])

# компиляция сети
# adam - функция оптимизации (оптимизатор на основе импульса и предотвращает застревание модели в локальных минимумах)
# mse - функция потерь (среднеквадратическая ошибка) Квадратичная разница между прогнозируемым и фактическим значением
# mae - средняя абсолютная ошибка
model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae'])

# обучение сетей
# обучающий набор будет подаваться в сеть 3 раз (мало эпох- недоученность, много эпох - переобучение)
model.fit(train_input, train_output, epochs=3, batch_size=1)


#endregion

# оценивание обученной модели на тестовом наборе данных
test_loss, test_acc = model.evaluate(test_input, test_output)
# вывод значения точности теста
print('Погрешность при тестировании:', test_acc)

# подставление реальных значений
input = np.array([[2000,3000],[4,5], [2,3],[400,500]])
output = model.predict(input)

for i in range(0, 4):
    print(input[i][0], '+', input[i][1], '~=', output[i][0])