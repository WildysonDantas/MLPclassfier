import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Carregando data
data = pd.read_csv('iris.data')
data.columns = ['a', 'b', 'c', 'd', 'class']

# Convertendo as classes em numeros
le = preprocessing.LabelEncoder()
data['class'] = le.fit_transform(data['class'])

# Divindo a features em dois grupos
x = data[['a', 'b', 'c', 'd']]
y = data['class']

# Dividindo o dataset em treiamnto e teste. 70% trieamento e 30% teste
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

# Criando um modelo
clf = MLPClassifier(hidden_layer_sizes=(6, 5),
                    random_state=4,
                    verbose=True,
                    learning_rate_init=0.01)

# adcionando o treimaneto no modelo
clf.fit(X_train, y_train)

# Predict do modelo com um dataset de teste
ypred = clf.predict(X_test)

# Acuracia do modelo
accuracy = accuracy_score(y_test, ypred)
print(accuracy)
