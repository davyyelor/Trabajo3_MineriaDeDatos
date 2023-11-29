import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import string
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

suicide_detection = pd.read_csv("Suicide_Detection.csv", sep=",", encoding="utf-8")

suicide_detection['class'] = suicide_detection['class'].replace({'non-suicide': 0, 'suicide': 1})

# Descargar los recursos necesarios para NLTK
nltk.download('punkt')
nltk.download('stopwords')

suicidal_data = pd.read_csv("suicidal_data.csv", sep=",", encoding='cp1252')

# Divide el conjunto de datos en train y test
train, test_data = train_test_split(suicidal_data, test_size=0.2, random_state=42)
tweets_test = test_data['tweet']
labels_test = test_data['label']


fscore = []

######################## Preproceso del test
# Preprocesamiento del texto
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convierte a minúsculas
    text = text.lower()
    
    # Tokenización
    tokens = word_tokenize(text)
    
    # Eliminación de stopwords y signos de puntuación
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    return tokens


test_data['processed_text'] = test_data['tweet'].apply(preprocess_text)

# Entrena el modelo Word2Vec
model_w2v = Word2Vec(sentences=test_data['processed_text'], vector_size=100, window=5, min_count=1, workers=4)

# Transforma los textos preprocesados en vectores usando Word2Vec
def text_to_vectors(text):
    vectors = []
    for word in text:
        if word in model_w2v.wv:
            vectors.append(model_w2v.wv[word])
    return vectors

test_data['vectors'] = test_data['processed_text'].apply(text_to_vectors)  
print("Tamaño del conjunto de prueba: ", len(test_data))

test_data['vectors_avg'] = test_data['vectors'].apply(lambda vectors: np.mean(vectors, axis=0) if vectors else np.zeros(100))




for x in range(1,9):
    if x == 1:
        proceso = "Borrado de 4000 instancias"
        suicidal_data = train
        # Quítale 4000 instancias a suicidal_data
        suicidal_data = suicidal_data.iloc[:-4000]
    if x == 2:
        proceso = "Borrado de 2000 instancias"
        suicidal_data = train
        # Quítale 2000 instancias a suicidal_data
        suicidal_data = suicidal_data.iloc[:-2000]
    if x == 3:
        proceso = "Conjunto original"
        suicidal_data = train
            # Usa el conjunto original de suicidal_data
    if x == 4:
        proceso = "Inserción de 2000 instancias"
        suicidal_data = train
        # Añade las instancias de suicide_detection a las columnas label y tweet de suicidal_data
        suicidal_data = pd.concat([suicidal_data, suicide_detection[['text', 'class']].rename(columns={'text': 'tweet', 'class': 'label'})[:2000]], ignore_index=True)
    if x == 5:
        proceso = "Inserción de 4000 instancias"
        suicidal_data = train
        suicidal_data = pd.concat([suicidal_data, suicide_detection[['text', 'class']].rename(columns={'text': 'tweet', 'class': 'label'})[:4000]], ignore_index=True)
    if x == 6:
        proceso = "Inserción de 6000 instancias"
        suicidal_data = train
        suicidal_data = pd.concat([suicidal_data, suicide_detection[['text', 'class']].rename(columns={'text': 'tweet', 'class': 'label'})[:6000]], ignore_index=True)
    if x == 7:
        proceso = "Inserción de 8000 instancias"
        suicidal_data = train
        suicidal_data = pd.concat([suicidal_data, suicide_detection[['text', 'class']].rename(columns={'text': 'tweet', 'class': 'label'})[:8000]], ignore_index=True)
    if x == 8:
        proceso = "Inserción de 10000 instancias"
        suicidal_data = train
        suicidal_data = pd.concat([suicidal_data, suicide_detection[['text', 'class']].rename(columns={'text': 'tweet', 'class': 'label'})[:10000]], ignore_index=True)
        suicidal_data.to_csv("nuevo.csv", index=False)
            
    print("###############################################################################################################################")
    print("#####################################    Iteración nº: ", x, "-> ", proceso, " #####################################")
    print("###############################################################################################################################")

    print(suicidal_data['label'].value_counts(), end="\n")



    # Divide el conjunto de datos en train y test
    train_data = suicidal_data['tweet']
    train_labels = suicidal_data['label']

    # Preprocesamiento del texto
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        # Convierte a minúsculas
        text = text.lower()
        
        # Tokenización
        tokens = word_tokenize(text)
        
        # Eliminación de stopwords y signos de puntuación
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        
        return tokens

    # Aplica el preprocesamiento a los datos de entrenamiento y prueba
    train_data['processed_text'] = train_data.apply(preprocess_text)

    # Entrena el modelo Word2Vec
    model_w2v = Word2Vec(sentences=train_data['processed_text'], vector_size=100, window=5, min_count=1, workers=4)

    # Transforma los textos preprocesados en vectores usando Word2Vec
    def text_to_vectors(text):
        vectors = []
        for word in text:
            if word in model_w2v.wv:
                vectors.append(model_w2v.wv[word])
        return vectors

    train_data['vectors'] = train_data['processed_text'].apply(text_to_vectors)

    print("Tamaño del conjunto de entrenamiento: ", len(train_data))    

    train_data['vectors_avg'] = train_data['vectors'].apply(lambda vectors: np.mean(vectors, axis=0) if vectors else np.zeros(100))

# Utilizar train_data['vectors_avg'] y test_data['vectors_avg'] para el clustering con KMeans


    # Train KMeans model
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(train_data['vectors_avg'].tolist())

    # Predict test data
    test_predictions = kmeans.predict(test_data['vectors_avg'].tolist())

    # Calculate F1 score
    f1score = f1_score(labels_test, test_predictions)

    print("F1 Score:", f1score)
    fscore.append(f1score)

    print("")

import matplotlib.pyplot as plt

instances = [5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000]  # Replace with your actual fscore values

plt.plot(instances, fscore, marker='o')
plt.xlabel('Number of Instances')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Number of Instances')
plt.xticks(instances)
plt.yticks([i/10 for i in range(11)])
plt.grid(True)
plt.show()


    

