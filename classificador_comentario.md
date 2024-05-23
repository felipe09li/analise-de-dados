import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Downloads do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Define o corpus de stopwords para o português(Exemplos de stopwords em português incluem "e", "de", "para", "o", "a", entre outras.)
stop_words = set(stopwords.words('portuguese'))

# Função para realizar pré-processamento
def preprocess_text(text):
    if isinstance(text, str):
        # Remover pontuações e caracteres especiais
        text = re.sub(r'[^\w\s]', '', text)
        # Converter para minúsculas
        text = text.lower()
        # Tokenização
        tokens = word_tokenize(text)
        # Remover stopwords
        tokens = [word for word in tokens if word not in stop_words]
        # Reunir os tokens em uma única string
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    else:
        return 'N/A'

# Carregar os dados do arquivo Excel
treinamento = 'caminho/arquivo.xlsx'
df = pd.read_excel(treinamento)

# Verificar e lidar com valores nulos
df = df.dropna(subset=['comentario'])  # Remover linhas com valores nulos na coluna 'comentario'
# Nome das colunas que contêm os comentários e as categorias
comentario_col = 'comentario'
categoria_col = 'reclamacao'

# Dados de exemplo (comentários e suas respectivas categorias)
comentarios = df[comentario_col].tolist()
categorias = df[categoria_col].tolist()

# Aplicar pré-processamento aos comentários
comentarios_preprocessados = [preprocess_text(comment) for comment in comentarios]

# Tokenização dos textos pré-processados
tokenizer = Tokenizer(num_words=1800)
tokenizer.fit_on_texts(comentarios_preprocessados)
sequences = tokenizer.texts_to_sequences(comentarios_preprocessados)
maxlen = max([len(x) for x in sequences])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Sequências de entrada padronizadas
X = pad_sequences(sequences, maxlen=maxlen)

# Conversão das categorias para números
categoria_para_numero = {cat: i for i, cat in enumerate(set(categorias))}
y = [categoria_para_numero[cat] for cat in categorias]
y = np.array(y)

# Divisão dos dados em treino e teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43, stratify=y)

# Construção do modelo de rede neural com RNN
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 1800, input_length=maxlen),  # Dimensão da embedding reduzida
    tf.keras.layers.LSTM(120),  # Camada LSTM com 120 unidades
    tf.keras.layers.Dense(1800, activation='relu'),
    tf.keras.layers.Dense(len(set(categorias)), activation='softmax')
])

# Compilação do modelo
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(X_train, y_train, epochs=40, batch_size=60, validation_data=(X_test, y_test))  # Reduzi o número de épocas e tamanho do lote

# Avaliação do modelo
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Fazendo previsões com o conjunto de teste
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Comparando as previsões com as respostas corretas
correct_predictions = np.sum(y_pred_classes == y_test)
total_samples = len(y_test)
accuracy = correct_predictions / total_samples
print("Accuracy on test set:", accuracy)

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

# Relatório de Classificação
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, zero_division=1))

# Carregar os dados do arquivo Excel
trabalho = 'caminho/arquivo.xlsx'
df_novos = pd.read_excel(trabalho)

# Nome da coluna que contém os comentários
comentario_col = 'comentario'

# Remover linhas nulas do DataFrame
df_novos = df_novos.dropna(subset=[comentario_col])

# Novos dados para prever
novos_comentarios = df_novos[comentario_col].tolist()

# Convert any non-string elements in novos_comentarios to strings
novos_comentarios = [str(comment) for comment in novos_comentarios]

# Fazendo previsões com os novos dados
novas_sequences = tokenizer.texts_to_sequences(novos_comentarios)
X_novos = pad_sequences(novas_sequences, maxlen=maxlen)
y_novos_pred = model.predict(X_novos)
y_novos_pred_classes = np.argmax(y_novos_pred, axis=1)

# Mapear números de classe de volta para categorias
numero_para_categoria = {v: k for k, v in categoria_para_numero.items()}
categorias_previstas = [numero_para_categoria[num] for num in y_novos_pred_classes]

# Adicionando as previsões ao DataFrame
df_novos['Predição'] = categorias_previstas

# Salvar o DataFrame modificado em um novo arquivo Excel
df_novos.to_excel('aequivo.xlsx', index=False)
