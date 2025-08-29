import pickle
import numpy as np
import streamlit as st
from sklearn.datasets import load_iris
from PIL import Image  # Biblioteca Pillow para manipular imagens

# Título da aplicação
st.title("Previsão de Espécie - Base Iris")

# Carregar o modelo salvo
with open('qda_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Carregar os dados da base Iris para obter os nomes das features e classes
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Interface para entrada de dados
st.write("### Insira os valores das características")
input_data = []
for feature in feature_names:
    value = st.number_input(f"Insira o valor para {feature}", min_value=0.0, max_value=10.0, step=0.1)
    input_data.append(value)

# Botão de previsão
if st.button("Realizar Previsão"):
    input_data = np.array(input_data).reshape(1, -1)  # Converter para formato adequado
    prediction = loaded_model.predict(input_data)[0]  # Prever classe
    prediction_label = target_names[prediction]  # Nome da classe prevista

    # Exibir o resultado da previsão
    st.write(f"### Resultado da Previsão: **{prediction_label.capitalize()}**")

    # Exibir a imagem correspondente à previsão
    image_path = f"imagens/{prediction_label}.jpg"  # Caminho da imagem baseado no nome da classe

    try:
        image = Image.open(image_path)  # Tenta abrir a imagem
        st.image(image, caption=f"Espécie: {prediction_label.capitalize()}")
    except FileNotFoundError:
        st.write("Imagem não encontrada para esta espécie.")
    except Exception as e:
        st.write(f"Erro ao carregar a imagem: {e}")
