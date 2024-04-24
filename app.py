import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import DropFeatures, OneHotEncodingNames, OrdinalFeature, MinMaxWithFeatNames
import joblib

dados = pd.read_csv("df_clean.csv")

def data_split(df, test_sz):
    SEED=1561651
    treino_df, teste_df = train_test_split(df, test_size=test_sz, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

##### CAMPOS DE INPUT DO USUARIO
st.write("# Simulador de avaliação de crédito")

st.write("### Idade")
input_idade = float(st.slider("Selecione sua idade",18,100))
#st.write(f"Idade: {input_idade}")

st.write("### Nível de escolaridade")
input_grau_escolaridade = st.selectbox("Qual é seu grau de escolaridade?", dados["Grau_escolaridade"].unique())

st.write("### Estado civil?")
input_estado_civil = st.selectbox("Qual é o seu estado civil?", dados['Estado_civil'].unique())

st.write("### Familia")
input_membros_familia = float(st.slider("Selecione quantos membros tem na sua familia",1,20))

st.write("### Carro próprio")
input_carro_proprio = st.radio("Possui carro próprio?", ["Sim","Não"])
input_carro_proprio_d = {"Sim":1,"Não":0}
input_carro_proprio = input_carro_proprio_d.get(input_carro_proprio)
#st.write(f"carro: {input_carro_proprio_d}")

st.write("### Casa própria")
input_casa_propria = {"Sim":1,"Não":0}.get(st.radio("Possui casa própria?", ["Sim","Não"]))
#st.write(f"casa: {input_casa_propria}")

st.write("### Tipo de residência")
input_tipo_moradia = st.selectbox("Qual é seu tipo de residência?", dados["Moradia"].unique())

st.write("### Categoria de renda")
input_categoria_renda = st.selectbox("Qual é a sua categoria de renda?", dados["Categoria_de_renda"].unique())

st.write("### Ocupação")
input_ocupacao = st.selectbox("Qual é a sua ocupação?", dados["Ocupacao"].unique())

st.write("### Experiência")
input_anos_empregado = float(st.slider("Quantos anos de experiência?",0,40))

st.write("### Rendimentos")
input_rendimento = float(st.number_input("Informe seu rendimento anual",0,1000000,0))

st.write("### Telefone corporativo")
input_telefone_trabalho = {"Sim":1,"Não":0}.get(st.radio("Possui telefone corporativo?", ["Sim","Não"]))

st.write("### Telefone fixo")
input_telefone_fixo = {"Sim":1,"Não":0}.get(st.radio("Possui telefone fixo?", ["Sim","Não"]))

st.write("### Email")
input_email = {"Sim":1,"Não":0}.get(st.radio("Possui email?", ["Sim","Não"]))


#### EXECUÇÃO DO MODELO
novo_cliente = [0, input_carro_proprio, input_casa_propria, input_telefone_trabalho, input_telefone_fixo,
                input_email, input_membros_familia, input_rendimento, input_idade, input_anos_empregado,
                input_categoria_renda, input_grau_escolaridade, input_estado_civil, input_tipo_moradia, input_ocupacao,0
            ]

treino_df, teste_df = data_split(dados, 0.2)

cliente_predict_df = pd.DataFrame([novo_cliente], columns=teste_df.columns)

teste_novo_cliente = pd.concat([teste_df, cliente_predict_df], ignore_index=True)

def pipeline_teste(df):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()), #Excluir coluna de ID
        ('OneHotEncoding', OneHotEncodingNames()), #Atribuir valor para as colunas categoricas
        ('ordinal_feature', OrdinalFeature()), #Atribuir valor para as colunas categoricas com ordem
        ('min_max_scaler', MinMaxWithFeatNames()) #Normalizar campos numéricos (Anos empregado, Rendimentos e etc)
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

teste_novo_cliente = pipeline_teste(teste_novo_cliente)

cliente_pred = teste_novo_cliente.drop(['Mau'], axis=1)

if st.button("Enviar"):
    model = joblib.load(r"modelo\xgb.joblib")
    final_pred = model.predict(cliente_pred)
    if final_pred[-1] == 0:
        st.success("### PARABÉNS!!!! CARTÃO APROVADO")
        st.balloons()
    else:
        st.error("### CAI FORA, OLHA O CALOTEEEEEE")
        st.snow()