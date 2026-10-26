import pandas as pd

df = pd.read_csv("Semana 03/Tarefa 02/train.csv").set_index("Id")

df_treino = df.sample(frac=0.8, random_state=42)

df_teste = df.drop(df_treino.index)

print(f"Tamanho treino: {df_treino.shape}")

print(f"Tamanho teste: {df_teste.shape}")

