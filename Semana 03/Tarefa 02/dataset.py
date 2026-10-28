import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

df = pd.read_csv("Semana 03/Tarefa 02/train.csv").set_index("Id")

cols_indesejadas=['Utilities','Street','Condition2','RoofMatl','Heating','PoolQC','Exterior1st','SaleType']
df = df.drop(columns=cols_indesejadas)

df["MSSubClass"] = df["MSSubClass"].astype(str) #é numérico, mas na verdade é categórica (tipos de construção).
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0) #tratando separadamente, se for NA é pq nao tem garagem.

colunas_anos = ["YearBuilt", "YearRemodAdd", "YrSold"] #vou precisar arredondar a média depois, entao separei das outras numéricas.
colunas_para_ignorar = colunas_anos + ["GarageYrBlt"]
colunas_numericas = df.select_dtypes(include=[np.number]).drop(columns=colunas_para_ignorar).columns #todas numéricas, menos as de anos
colunas_categoricas = df.select_dtypes(include=['object','category','string']).columns
colunas_ordinais= ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond'] 
cols_zscore = [col for col in colunas_numericas if col != "SalePrice"] #saleprice vai ter tratamento diferente.

qualidades={"None":0, "Vazio":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5} 
for col in colunas_ordinais:
    df[col] = df[col].fillna("None").replace(qualidades).astype(int)

#divisão do df em treino e teste

df_treino, df_teste = train_test_split(df, test_size=0.20, random_state=42)

#calculo das estatisticas (com treino)

medias_anos = df_treino[colunas_anos].mean().round()
medias_numericas = df_treino[colunas_numericas].mean()

media_z = df_treino[cols_zscore].mean()
std_z = df_treino[cols_zscore].std()

#estatísticas do sale price

log_sp_treino = np.log(df_treino["SalePrice"])
media_log_sp= log_sp_treino.mean()
std_log_sp = log_sp_treino.std()

#aplicação das estatísticas em ambos os dfs 

df_treino[colunas_anos] = df_treino[colunas_anos].fillna(medias_anos)
df_teste[colunas_anos] = df_teste[colunas_anos].fillna(medias_anos)

df_treino[colunas_numericas] = df_treino[colunas_numericas].fillna(medias_numericas)
df_teste[colunas_numericas] = df_teste[colunas_numericas].fillna(medias_numericas)

df_treino[colunas_categoricas] = df_treino[colunas_categoricas].fillna('Vazio')
df_teste[colunas_categoricas] = df_teste[colunas_categoricas].fillna('Vazio')

# normalização Z-Score

df_treino[cols_zscore] = (df_treino[cols_zscore] - media_z) / std_z
df_teste[cols_zscore] = (df_teste[cols_zscore] - media_z) / std_z

# normalização do SalePrice
df_treino["SalePrice"] = (np.log(df_treino["SalePrice"]) - media_log_sp) / std_log_sp
df_teste["SalePrice"] = (np.log(df_teste["SalePrice"]) - media_log_sp) / std_log_sp

# aplicando get_dummies

df_treino = pd.get_dummies(df_treino, columns=colunas_categoricas, drop_first=True, dtype=int)
df_teste = pd.get_dummies(df_teste, columns=colunas_categoricas, drop_first=True, dtype=int)

df_teste = df_teste.reindex(columns=df_treino.columns, fill_value=0)


# variáveis 

X_treino_df = df_treino.drop(columns=["SalePrice"])
y_treino_df = df_treino[["SalePrice"]]

X_teste_df = df_teste.drop(columns=["SalePrice"])
y_teste_df = df_teste[["SalePrice"]]

# criando os tensores

X_train = torch.tensor(X_treino_df.values, dtype=torch.float32)
y_train = torch.tensor(y_treino_df.values, dtype=torch.float32)

X_teste = torch.tensor(X_teste_df.values, dtype=torch.float32)
y_teste = torch.tensor(y_teste_df.values, dtype=torch.float32)