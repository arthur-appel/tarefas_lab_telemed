import pandas as pd
import numpy as np
'''Dúvidas que surgiram: 
1 - qual a diferença de reatribuir o df.drop a variavel df de novo e não fazer isso? (respondida)
2 - pra filtrar colunas não numéricas (categóricas), usar include string, category e object ou só exclude np.numbers?
3 - a coluna GarageYrBlt, quando tem NA é porque nao tem garagem, nesse caso devo dar fill com a média dos anos ou não? (escolhi preencher com zeros as sem garagem).
4 - as variaveis ordinais que foram convertidas pra 1 a 5, devo transformar no z-score ou não? optei por não fazer.
'''

df = pd.read_csv("Semana 02/Tarefa 01/train.csv")
df = df.set_index("Id") #não dar fill com média, é identificador.



cols_indesejadas=['Utilities','Street','Condition2','RoofMatl','Heating','PoolQC','Exterior1st','SaleType']
df = df.drop(columns=cols_indesejadas)
df["MSSubClass"] = df["MSSubClass"].astype(str) #é numérico, mas na verdade é categórica (tipos de construção).


if not any(col in df.columns for col in cols_indesejadas): #checagem remoções.
    print("Colunas removidas com sucesso.")

df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0) #tratando separadamente, se for NA é pq nao tem garagem.

colunas_anos = ["YearBuilt", "YearRemodAdd", "YrSold"] #vou precisar arredondar a média depois, entao separei das outras numéricas.
colunas_para_ignorar = colunas_anos + ["GarageYrBlt"]
colunas_numericas = df.select_dtypes(include=[np.number]).drop(columns=colunas_para_ignorar).columns #todas numéricas, menos as de anos
colunas_categoricas = df.select_dtypes(include=['object','category','string']).columns
colunas_ordinais= ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond'] #n consegui automatizar identificaçao de cols ordinais, analisei o data_description no olho.
cols_zscore = [col for col in colunas_numericas if col != "SalePrice"] #saleprice vai ter tratamento diferente.

qualidades={"None":0, "Vazio":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5} 
medias_anos = df[colunas_anos].mean().round()

df[colunas_anos]=df[colunas_anos].fillna(medias_anos) #só os anos arredondam
df[colunas_numericas]= df[colunas_numericas].fillna(df[colunas_numericas].mean()) #nao tem .round()
df[colunas_categoricas]=df[colunas_categoricas].fillna('Vazio')
df[cols_zscore] = ( df[cols_zscore] - df[cols_zscore].mean()) / df[cols_zscore].std() #transformar em z score
for col in colunas_ordinais:
    df[col] = df[col].replace(qualidades).astype(int)


colunas_categoricas = df.select_dtypes(include=["object", "category", "string"]).columns #atualizando as categóricas
df = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True, dtype=int) #one hot encoding
log_sp = np.log(df["SalePrice"])
df["SalePrice"] = (log_sp - log_sp.mean()) / log_sp.std()

print("Total de valores nulos no dataset:", df.isnull().sum().sum()) #checagem de nulls


