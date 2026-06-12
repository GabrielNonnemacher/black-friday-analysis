import pandas as pd
import numpy as np
import time
import random

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

arquivo = "dataset_retail_black_friday_sales_50k.xlsx"
df = pd.read_excel(arquivo)

print("===== VISÃO INICIAL DO DATASET =====")
print(df.head())
print("\nDimensão do dataset:", df.shape)
print("\nTipos de dados:")
print(df.dtypes)

print("\n===== GASTO POR SEGMENTO DE CLIENTE =====")

resumo_segmentos = (
    df.groupby("segmento_cliente")["valor_compra"]
    .agg(
        quantidade_compras="count",
        gasto_medio="mean",
        gasto_mediano="median",
        gasto_total="sum"
    )
    .sort_values(by="gasto_medio", ascending=False)
)

print(resumo_segmentos)

p33 = df["valor_compra"].quantile(0.33)
p66 = df["valor_compra"].quantile(0.66)
max_valor = df["valor_compra"].max()

print(f"Cortes usados → Baixo: até {p33:.2f} | Médio: até {p66:.2f} | Alto: até {max_valor:.2f}")

df["faixa_gasto"] = pd.cut(
    df["valor_compra"],
    bins=[0, p33, p66, max_valor],
    labels=["Baixo", "Médio", "Alto"],
    include_lowest=True
)
target = "faixa_gasto"

print("\n===== DISTRIBUIÇÃO DA NOVA CLASSE =====")
print(df[target].value_counts())
print("\nProporção das classes:")
print(df[target].value_counts(normalize=True))

df["data_compra"] = pd.to_datetime(df["data_compra"], errors="coerce")

df["ano_compra"] = df["data_compra"].dt.year
df["mes_compra"] = df["data_compra"].dt.month
df["dia_compra"] = df["data_compra"].dt.day
df["dia_semana_compra"] = df["data_compra"].dt.dayofweek

colunas_remover = [
    "id_transacao",
    "id_cliente",
    "id_produto",
    "data_compra",
    "valor_compra",
    "preco_original",
    "pct_desconto",
    "preco_final",
    "quantidade"
]

X = df.drop(columns=colunas_remover + [target])
y = df[target]

print("\n===== FEATURES UTILIZADAS =====")
print(X.columns.tolist())

colunas_numericas = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
colunas_categoricas = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nColunas numéricas:")
print(colunas_numericas)

print("\nColunas categóricas:")
print(colunas_categoricas)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=random.random(),
    random_state=42,
    stratify=y
)

print("\n===== TAMANHOS =====")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

preprocessador_arvore = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), colunas_numericas),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), colunas_categoricas)
    ]
)

preprocessador_logistico = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), colunas_numericas),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), colunas_categoricas)
    ]
)

modelos = {
    "Árvore de Decisão": Pipeline([
        ("preprocessamento", preprocessador_arvore),
        ("modelo", DecisionTreeClassifier(
            criterion="entropy",
            random_state=42
        ))
    ]),
    "Random Forest": Pipeline([
        ("preprocessamento", preprocessador_arvore),
        ("modelo", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]),
    "Regressão Logística": Pipeline([
        ("preprocessamento", preprocessador_logistico),
        ("modelo", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])
}

resultados = []

for nome_modelo, pipeline in modelos.items():
    print(f"\n{'='*60}")
    print(f"MODELO: {nome_modelo}")
    print(f"{'='*60}")

    inicio = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fim = time.perf_counter()

    tempo_treinamento = fim - inicio

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    resultados.append({
        "Algoritmo": nome_modelo,
        "Acurácia": acc,
        "Precisão": prec,
        "Recall": rec,
        "F1-score": f1,
        "Tempo de treinamento (s)": tempo_treinamento
    })

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values(by="F1-score", ascending=False)

# intervalos entre as categorias BAIXA, MEDIA e ALTA
# Explicacao do metodo utilizado está no relatorio, 2.4
p33 = df["valor_compra"].quantile(0.33)
p66 = df["valor_compra"].quantile(0.66)
max_valor = df["valor_compra"].max()

limites = pd.DataFrame({
    "Faixa": ["Baixo", "Médio", "Alto"],
    "Limite inferior (R$)": [0, p33, p66],
    "Limite superior (R$)": [p33, p66, max_valor]
})

limites["Limite inferior (R$)"] = limites["Limite inferior (R$)"].map(lambda x: f"{x:,.2f}")
limites["Limite superior (R$)"] = limites["Limite superior (R$)"].map(lambda x: f"{x:,.2f}")

print("\n===== INTERVALOS DAS FAIXAS DE GASTO =====")
print(limites.to_string(index=False))

print("\n===== RESULTADOS FINAIS =====")
print(df_resultados)


