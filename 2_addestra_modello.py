import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

STOP_WORDS_IT = [
    'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 
    'che', 'mi', 'ti', 'ci', 'vi', 'si', 'ne', 'non', 'sono', 'è', 'siamo', 'siete', 'hanno', 'abbiamo', 'avete', 
    'ho', 'hai', 'ha', 'questo', 'questa', 'quello', 'quella', 'qui', 'lì', 'me', 'tu', 'te', 'lui', 'lei', 'noi', 
    'voi', 'loro', 'come', 'cosa', 'dove', 'quando', 'perchè', 'perché', 'quale', 'quanto', 'al', 'allo', 'alla', 
    'ai', 'agli', 'alle', 'del', 'dello', 'della', 'dei', 'degli', 'delle', 'dal', 'dallo', 'dalla', 'dai', 'dagli', 
    'dalle', 'nel', 'nello', 'nella', 'nei', 'negli', 'nelle', 'sul', 'sullo', 'sulla', 'sui', 'sugli', 'sulle'
]

def pulisci_testo(testo: str) -> str:
    testo_min = str(testo).lower()
    testo_pulito = re.sub(r'[^\w\s]', ' ', testo_min)
    return " ".join(testo_pulito.split())

def prepara_dati(filepath: str) -> Tuple:
    df = pd.read_csv(filepath)
    df['testo_processato'] = (df['title'].fillna('') + " " + df['body'].fillna('')).apply(pulisci_testo)
    return train_test_split(
        df['testo_processato'], df['category'], df['priority'], 
        test_size=0.20, random_state=42
    )

def crea_pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=STOP_WORDS_IT)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

def salva_matrice_confusione(y_true: pd.Series, y_pred: list, labels: List[str], titolo: str, filename: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=labels, yticklabels=labels, linewidths=0.5)
    plt.title(titolo)
    plt.ylabel('Classe Reale')
    plt.xlabel('Classe Predetta')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def salva_f1_plot(y_true: pd.Series, y_pred: list, labels: List[str], titolo: str, filename: str) -> None:
    scores = f1_score(y_true, y_pred, labels=labels, average=None)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=scores, hue=labels, palette='Set2', legend=False)
    plt.title(titolo)
    plt.ylabel('Punteggio F1')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main() -> None:
    os.makedirs('grafici', exist_ok=True)
    os.makedirs('modelli', exist_ok=True)
    
    X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = prepara_dati('dataset_ticket.csv')

    pipeline_categoria = crea_pipeline()
    pipeline_priorita = crea_pipeline()

    pipeline_categoria.fit(X_train, y_cat_train)
    pipeline_priorita.fit(X_train, y_prio_train)

    pred_cat = pipeline_categoria.predict(X_test)
    pred_prio = pipeline_priorita.predict(X_test)

    acc_cat = accuracy_score(y_cat_test, pred_cat)
    f1_macro_cat = f1_score(y_cat_test, pred_cat, average='macro')

    print("\n" + "="*30)
    print(" VALUTAZIONE MODELLO ")
    print("-" * 30)
    print(f"Accuracy: {acc_cat:.4f}")
    print(f"F1-Score: {f1_macro_cat:.4f}")
    print("="*30 + "\n")

    salva_matrice_confusione(y_cat_test, pred_cat, list(pipeline_categoria.classes_), 
                             'Matrice di Confusione - Classi', 'grafici/matrice_confusione_classi.png')
    
    salva_f1_plot(y_cat_test, pred_cat, list(pipeline_categoria.classes_), 
                  'Affidabilità Predizioni per Categoria', 'grafici/affidabilita_predizioni_categoria.png')
    
    joblib.dump(pipeline_categoria, 'modelli/modello_categoria.joblib')
    joblib.dump(pipeline_priorita, 'modelli/modello_priorita.joblib')

if __name__ == "__main__":
    main()