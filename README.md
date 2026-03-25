# Sistema di Triage Ticket Aziendali (NLP)

Prototipo di Machine Learning (NLP) per automatizzare lo smistamento dei ticket. Il sistema analizza il testo per prevedere il reparto di competenza e la priorità.

## Installazione

Assicurarsi di avere Python 3. Aprire il terminale e installare le dipendenze con questo comando:

pip install pandas scikit-learn matplotlib seaborn streamlit joblib numpy

## Guida all'Utilizzo

Eseguire gli script da terminale rigorosamente in questo ordine:

### Fase 1: Creazione Dataset
Comando: python 1_crea_dataset.py

Azione: Genera "dataset_ticket.csv" (420 ticket) assegnando le priorità tramite parole chiave.

### Fase 2: Addestramento Modello
Comando: python 2_addestra_modello.py

Azione: Preprocessa i dati e addestra il modello, esportando i risultati in "modelli/" e le metriche in "grafici/".

### Fase 3: Avvio Dashboard
Comando: streamlit run 3_dashboard.py

Azione: Avvia l'interfaccia web per la classificazione dei ticket singoli o l'elaborazione di ticket multipli in file CSV.