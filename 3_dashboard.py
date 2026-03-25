import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
from typing import List

def pulisci_testo(testo: str) -> str:
    testo_min = str(testo).lower()
    testo_pulito = re.sub(r'[^\w\s]', ' ', testo_min)
    return " ".join(testo_pulito.split())

def estrai_parole_chiave(pipeline, testo_pulito: str, classe_predetta: str) -> List[str]:
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    indice_classe = list(clf.classes_).index(classe_predetta)
    vettore_tfidf = tfidf.transform([testo_pulito]).toarray()[0]
    nomi_feature = tfidf.get_feature_names_out()
    coefficienti = clf.coef_[indice_classe]
    punteggi_parole = vettore_tfidf * coefficienti
    indici_non_zero = np.where(vettore_tfidf > 0)[0]
    
    if len(indici_non_zero) == 0:
        return ["Nessuna parola rilevante"]
        
    indici_ordinati = indici_non_zero[np.argsort(punteggi_parole[indici_non_zero])[::-1]]
    return [nomi_feature[i] for i in indici_ordinati[:5]]

def elabora_batch(df: pd.DataFrame, modello_cat, modello_prio) -> pd.DataFrame:
    if 'Testo' in df.columns:
        testi_grezzi = df['Testo'].fillna('')
    elif 'title' in df.columns:
        testi_grezzi = df['title'].fillna('')
    else:
        col_testo = df.columns[-1]
        testi_grezzi = df[col_testo].fillna('')
        
    testi_puliti = testi_grezzi.apply(pulisci_testo)
    df['Categoria_Prevista'] = modello_cat.predict(testi_puliti)
    df['Priorita_Suggerita'] = modello_prio.predict(testi_puliti)
    return df

def main() -> None:
    st.set_page_config(page_title="Ticket Aziendali", layout="centered", page_icon="🏢")
    st.markdown("<h2 style='text-align: center;'>🏢 Smistamento Ticket Aziendali</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Sistema intelligente per la classificazione e priorità dei ticket.</p>", unsafe_allow_html=True)

    try:
        modello_cat = joblib.load('modelli/modello_categoria.joblib')
        modello_prio = joblib.load('modelli/modello_priorita.joblib')
    except FileNotFoundError:
        st.error("⚠️ File dei modelli non trovati nella cartella 'modelli/'. Eseguire prima 2_addestra_modello.py")
        return

    st.markdown("### 📝 Inserimento Manuale Ticket")
    with st.form("analisi_form", clear_on_submit=False):
        testo_ticket = st.text_input("Descrizione del problema:", placeholder="Scrivi qui e premi Invio...")
        submit = st.form_submit_button("Analizza ticket", type="primary", use_container_width=True)

    if submit:
        if not testo_ticket.strip():
            st.warning("Inserisci il testo del ticket.")
        else:
            testo_proc = pulisci_testo(testo_ticket)
            pred_cat = modello_cat.predict([testo_proc])[0]
            pred_prio = modello_prio.predict([testo_proc])[0]
            parole = estrai_parole_chiave(modello_cat, testo_proc, pred_cat)
            
            st.success("✅ Analisi completata!")
            c1, c2 = st.columns(2)
            c1.metric("🏷️ Categoria", pred_cat)
            c2.metric("⚡ Priorità", pred_prio)
            st.info(f"🔑 **Parole chiave influenti:** {', '.join(parole).title()}")

    st.divider()
    st.markdown("### 📂 Smistamento Multiplo da File (CSV)")
    file_caricato = st.file_uploader("Carica file CSV", type=['csv'])
    
    if file_caricato:
        try:
            contenuto = file_caricato.getvalue().decode("utf-8").splitlines()
            dati_estratti = []
            
            for riga in contenuto:
                if not riga.strip():
                    continue
                if ';' in riga:
                    parti = riga.split(';', 1)
                elif ',' in riga:
                    parti = riga.split(',', 1)
                else:
                    parti = ["", riga]
                    
                testo = parti[1].strip() if len(parti) > 1 else parti[0].strip()
                if testo.startswith('"') and testo.endswith('"'):
                    testo = testo[1:-1]
                
                dati_estratti.append({"Testo": testo})
                
            df_input = pd.DataFrame(dati_estratti)
            
            if not df_input.empty and df_input['Testo'].iloc[0].lower() in ['testo', 'title', 'descrizione']:
                df_input = df_input.iloc[1:].reset_index(drop=True)
                
            df_risultato = elabora_batch(df_input, modello_cat, modello_prio)
            csv = df_risultato.to_csv(index=False).encode('utf-8')
            
            st.success("🎉 Analisi batch completata!")
            st.download_button("📥 Scarica CSV Analizzato", csv, "ticket_risolti.csv", "text/csv", type="primary", use_container_width=True)
            
        except Exception:
            st.error("❌ Impossibile leggere il file. Assicurati che sia un file di testo valido.")

if __name__ == "__main__":
    main()