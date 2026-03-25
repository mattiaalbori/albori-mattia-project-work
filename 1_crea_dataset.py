import pandas as pd
import random
from typing import List, Dict, Any

def inserisci_typo(testo: str, probabilita: float = 0.11) -> str:
    parole = testo.split()
    for i in range(len(parole)):
        if random.random() < probabilita and len(parole[i]) > 4:
            p = list(parole[i])
            idx = random.randint(0, len(p) - 2)
            p[idx], p[idx + 1] = p[idx + 1], p[idx]
            parole[i] = "".join(p)
    return " ".join(parole)

def genera_dataset(seed_value: int = 42, ticket_per_categoria: int = 140) -> List[Dict[str, Any]]:
    random.seed(seed_value)
    
    ticket_templates = {
        "Tecnico": [
            ("PC bloccato", "Il computer non si accende, la situazione è bloccante per il mio lavoro."),
            ("Errore di sistema", "Il programma restituisce un errore durante il salvataggio."),
            ("Server offline", "Il server principale ha subito un guasto hardware, siamo offline."),
            ("Anomalia stampante", "La stampante laser ha un'anomalia nella stampa dei colori."),
            ("Urgenza monitor", "Lo schermo è nero, mi serve un intervento urgente."),
            ("Pezzo difettoso", "Il mouse appena consegnato sembra difettoso."),
            ("PC bloccati", "I computer dell'ufficio non si accendono, siamo tutti fermi."),
            ("Errori di sistema", "I programmi restituiscono continui errori, non riusciamo a lavorare."),
            ("Server guasti", "I server principali sono caduti, l'intera rete aziendale è giù."),
            ("Stampanti rotte", "Le stampanti del secondo piano sono rotte, i toner sono difettosi."),
            ("Monitor spenti", "Gli schermi delle nuove postazioni rimangono neri."),
            ("Guasto", "Rete giù. Fate presto."),
            ("Virus", "Malware rilevato. PC inutilizzabile."),
            ("Password", "Password scaduta. Reset urgente."),
            ("Lentezza navigazione", "Buongiorno, vi scrivo per segnalare che da questa mattina si registra un rallentamento della rete."),
            ("Software in crash", "Salve, volevo segnalare che il gestionale crasha in continuazione.")
        ],
        "Amministrazione": [
            ("Coordinate IBAN", "Mandatemi l'IBAN per il nuovo cliente americano."),
            ("Fattura da controllare", "C'è un problema con la fattura del fornitore Rossi."),
            ("Stipendio bloccato", "Il bonifico del mio stipendio è bloccato in banca."),
            ("Errore IVA", "L'importo calcolato per l'imposta contiene un errore."),
            ("Calcolo sbagliato", "Ho notato che la trattenuta in busta paga è sbagliata."),
            ("Codici IBAN", "Mandatemi i codici bancari e gli IBAN per i nuovi clienti."),
            ("Fatture errate", "Ci sono problemi con le fatture inviate dai fornitori questo mese."),
            ("Stipendi bloccati", "I bonifici degli stipendi sono stati respinti, situazione grave."),
            ("Errori IVA", "Gli importi calcolati per le imposte contengono numerosi errori."),
            ("Rimborsi ritardo", "Ci sono ritardi nei rimborsi delle trasferte di Roma e Milano."),
            ("F24", "Scadenza F24 oggi. Urgente!"),
            ("Spese", "Allego note spese."),
            ("Pagamento", "Pagamento fornitore respinto."),
            ("Guasto portale", "Buongiorno, vi comunico che il portale pagamenti ha un guasto."),
            ("Documentazione", "Salve, avrei necessità che mi mandaste le certificazioni uniche.")
        ],
        "Commerciale": [
            ("Preventivo sbagliato", "Abbiamo inviato un'offerta con un prezzo sbagliato, va corretta."),
            ("Ordine bloccato", "La spedizione per il cliente VIP è ferma, situazione critica."),
            ("Contratto in scadenza", "Il rinnovo è scaduto ieri, rischiamo di perdere il cliente, urgente!"),
            ("Anomalia listino", "Ho riscontrato un'anomalia nel prezzo caricato a sistema."),
            ("Errore contratto", "C'è un errore di battitura nella clausola 4 del contratto."),
            ("Preventivi sbagliati", "Abbiamo inviato delle offerte con i prezzi sbagliati, vanno corrette."),
            ("Ordini bloccati", "Le spedizioni per i clienti VIP sono ferme in magazzino."),
            ("Contratti scaduti", "I rinnovi dei contratti sono scaduti, dobbiamo ricontattarli."),
            ("Nuovi lead", "Abbiamo ricevuto ottimi contatti e nuovi lead dall'evento di ieri."),
            ("Invio cataloghi", "Ricordatevi di mandare i pdf con i nuovi cataloghi ai clienti storici."),
            ("Call", "Domani alle 10 call con prospect."),
            ("Sconto", "Applicare sconto 10% subito."),
            ("Ritardo", "Forte ritardo consegna. Cliente furioso."),
            ("Problema legale", "Vi scrivo urgentemente perché il cliente ha minacciato vie legali."),
            ("Strategia fiera", "In seguito all'evento di ieri, vorrei fissare un incontro per discutere la strategia.")
        ]
    }

    parole_chiave_alta = ["blocc", "urgent", "critic", "grav", "immediat", "scadut", "scadenz", "guast", "crash", "offline", "virus", "malware"]
    parole_chiave_media = ["error", "fattur", "sbagliat", "anomal", "ritard", "difett", "rallentament"]

    lista_ticket = []

    for cat, templates in ticket_templates.items():
        for _ in range(ticket_per_categoria):
            titolo_orig, corpo_orig = random.choice(templates)
            testo_completo = f"{titolo_orig} {corpo_orig}".lower()
            
            priorita = "Bassa"
            if any(radice in testo_completo for radice in parole_chiave_alta): 
                priorita = "Alta"
            elif any(radice in testo_completo for radice in parole_chiave_media): 
                priorita = "Media"
            
            titolo = inserisci_typo(titolo_orig)
            corpo = inserisci_typo(corpo_orig)
            
            cat_assegnata = random.choice(list(ticket_templates.keys())) if random.random() < 0.09 else cat
            prio_assegnata = random.choice(["Bassa", "Media", "Alta"]) if random.random() < 0.09 else priorita
                
            lista_ticket.append({
                "title": titolo, 
                "body": corpo, 
                "category": cat_assegnata, 
                "priority": prio_assegnata
            })

    random.shuffle(lista_ticket)
    for i, ticket in enumerate(lista_ticket, start=1):
        ticket["id"] = i
    return lista_ticket

def main() -> None:
    dataset = genera_dataset()
    df = pd.DataFrame(dataset)[['id', 'title', 'body', 'category', 'priority']]
    df.to_csv('dataset_ticket.csv', index=False, encoding='utf-8')
    print("Dataset generato e salvato in 'dataset_ticket.csv'.")

if __name__ == "__main__":
    main()