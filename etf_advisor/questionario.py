"""
questionario.py
=============================================================================
Logica del Questionario Comportamentale (profilazione del rischio).

Il questionario è volutamente formulato con *esempi di vita reale* e
linguaggio non tecnico, per evitare la "tecnocrazia" (nessun "Value at Risk").

Ogni risposta ha un punteggio da 1 (prudente) a 3 (audace).
Il punteggio totale (4..12) viene mappato su un PROFILO DI INVESTIMENTO
che definisce l'Asset Allocation Target (peso azionario/obbligazionario).
"""

from __future__ import annotations
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Definizione delle 4 domande (una per sezione di profilazione)
# ---------------------------------------------------------------------------
QUESTIONS = [
    {
        "id": "loss",
        "section": "Tolleranza alle Perdite (psicologia del rischio)",
        "question": (
            "Hai investito 10.000 €. Dopo 6 mesi il tuo conto segna "
            "7.000 € (-30%) per il crollo dei mercati. Come ti comporti?"
        ),
        "options": [
            {"label": "Vendo tutto subito per non perdere altro", "score": 1},
            {"label": "Mantengo la posizione e aspetto senza toccare nulla", "score": 2},
            {"label": "Investo altri 2.000 € per riacquistare a sconto", "score": 3},
        ],
    },
    {
        "id": "horizon",
        "section": "Orizzonte Temporale (durata)",
        "question": "Tra quanto tempo ti serviranno realisticamente questi soldi?",
        "options": [
            {"label": "Entro 1-3 anni (es. auto, spese imminenti)", "score": 1},
            {"label": "Tra 3 e 7 anni (es. anticipo casa, progetti medi)", "score": 2},
            {"label": "Oltre 10-15 anni (es. pensione, indipendenza)", "score": 3},
        ],
    },
    {
        "id": "knowledge",
        "section": "Conoscenza del Mercato e Capacità finanziaria",
        "question": "Quali di questi strumenti hai già posseduto o conosci bene?",
        "options": [
            {"label": "Solo conto corrente e BTP / Conti Deposito", "score": 1},
            {"label": "Fondi della banca, azioni singole o ETF basilari", "score": 2},
            {"label": "Derivati, criptovalute, leva, portafogli complessi", "score": 3},
        ],
    },
    {
        "id": "saving",
        "section": "Capacità di Risparmio e Flusso di Cassa",
        "question": "Questo investimento rappresenta:",
        "options": [
            {"label": "Quasi la totalità dei miei risparmi", "score": 1},
            {"label": "Quota significativa, ma ho un fondo di emergenza", "score": 2},
            {"label": "Capitale libero da cui non dipendo per vivere", "score": 3},
        ],
    },
]


@dataclass
class Profile:
    """Profilo di investimento risultante dalla profilazione."""
    key: str            # 'conservativo' | 'bilanciato' | 'crescita' | 'aggressivo'
    name: str           # Nome leggibile
    description: str    # Spiegazione in linguaggio semplice
    equity_target: float   # % azionaria target (0..1)
    bond_target: float      # % obbligazionaria target (0..1)


# Mappatura punteggio -> profilo.
# Punteggio minimo = 4 (tutto "1"), massimo = 12 (tutto "3").
def map_profile(total_score: int) -> Profile:
    if total_score <= 6:
        return Profile(
            key="conservativo",
            name="Conservativo",
            description=(
                "Priorità alla protezione del capitale. Accetti rendimenti "
                "modesti pur di dormire sonni tranquilli anche in caso di "
                "scossoni dei mercati."
            ),
            equity_target=0.20,
            bond_target=0.80,
        )
    elif total_score <= 9:
        return Profile(
            key="bilanciato",
            name="Bilanciato",
            description=(
                "Cerchi un equilibrio tra crescita e sicurezza. Sopporti "
                "fluctuazioni moderate del portafoglio nel medio periodo."
            ),
            equity_target=0.50,
            bond_target=0.50,
        )
    elif total_score <= 11:
        return Profile(
            key="crescita",
            name="Crescita",
            description=(
                "Obiettivo principale la crescita del capitale nel lungo "
                "periodo. Accetti oscillazioni anche marcate dei prezzi."
            ),
            equity_target=0.75,
            bond_target=0.25,
        )
    else:
        return Profile(
            key="aggressivo",
            name="Aggressivo",
            description=(
                "Massima propensione al rischio per massimizzare la crescita. "
                "Le fluttuazioni violente non ti spaventano e hai orizzonte "
                "molto lungo."
            ),
            equity_target=0.90,
            bond_target=0.10,
        )


def compute_score(answers: dict) -> int:
    """Somma i punteggi delle risposte selezionate.

    `answers` ha forma {question_id: score}. Risposte mancanti valgono 0.
    """
    return sum(int(v) for v in answers.values() if v is not None)


def run_questionnaire(answers: dict) -> Profile:
    """Calcola il profilo a partire dalle risposte del questionario."""
    score = compute_score(answers)
    return map_profile(score)
