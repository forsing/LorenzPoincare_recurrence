# Potreban je NumPy ≥ 2.0


import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import json
import math
import numpy as np


# ============================================================
# CSV: najstarije izvlačenje prvo, najnovije poslednje.
# Svaki red mora sadržati sedam različitih brojeva od 1 do 39.
# ============================================================

CSV_FILES = [
    "/data/loto7_4696_k79.csv",
    "/data/loto7_4696_k79_loto_2970.csv",
    "/data/loto7_4696_k79_loto_plus_1726.csv",
]

WINDOWS = (1, 2, 3, 5, 8, 13, 21, 34)
NEIGHBOR_COUNTS = (32, 64, 128, 256)
BETAS = np.array([0.15, 0.35, 0.7])

MAX_HISTORY = max(WINDOWS)
TOTAL_COMBINATIONS = math.comb(39, 7)

BEAM_WIDTH = 16
SEARCH_STEPS = 12
BATCH_SIZE = 256

CONTROL_RUNS = 19
CONTROL_SEED = 7392026

# Broj kombinacija koje sa zadatom kombinacijom dele r brojeva.
INTERSECTION_COUNTS = np.array([
    math.comb(7, r) * math.comb(32, 7 - r)
    for r in range(8)
])

NORMALIZERS = (
    np.exp(BETAS[:, None] * np.arange(8))
    @ INTERSECTION_COUNTS
)

# Konfiguracija: (dužina istorije, najmanje suseda, beta).
#
# (0, 0, 0): ravnomerna raspodela.
# (0, 0, beta): raspodela iz istorije bez vremenskog konteksta.
# Ostale: povratci sličnim prethodnim stanjima.
#
# Ukupno: 1 + 3 + 8 * 4 * 3 = 100 konfiguracija.

CONFIGS = (
    [(0, 0, 0.0)]
    + [(0, 0, float(beta)) for beta in BETAS]
    + [
        (window, neighbors, float(beta))
        for window in WINDOWS
        for neighbors in NEIGHBOR_COUNTS
        for beta in BETAS
    ]
)


def load_csv(path):
    rows = []

    with open(path, encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row:
                continue

            try:
                values = tuple(sorted(int(value.strip()) for value in row))
            except ValueError as error:
                raise ValueError(
                    f"{path}, red {line_number}: "
                    "očekujem sedam celih brojeva, bez zaglavlja."
                ) from error

            if (
                len(values) != 7
                or len(set(values)) != 7
                or min(values) < 1
                or max(values) > 39
            ):
                raise ValueError(
                    f"{path}, red {line_number}: "
                    f"neispravna kombinacija {values}"
                )

            rows.append(values)

    if len(rows) < 300:
        raise ValueError(
            f"{path}: potrebno je najmanje 300 izvlačenja."
        )

    # Svaki bit čuva identitet jednog konkretnog broja.
    masks = np.array([
        sum(1 << (number - 1) for number in row)
        for row in rows
    ], dtype=np.uint64)

    return rows, masks


def intersection_matrix(masks):
    """Broj zajedničkih brojeva za svaki par izvučenih kombinacija."""
    count = len(masks)
    result = np.empty((count, count), dtype=np.uint8)

    for start in range(0, count, BATCH_SIZE):
        result[start:start + BATCH_SIZE] = np.bitwise_count(
            masks[start:start + BATCH_SIZE, None]
            & masks[None, :]
        )

    return result


def evaluate(matrix, start, stop):
    """
    Rekonstrukcija stanja:
        stanje pre t = niz prethodnih kombinacija.

    Povratci:
        pronalazimo slična ranija stanja.

    Procena:
        naredne kombinacije tih ranijih stanja određuju
        uslovnu distribuciju za trenutno stanje.

    Istorijski primer j koristi:
        kontekst: j-window ... j-1
        poznati nastavak: j

    Izostavljamo najbliža MAX_HISTORY prethodna ishoda kako
    se istorijski primeri ne bi preklapali sa trenutnim stanjem.
    """
    scores = np.zeros((stop - start, len(CONFIGS)))

    for output_row, t in enumerate(range(start, stop)):
        archive_end = t - MAX_HISTORY

        # Ishodi istorijskih primera:
        # MAX_HISTORY ... archive_end-1.
        intersections = matrix[t, MAX_HISTORY:archive_end]

        kernels = np.exp(
            BETAS[:, None] * intersections[None, :]
        )

        # Tri modela bez vremenskog konteksta.
        scores[output_row, 1:4] = np.log(
            kernels.mean(axis=1)
            / NORMALIZERS
            * TOTAL_COMBINATIONS
        )

        similarity = np.zeros(archive_end - MAX_HISTORY)
        column = 4

        for lag in range(1, MAX_HISTORY + 1):
            similarity += (
                matrix[
                    t - lag,
                    MAX_HISTORY - lag:archive_end - lag,
                ]
                / math.sqrt(lag)
            )

            if lag not in WINDOWS:
                continue

            # Zaokruživanje stabilizuje numeričko prepoznavanje
            # potpuno jednakih sličnosti.
            rounded = np.round(similarity, 10)
            order = np.argsort(-rounded, kind="stable")
            ranked = rounded[order]

            cumulative = np.cumsum(
                kernels[:, order],
                axis=1,
            )

            for requested_neighbors in NEIGHBOR_COUNTS:
                position = min(
                    requested_neighbors,
                    len(ranked),
                ) - 1

                threshold = ranked[position]

                # Uključujemo sve jednako slične susede na granici.
                actual_neighbors = int(np.searchsorted(
                    -ranked,
                    -threshold,
                    side="right",
                ))

                probabilities = (
                    cumulative[:, actual_neighbors - 1]
                    / actual_neighbors
                    / NORMALIZERS
                )

                scores[
                    output_row,
                    column:column + len(BETAS),
                ] = np.log(
                    probabilities * TOTAL_COMBINATIONS
                )

                column += len(BETAS)

    # Kolona 0 je uniformni model: log-dobitak je nula.
    return scores


def assess(matrix):
    """
    Prvih 60%: početna istorija.
    Sledećih 20%: izbor konfiguracije.
    Poslednjih 20%: test fiksirane konfiguracije.

    Svaka pojedinačna prognoza koristi samo prethodne podatke.
    """
    count = len(matrix)
    validation_start = int(count * 0.6)
    test_start = int(count * 0.8)

    scores = evaluate(
        matrix,
        validation_start,
        count,
    )

    validation_length = test_start - validation_start
    validation_means = scores[:validation_length].mean(axis=0)

    # Test NE učestvuje u izboru konfiguracije.
    selected_index = int(np.argmax(validation_means))

    test_mean = float(
        scores[validation_length:, selected_index].mean()
    )

    return (
        selected_index,
        float(validation_means[selected_index]),
        test_mean,
    )


def decode(mask):
    return [
        number + 1
        for number in range(39)
        if int(mask) & (1 << number)
    ]


def predict(masks, config):
    window, requested_neighbors, beta = config

    if beta == 0:
        # Uniformni model ne razlikuje kombinacije.
        # Ovo je deterministički predstavnik jednakih ocena,
        # a ne posebno favorizovana kombinacija.
        return (
            [1, 2, 3, 4, 5, 6, 7],
            1.0 / TOTAL_COMBINATIONS,
            0,
            0,
        )

    count = len(masks)
    archive_end = count - MAX_HISTORY

    similarity = np.zeros(archive_end - MAX_HISTORY)

    for lag in range(1, window + 1):
        similarity += (
            np.bitwise_count(
                masks[count - lag]
                & masks[MAX_HISTORY - lag:archive_end - lag]
            )
            / math.sqrt(lag)
        )

    similarity = np.round(similarity, 10)
    archive = masks[MAX_HISTORY:archive_end]

    if window:
        neighbors = min(requested_neighbors, len(similarity))
        position = len(similarity) - neighbors
        threshold = np.partition(similarity, position)[position]

        # Zadržavamo sve susede koji dele graničnu ocenu.
        targets = archive[similarity >= threshold]
    else:
        targets = archive

    kernel = np.exp(beta * np.arange(8))
    normalizer = float(kernel @ INTERSECTION_COUNTS)

    cache = {}

    def score_candidates(candidates):
        missing = sorted(
            set(map(int, candidates)) - cache.keys()
        )

        for start in range(0, len(missing), BATCH_SIZE):
            block = missing[start:start + BATCH_SIZE]
            candidate_masks = np.array(block, dtype=np.uint64)

            intersections = np.bitwise_count(
                candidate_masks[:, None]
                & targets[None, :]
            )

            probabilities = (
                kernel[intersections].mean(axis=1)
                / normalizer
            )

            cache.update(
                zip(block, map(float, probabilities))
            )

    def leaders():
        # Nema slučajnog biranja pri jednakim ocenama.
        return sorted(
            cache,
            key=lambda mask: (-cache[mask], mask),
        )[:BEAM_WIDTH]

    # Početni kandidati: sve istorijske kombinacije.
    score_candidates(masks)

    # Zatim generišemo nove kombinacije zamenom jednog broja.
    # Ovo je približna pretraga, ne iscrpno pretraživanje
    # svih 15.380.937 mogućih kombinacija.
    for _ in range(SEARCH_STEPS):
        previous_leaders = leaders()
        neighbors = set()

        for mask in previous_leaders:
            inside = [
                number
                for number in range(39)
                if mask & (1 << number)
            ]

            outside = [
                number
                for number in range(39)
                if not mask & (1 << number)
            ]

            for removed in inside:
                for added in outside:
                    neighbors.add(
                        mask
                        ^ (1 << removed)
                        ^ (1 << added)
                    )

        score_candidates(neighbors)

        if leaders() == previous_leaders:
            break

    best_mask = leaders()[0]
    candidate = decode(best_mask)

    assert len(candidate) == len(set(candidate)) == 7
    assert all(1 <= number <= 39 for number in candidate)

    return (
        candidate,
        cache[best_mask],
        len(cache),
        len(targets),
    )


def run(path):
    print(f"\nObrada: {path}", flush=True)

    rows, masks = load_csv(path)
    matrix = intersection_matrix(masks)

    selected_index, validation_gain, test_gain = assess(matrix)
    config = CONFIGS[selected_index]

    candidate, probability, examined, actual_neighbors = predict(
        masks,
        config,
    )

    # Kontrola redosleda.
    # Za svako mešanje ponavlja se CEO izbor konfiguracije,
    # čime se u kontrolu uključuje i samo traženje modela.
    rng = np.random.default_rng(CONTROL_SEED)
    control_results = []

    for repetition in range(CONTROL_RUNS):
        order = rng.permutation(len(rows))
        shuffled = matrix[np.ix_(order, order)]

        _, _, control_gain = assess(shuffled)
        control_results.append(control_gain)

        completed = repetition + 1
        if completed % 5 == 0 or completed == CONTROL_RUNS:
            print(
                f"Kontrola redosleda: {completed}/{CONTROL_RUNS}",
                flush=True,
            )

    # Jednostrana Monte Carlo procena.
    # Sa 19 kontrola rezolucija je 0,05.
    permutation_p = (
        1 + sum(value >= test_gain for value in control_results)
    ) / (CONTROL_RUNS + 1)

    window, requested_neighbors, beta = config

    if beta == 0:
        model_name = "uniformna_raspodela"
    elif window == 0:
        model_name = "istorijska_distribucija_bez_konteksta"
    else:
        model_name = "rekonstrukcija_stanja_i_povratci"

    result = {
        "CSV": path,
        "broj_izvlacenja": len(rows),
        "poslednja_kombinacija": list(rows[-1]),
        "NEXT": candidate,
        "izabrani_model": model_name,
        "uporedjenih_konfiguracija": len(CONFIGS),
        "duzina_stanja": window,
        "trazeni_minimum_suseda": requested_neighbors,
        "stvarni_broj_suseda_za_NEXT": actual_neighbors,
        "beta": beta,
        "validacija_log_dobitak": validation_gain,
        "test_log_dobitak": test_gain,
        "kontrole_broj": CONTROL_RUNS,
        "kontrole_prosecni_log_dobitak": float(
            np.mean(control_results)
        ),
        "kontrola_redosleda_p": permutation_p,
        "model_verovatnoca_NEXT": probability,
        "uniformna_verovatnoca": 1.0 / TOTAL_COMBINATIONS,
        "pojedinacno_ocenjenih_kandidata": examined,
        "napomena": (
            "NEXT je najviše ocenjen među pregledanim kandidatima. "
            "Model-verovatnoća nije potvrđena stvarna verovatnoća. "
            "Kontrola redosleda je preliminarna i ne dokazuje haos."
        ),
    }

    print(
        json.dumps(result, ensure_ascii=False, indent=2),
        flush=True,
    )


def main():
    if not hasattr(np, "bitwise_count"):
        raise RuntimeError(
            'Potreban je NumPy >= 2.0. Instaliraj pomoću: '
            'python3 -m pip install "numpy>=2.0"'
        )

    assert len(CONFIGS) == 100
    assert int(INTERSECTION_COUNTS.sum()) == TOTAL_COMBINATIONS

    for path in CSV_FILES:
        run(path)


if __name__ == "__main__":
    main()



"""
Obrada: /data/loto7_4696_k79.csv
Kontrola redosleda: 5/19
Kontrola redosleda: 10/19
Kontrola redosleda: 15/19
Kontrola redosleda: 19/19
{
  "CSV": "/data/loto7_4696_k79.csv",
  "broj_izvlacenja": 4696,
  "poslednja_kombinacija": [
    5,
    6,
    16,
    21,
    23,
    26,
    36
  ],
  "NEXT": [
    3,
    x,
    5,
    y,
    11,
    z,
    25
  ],
  "izabrani_model": "rekonstrukcija_stanja_i_povratci",
  "uporedjenih_konfiguracija": 100,
  "duzina_stanja": 3,
  "trazeni_minimum_suseda": 256,
  "stvarni_broj_suseda_za_NEXT": 271,
  "beta": 0.7,
  "validacija_log_dobitak": 0.0010381675569929576,
  "test_log_dobitak": 5.576210796518583e-06,
  "kontrole_broj": 19,
  "kontrole_prosecni_log_dobitak": -0.0013194181454380066,
  "kontrola_redosleda_p": 0.3,
  "model_verovatnoca_NEXT": 8.519895828925127e-08,
  "uniformna_verovatnoca": 6.501554489170588e-08,
  "pojedinacno_ocenjenih_kandidata": 9636,
  "napomena": "NEXT je najviše ocenjen među pregledanim kandidatima. Model-verovatnoća nije potvrđena stvarna verovatnoća. Kontrola redosleda je preliminarna i ne dokazuje haos."
}





Obrada: /data/loto7_4696_k79_loto_2970.csv
Kontrola redosleda: 5/19
Kontrola redosleda: 10/19
Kontrola redosleda: 15/19
Kontrola redosleda: 19/19
{
  "CSV": "/data/loto7_4696_k79_loto_2970.csv",
  "broj_izvlacenja": 2970,
  "poslednja_kombinacija": [
    4,
    10,
    14,
    15,
    28,
    33,
    34
  ],
  "NEXT": [
    6,
    x,
    15,
    y,
    24,
    z,
    34
  ],
  "izabrani_model": "rekonstrukcija_stanja_i_povratci",
  "uporedjenih_konfiguracija": 100,
  "duzina_stanja": 21,
  "trazeni_minimum_suseda": 64,
  "stvarni_broj_suseda_za_NEXT": 64,
  "beta": 0.7,
  "validacija_log_dobitak": 0.005370348896140394,
  "test_log_dobitak": -0.00493634062291206,
  "kontrole_broj": 19,
  "kontrole_prosecni_log_dobitak": -0.001691006497016588,
  "kontrola_redosleda_p": 0.9,
  "model_verovatnoca_NEXT": 1.277086218868953e-07,
  "uniformna_verovatnoca": 6.501554489170588e-08,
  "pojedinacno_ocenjenih_kandidata": 7320,
  "napomena": "NEXT je najviše ocenjen među pregledanim kandidatima. Model-verovatnoća nije potvrđena stvarna verovatnoća. Kontrola redosleda je preliminarna i ne dokazuje haos."
}





Obrada: /data/loto7_4696_k79_loto_plus_1726.csv
Kontrola redosleda: 5/19
Kontrola redosleda: 10/19
Kontrola redosleda: 15/19
Kontrola redosleda: 19/19
{
  "CSV": "/data/loto7_4696_k79_loto_plus_1726.csv",
  "broj_izvlacenja": 1726,
  "poslednja_kombinacija": [
    5,
    6,
    16,
    21,
    23,
    26,
    36
  ],
  "NEXT": [
    2,
    x,
    12,
    y,
    20,
    z,
    29
  ],
  "izabrani_model": "rekonstrukcija_stanja_i_povratci",
  "uporedjenih_konfiguracija": 100,
  "duzina_stanja": 2,
  "trazeni_minimum_suseda": 64,
  "stvarni_broj_suseda_za_NEXT": 91,
  "beta": 0.7,
  "validacija_log_dobitak": 0.0020759535242438924,
  "test_log_dobitak": 0.003915063232421032,
  "kontrole_broj": 19,
  "kontrole_prosecni_log_dobitak": -0.0011007503397320404,
  "kontrola_redosleda_p": 0.1,
  "model_verovatnoca_NEXT": 1.0276050066651803e-07,
  "uniformna_verovatnoca": 6.501554489170588e-08,
  "pojedinacno_ocenjenih_kandidata": 5450,
  "napomena": "NEXT je najviše ocenjen među pregledanim kandidatima. Model-verovatnoća nije potvrđena stvarna verovatnoća. Kontrola redosleda je preliminarna i ne dokazuje haos."
}
"""



"""
Lorenca i Poenkarea u teoriji haosa, njihove metode mogu služiti za ispitivanje dinamike loto niza:
- Lorenc: istraživanje da li iza nepravilnog niza postoji deterministička dinamika osetljiva na početne uslove.
- Poenkare: istraživanje povrataka sistema u slična stanja i ponašanja nakon tih povrataka.


Ključna su tri pitanja:
- Da li istorija izvlačenja sadrži merljivu nelinearnu zavisnost? Istraživati na CSV-u.
- Da li je ta zavisnost posledica determinističkog haosa? Sam pronađen obrazac to ne dokazuje.
- Da li omogućava predviđanje sledeće kombinacije? To mora posebno da pokaže provera na kasnijim izvlačenjima.


Lorencovog atraktora.
Niz koji izgleda nepravilan možda ima skrivenu dinamiku, pa slična prethodna stanja mogu imati sličan nastavak.
Na CSV-u istražiti:
1. Niz od nekoliko uzastopnih kombinacija predstavlja jedno posmatrano stanje.
2. Pronađem njegova najsličnija stanja u ranijoj istoriji.
3. Ispitam da li su njihovi nastavci međusobno sličniji nego kod nasumično izabranih stanja.
4. Proverim da li taj odnos opstaje na kasnijim izvlačenjima.
Vrednost ideje je u rekonstrukciji stanja i proučavanju njihovih nastavaka. 


Lorenz + Poincaré: skrivena stanja, povratci i njihovi nastavci - traži ponovljive putanje
Lorenz — deterministički haos	Rekonstrukcija skrivenog stanja iz niza posmatranja.	Da li slični nizovi prethodnih kombinacija imaju slične nastavke i koliko brzo ta sličnost nestaje.
Poincaré — povratci	Sistem se vraća u blizinu ranije posećenih stanja.	Distribuciju vremena između sličnih stanja i da li povratak nosi informaciju o narednom izvlačenju.


Kombinacija Lorenz-Poincaré pristupa sa učenjem evolucije distribucije. 
To je najkoherentnija istraživačka celina.
Povezani ovako:
1. Rekonstrukcija stanja — Lorenzova inspiracija. Iz prethodnih izvlačenja oblikujem opis trenutnog stanja. Koliko istorije treba koristiti određujem proverom.
2. Povratci — Poincaréova inspiracija. Tražim slična ranija stanja i posmatram njihove nastavke. Proveravam da li bliža stanja zaista imaju sličnije nastavke.
3. Evolucija distribucije. Iz tih istorijskih nastavaka procenjujem uslovnu distribuciju sledeće kombinacije i prema njoj ocenjujem kandidate.
Prednost povezivanja je što svaki deo ima jasnu ulogu: prvi definiše stanje, drugi pronalazi relevantnu istoriju, treći daje predikciju.


Kompletna V1 zasnovanu na rekonstrukciji stanja, povratcima sličnim stanjima i proceni naredne distribucije. 
Uporedjena podešavanja i provereni rezultati odvojeno za sva tri CSV-a.
Provere koda su prošle: prognoza ne koristi buduće redove, a ponovljeno pokretanje daje isti rezultat. 
U poređenju 100 konfiguracija izabrani su konteksti od 3, 21 i 2 izvlačenja za tri CSV-a. 
Kontrola sa izmešanim redosledom podataka; ona proverava da li je vremenski redosled zaista koristan.

Provera hronološki test i 19 kontrola sa izmešanim redosledom. 
Izbor NEXT je deterministički. 
Provere nisu potvrdile pouzdanu prediktivnu prednost; pretraga kombinacija je približna.



Potreban je NumPy ≥ 2.0
"""
