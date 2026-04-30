import math
import random
import matplotlib.pyplot as plt


def exponential(rate):
    if rate <= 0:
        raise ValueError("rate must be strictly greater than 0")
    u = random.random()
    return -math.log(1 - u) / rate


def simulate(lam, mu, end_time, verbose=True):
    if lam <= 0 or mu <= 0:
        raise ValueError("lam and mu must be strictly greater than 0")
    if end_time <= 0:
        raise ValueError("end_time must be strictly greater than 0")

    # Initialisation des parametres de la simulation
    clock = 0.0  
    queue = []
    next_arrival = exponential(lam)
    next_departure = float("inf")

    T, Tq, Ts = 0.0, 0.0, 0.0
    Nq, N = 0.0, 0.0
    SU = 0.0
    nb_arr, nb_dep = 0, 0

    # Variables supplementaires pour le suivi/calcul metriques
    sumN, nbN, upN = 0.0, 0, 0.0
    sumQ, nbQ, upQ = 0.0, 0, 0.0
    sumS, upS = 0.0, 0.0
    sumT, sumTq, sumTs = 0.0, 0.0, 0.0

    busy = False
    current_service_start = None
    current_customer_arrival = None

    # Boucle de simulation
    while queue or next_arrival <= end_time or next_departure != float("inf"):
        if next_arrival <= next_departure and next_arrival < end_time:
            # Evenement d'arrivee
            clock = next_arrival

            # Mise a jour de N(t)
            sumN = sumN + nbN * (clock - upN)
            nbN = nbN + 1
            upN = clock
            nb_arr += 1

            # Planification de la prochaine arrivee
            inter_arr_t = exponential(lam)
            next_arrival = next_arrival + inter_arr_t

            if not busy:
                # Debut service immediat
                service_t = exponential(mu)
                next_departure = clock + service_t

                # Mise a jour metriques temporelles
                sumT = sumT + service_t
                sumTs = sumTs + service_t
                upS = clock
                current_service_start = clock
                current_customer_arrival = clock
                busy = True
            else:
                # Mise en file d'attente
                queue.append(clock)

                # Mise a jour de Nq(t)
                sumQ = sumQ + nbQ * (clock - upQ)
                nbQ = nbQ + 1
                upQ = clock
        else:
            # Evenement de depart
            clock = next_departure

            # Mise a jour de N(t)
            sumN = sumN + nbN * (clock - upN)
            nbN = nbN - 1
            upN = clock
            nb_dep += 1

            # Client qui vient de quitter le systeme
            if current_service_start is not None and current_customer_arrival is not None:
                sumTq = sumTq + (current_service_start - current_customer_arrival)

            if nbQ > 0:
                # Un client en attente commence son service
                sumQ = sumQ + nbQ * (clock - upQ)
                nbQ = nbQ - 1
                upQ = clock

                first = queue.pop(0)
                service_t = exponential(mu)
                next_departure = clock + service_t

                # Mise a jour metriques temporelles
                sumT = sumT + (clock - first) + service_t
                sumTq = sumTq + (clock - first)
                sumTs = sumTs + service_t

                current_service_start = clock
                current_customer_arrival = first
            else:
                # Plus de client a servir
                sumS += clock - upS
                busy = False
                next_departure = float("inf")
                current_service_start = None
                current_customer_arrival = None

    # Calculs finaux
    N = sumN / clock if clock > 0 else 0.0
    Nq = sumQ / clock if clock > 0 else 0.0
    SU = sumS / clock if clock > 0 else 0.0
    T = sumT / nb_dep if nb_dep > 0 else 0.0
    Tq = sumTq / nb_dep if nb_dep > 0 else 0.0
    Ts = sumTs / nb_dep if nb_dep > 0 else 0.0

    if verbose:
        print(" =================Simulateur du modele M/M/1====================")
        print("N =", round(N, 3))
        print("Nq =", round(Nq, 3))
        print("T =", round(T, 3))
        print("Tq =", round(Tq, 3))
        print("Ts =", round(Ts, 3))
        print("Utilisation du serveur :", round(SU, 3))

    return N, Nq, T, Tq, Ts, SU


def plot_tp3_metrics(N, Nq, T, Tq, Ts, SU):
    labels = ["N", "Nq", "T", "Tq", "Ts", "SU"]
    values = [N, Nq, T, Tq, Ts, SU]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=["#2563eb", "#3b82f6", "#0ea5e9", "#06b6d4", "#14b8a6", "#10b981"])
    plt.title("TP3 M/M/1 Simulation Metrics")
    plt.ylabel("Value")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Exemple d'execution
    result = simulate(lam=2.0, mu=3.0, end_time=10_000, verbose=True)
    plot_tp3_metrics(*result)