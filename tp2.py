from math import factorial


def _validate_rates(lamb, mu):
    if lamb < 0:
        raise ValueError("lambda must be greater than or equal to 0")
    if mu <= 0:
        raise ValueError("mu must be strictly greater than 0")


def _validate_servers(servers):
    if int(servers) != servers or servers <= 0:
        raise ValueError("servers must be a positive integer")
    return int(servers)


# Exercise 1: M/M/1

#Calcul de la charge
def rho(lamb, mu):
    _validate_rates(lamb, mu)
    return lamb / mu

#Calcul de la probabilite d'avoir 0 client
def p0(lamb, mu):
    return 1 - rho(lamb, mu)

#Calcul du nombre moyen de clients
def N(lamb, mu):
    current_rho = rho(lamb, mu)
    if current_rho >= 1:
        raise ValueError(f"Le systeme n'est pas stationnaire : rho = {current_rho:.3f} >= 1")
    return current_rho / (1 - current_rho)

#Calcul du nombre moyen de clients dans la file d'attente
def Nq(lamb, mu):
    current_rho = rho(lamb, mu)
    if current_rho >= 1:
        raise ValueError(f"Le systeme n'est pas stationnaire : rho = {current_rho:.3f} >= 1")
    return current_rho ** 2 / (1 - current_rho)

#Calcul du nombre moyen de clients dans le systeme
def Ns(lamb, mu):
    current_rho = rho(lamb, mu)
    if current_rho >= 1:
        raise ValueError(f"Le systeme n'est pas stationnaire : rho = {current_rho:.3f} >= 1")
    return current_rho

#Calcul du temps moyen d'attent dans le sys
def T(lamb, mu):
    current_rho = rho(lamb, mu)
    if current_rho >= 1:
        raise ValueError(f"Le systeme n'est pas stationnaire : rho = {current_rho:.3f} >= 1")
    return 1 / (mu - lamb)

#Calcul du temps moyen d'attent dans la file d'attente
def Tq(lamb, mu):
    current_rho = rho(lamb, mu)
    if current_rho >= 1:
        raise ValueError(f"Le systeme n'est pas stationnaire : rho = {current_rho:.3f} >= 1")
    return current_rho / (mu - lamb)

#Calcul du temps moyen de service
def Ts(lamb, mu):
    _validate_rates(lamb, mu)
    return 1 / mu

#Calcul de la probabilite d'avoir n clients
def p_n(n, lamb, mu):
    if n < 0:
        raise ValueError("n must be greater than or equal to 0")
    current_rho = rho(lamb, mu)
    if current_rho >= 1:
        raise ValueError(f"Le systeme n'est pas stationnaire : rho = {current_rho:.3f} >= 1")
    return (1 - current_rho) * (current_rho ** n)

#Calcul de la probabilite d'avoir n clients
def probability_n(n, lamb, mu):
    return p_n(n, lamb, mu)

#Calcul des probabilite d'avoir n clients
def calculate_probabilities(lamb, mu, max_n):
    

    if max_n < 0:
        raise ValueError("max_n must be greater than or equal to 0")
    return [p_n(n, lamb, mu) for n in range(max_n + 1)]

#Calcul des metriques
def mm1_metrics(lamb, mu):

    current_rho = rho(lamb, mu)
    result = {
        "lambda": float(lamb),
        "mu": float(mu),

        "rho": current_rho,
        "stable": current_rho < 1,
    }

    if current_rho >= 1:

        result["message"] = f"Le systeme n'est pas stationnaire : rho = {current_rho:.3f} >= 1"
        return result

    result.update(
        {
            "p0": p0(lamb, mu),
            "N": N(lamb, mu),

            "Nq": Nq(lamb, mu),
            "Ns": Ns(lamb, mu),
            "T": T(lamb, mu),
            "Tq": Tq(lamb, mu),
            "Ts": Ts(lamb, mu),
        }
    )
    return result


# Exercise 2: M/M/S
#Calcul de la charge
def offered_load(lamb, mu):

    return lamb / mu

#Calcul de l'utilisation
def utilization(lamb, mu, s):
    return lamb / (s * mu)

#Calcul de la probabilite d'avoir 0 client
def p0_mm_s(lamb, mu, s):
    a = offered_load(lamb, mu)
    rho = utilization(lamb, mu, s)

    if rho >= 1:
        raise ValueError("Systeme non stable (rho >= 1)")

    normalizer = sum((a ** n) / factorial(n) for n in range(s))
    normalizer += (a ** s) / (factorial(s) * (1 - rho))

    return 1 / normalizer

#proba d'attente
def erlang_c(lamb, mu, s):
    a = offered_load(lamb, mu)

    rho = utilization(lamb, mu, s)
    p0 = p0_mm_s(lamb, mu, s)

    return ((a ** s) / (factorial(s) * (1 - rho))) * p0

#Calcul des metriques
def metrics_mm_s(lamb, mu, s):

    a = offered_load(lamb, mu)
    rho = utilization(lamb, mu, s)
    p0 = p0_mm_s(lamb, mu, s)
    pw = erlang_c(lamb, mu, s)

    Nq = (pw * rho) / (1 - rho)
    N = Nq + a
    Ns = a

    Tq = Nq / lamb
    T = Tq + (1 / mu)
    Ts = 1 / mu

    return {
        "a": a,
        "rho": rho,
        "P0": p0,
        "Pw": pw,
        "N": N,
        "Nq": Nq,
        "Ns": Ns,
        "T": T,
        "Tq": Tq,
        "Ts": Ts
    }

#Calcul de la probabilite d'avoir n clients
def p_n_mm_s(lamb, mu, s, n):
    a = offered_load(lamb, mu)
    rho = utilization(lamb, mu, s)
    p0 = p0_mm_s(lamb, mu, s)

    if n < s:
        return (a ** n) / factorial(n) * p0
    else:
        return (a ** n) / (factorial(s) * (s ** (n - s))) * p0


# Adapters to global code
def calculate_probabilities_mm_s(lamb, mu, servers, max_n):
    if max_n < 0:
        raise ValueError("max_n must be greater than or equal to 0")
    if utilization(lamb, mu, servers) >= 1:
        return []
    return [p_n_mm_s(lamb, mu, servers, n) for n in range(max_n + 1)]


#Calcul des metriques
def mm_s_metrics(lamb, mu, servers):
    rho = utilization(lamb, mu, servers)
    if rho >= 1:
        return {
            "lambda": float(lamb),
            "mu": float(mu),
            "servers": servers,
            "a": rho,
            "stable": False,
            "message": f"Le systeme n'est pas stationnaire : a = {rho:.3f} >= 1"
        }
        
    m = metrics_mm_s(lamb, mu, servers)
    return {
        "lambda": float(lamb),
        "mu": float(mu),
        "servers": servers,
        "stable": True,
        "a": m["rho"],
        "p0": m["P0"],
        "pw": m["Pw"],
        "N": m["N"],
        "Nq": m["Nq"],
        "Ns": m["Ns"],
        "T": m["T"],
        "Tq": m["Tq"],
        "Ts": m["Ts"],
    }


def main():
    print("TP2 is available through the Flask interface.")
    print("Run: python app.py")
    print("Then open: http://127.0.0.1:5000")
    print("Use the 'TP2 Queueing' tab in the interface.")


if __name__ == "__main__":
    main()