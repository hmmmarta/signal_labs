import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.integrate import quad

n = 5
a = 0
b = np.pi

def f(x):
    return n * np.sin(np.pi * n * x)

def b_k_quad(k):
    integrand = lambda x: f(x) * np.sin(k * x)
    result, _ = quad(integrand, a, b)
    return (2 / np.pi) * result

def S_N(x, N):
    return sum(b_k_quad(k) * np.sin(k * x) for k in range(1, N + 1))

def err(x, N):
    return (f(x) - S_N(x, N)) / f(x)

def save_results(x, N, filename="results.csv"):
    with open(filename, mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["Результати розкладу в ряд Фур'є"])
        writer.writerow([f"n = {n}", f"Інтервал = [{a}, {b}]", f"N = {N}"])
        writer.writerow([])

        writer.writerow(["k", "b_k"])
        for k in range(1, N + 1):
            writer.writerow([k, f"{b_k_quad(k):.6f}"])

        writer.writerow([])
        writer.writerow(["x", "f(x)", "S_N(x)", "Відносна похибка"])
        writer.writerow([f"{x:.6f}", f"{f(x):.6f}", f"{S_N(x, N):.6f}", f"{err(x, N):.6e}"])

    print(f"Результати збережено у файл '{filename}'")

def graf_all(N):
    """Порівняння f(x) і наближення + спектр коефіцієнтів b_k"""
    x = np.linspace(0, 0.1, 500)
    y_true = f(x)
    y_approx = [S_N(xi, N) for xi in x]

    ks = np.arange(1, N + 1)
    bs = [b_k_quad(k) for k in ks]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # --- Ліва частина: f(x) і наближення ---
    axs[0].plot(x, y_true, label="f(x)", linewidth=2)
    axs[0].plot(x, y_approx, label=f"S_N(x), N={N}", linestyle="--")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("Значення")
    axs[0].set_title("Точна функція та наближення рядом Фур'є")
    axs[0].legend()
    axs[0].grid(True)

    # --- Права частина: спектр коефіцієнтів ---
    axs[1].stem(ks, bs, basefmt=" ", markerfmt="o", linefmt="-")
    axs[1].set_xlabel("k")
    axs[1].set_ylabel("b_k")
    axs[1].set_title("Спектр коефіцієнтів b_k")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    N = 50
    x = np.pi / 2

    print(f"Обчислення для x = {x:.4f}, N = {N}")
    print(f"f(x) = {f(x):.6f}")
    print(f"S_N(x) = {S_N(x, N):.6f}")
    print(f"Відносна похибка = {err(x, N):.6e}")

    graf_all(N)
    save_results(x, N, filename="results.csv")
