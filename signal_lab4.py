import numpy as np
import matplotlib.pyplot as plt


n = 5
N = 100 * n

def f(t):
    return t ** (2 * n)


def fourier_integral_k(k, T, N):
    cycles = max(1, 2 * N * k / T)
    num_steps = int(100 * cycles)
    num_steps = max(num_steps, 4000)
    num_steps = min(num_steps, 50000)

    omega_k = 2 * np.pi * k / T
    t = np.linspace(-N, N, num_steps)
    ft = f(t)

    cos_part = np.cos(omega_k * t)
    sin_part = np.sin(omega_k * t)

    ReF = np.trapezoid(ft * cos_part, t)
    ImF = np.trapezoid(-ft * sin_part, t)

    return ReF, ImF


def amplitude_spectrum(ReF, ImF):
    return np.sqrt(ReF**2 + ImF**2)


def main():
    periods = [4, 8, 16, 32, 64, 128]
    K_MAX = 10
    k_values = np.arange(0, K_MAX + 1)

    for T in periods:
        print(f"\n=======================================")
        print(f"     ОБРОБКА T = {T} ")
        print(f"=======================================")

        Re_vals = []
        Im_vals = []
        Amp_vals = []
        omega_vals = []

        for k in k_values:
            ReF, ImF = fourier_integral_k(k, T, N)
            Amp = amplitude_spectrum(ReF, ImF)
            omega_k = 2 * np.pi * k / T

            Re_vals.append(ReF)
            Im_vals.append(ImF)
            Amp_vals.append(Amp)
            omega_vals.append(omega_k)

            print(f"k={k:2d} | ω_k={omega_k:10.5f} | ReF={ReF:12.5e} | ImF={ImF:12.5e} | |F|={Amp:12.5e}")

        Re_vals = np.array(Re_vals)
        Amp_vals = np.array(Amp_vals)
        omega_vals = np.array(omega_vals)

        mask = k_values > 0

        # ---- очистка ----
        Amp_plot = Amp_vals.copy()
        Re_plot = Re_vals.copy()

        amp_thr = np.percentile(Amp_plot[mask], 95)
        re_thr = np.percentile(np.abs(Re_plot[mask]), 95)

        Amp_plot[Amp_plot > amp_thr] = amp_thr
        Re_plot[Re_plot > re_thr] = re_thr
        Re_plot[Re_plot < -re_thr] = -re_thr

        # ============================================================
        #      ЄДИНЕ ВІКНО ДЛЯ ДВОХ ГРАФІКІВ (subplot)
        # ============================================================

        fig, ax = plt.subplots(2, 1, figsize=(10, 7))

        # ReF
        ax[0].stem(omega_vals[mask], Re_plot[mask])
        ax[0].set_title(f"Дійсна частина Re F(ω_k) для T = {T}")
        ax[0].set_xlabel("ω_k = 2πk / T")
        ax[0].set_ylabel("Re F")
        ax[0].grid(True)

        # |F|
        ax[1].stem(omega_vals[mask], Amp_plot[mask])
        ax[1].set_title(f"Амплітудний спектр |F(ω_k)| для T = {T}")
        ax[1].set_xlabel("ω_k = 2πk / T")
        ax[1].set_ylabel("|F|")
        ax[1].grid(True)

        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
