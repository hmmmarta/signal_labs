import numpy as np
import matplotlib.pyplot as plt
import time

n = 5
N = 10 + n  # для Частини I

print("=" * 60)
print("ЧАСТИНА I: ДПФ")
print("=" * 60)

# -----------------------
# Функції
# -----------------------
def dft_single_trig(f, k, N):
    A = 0.0
    B = 0.0
    Ck = 0+0j
    for m in range(N):
        angle = 2 * np.pi * k * m / N

        A += f[m] * np.cos(angle)
        B += f[m] * (-np.sin(angle))
        # комплексний варіант
        Ck += f[m] * (np.cos(angle) - 1j * np.sin(angle))
    A = (2.0 / N) * A
    B = (2.0 / N) * B
    Ck = Ck / N
    return A, B, Ck

def correct_dft_slow_with_ops(f, N):
    start_time = time.time()
    Ck = np.zeros(N, dtype=complex)
    mul_ops = 0
    add_ops = 0
    for k in range(N):
        acc = 0+0j
        for i in range(N):
            angle = -2 * np.pi * k * i / N
            # обчислення e^{-j angle} як cos + j sin
            c = np.cos(angle)
            s = np.sin(angle)
            mul_ops += 4
            add_ops += 2
            acc += f[i] * (c + 1j * s)
            add_ops += 2
        Ck[k] = acc / N
    time_elapsed = time.time() - start_time
    return Ck, time_elapsed, mul_ops, add_ops

# -----------------------
# Генерація вхідного вектору (Частина I)
# -----------------------
np.random.seed(42)
f = np.random.rand(N)
print(f"Розмірність ДПФ: N = {N}")

# Обчислення повільним методом з підрахунком операцій
Ck_slow, time_slow, mul_ops, add_ops = correct_dft_slow_with_ops(f, N)

# FFT (узгоджене масштабування)
start_time = time.time()
Ck_fast = np.fft.fft(f) / N
time_fast = time.time() - start_time

print(f"Повільне ДПФ: {time_slow:.6f} сек")
print(f"Швидке ДПФ (FFT): {time_fast:.6f} сек")
print(f"Максимальна різниця: {np.max(np.abs(Ck_slow - Ck_fast)):.2e}")
print(f"Орієнтовна кількість операцій (повільний): множень ≈ {mul_ops}, додавань ≈ {add_ops}")

# Побудова спектрів
amplitude_spectrum = np.abs(Ck_fast)
phase_spectrum = np.angle(Ck_fast)

plt.figure(figsize=(10, 6))
plt.subplot(2,1,1)
plt.stem(range(N), amplitude_spectrum)
plt.title('Спектр амплітуд')
plt.xlabel('k')
plt.ylabel('|Ck|')
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(range(N), phase_spectrum)
plt.title('Спектр фаз')
plt.xlabel('k')
plt.ylabel('arg(Ck)')
plt.grid(True)
plt.tight_layout()
plt.show()

k_test = 1
A_k, B_k, Ck_k = dft_single_trig(f, k_test, N)
print(f"\nДля k={k_test}: A_k = {A_k:.6f}, B_k = {B_k:.6f}, C_k = {Ck_k:.6f}")

# =============================================================================
# ЧАСТИНА II: ВІДТВОРЕННЯ
# =============================================================================
print("\n" + "=" * 60)
print("ЧАСТИНА II: відтворення")
print("=" * 60)

N_binary = 96 + n
binary_repr = format(N_binary, '08b')
binary_list = list(binary_repr)

if (n % 2) == 1:
    binary_list[0] = '1'
else:
    binary_list[0] = '0'

modified_binary = ''.join(binary_list)
s_n = np.array([int(bit) for bit in modified_binary], dtype=float)
N_samples = len(s_n)

print(f"N_binary (96 + n) = {N_binary}, двійковий: {binary_repr} -> модифікований: {modified_binary}")
print(f"Відліки сигналу (8 точок): {s_n}")

Ck_analog = np.fft.fft(s_n) / N_samples
magnitudes = np.abs(Ck_analog)
arguments = np.angle(Ck_analog)

print("\nКоефіцієнти ДПФ (всі):")
for k in range(N_samples):
    print(f"k={k:2d}: Ck = {Ck_analog[k]: .6f}, |Ck| = {magnitudes[k]: .6f}, arg = {arguments[k]: .6f} ")

# ВІДТВОРЕННЯ АНАЛОГОВОГО СИГНАЛУ
def reconstruct_analog_signal(t, Tc, Ck, N):
    s_t = np.zeros_like(t, dtype=float)
    s_t += np.abs(Ck[0])  # постійна складова
    half = N // 2
    for k in range(1, half):
        s_t += 2.0 * np.abs(Ck[k]) * np.cos(2 * np.pi * k * t / Tc + np.angle(Ck[k]))
    if N % 2 == 0:
        s_t += np.abs(Ck[half]) * np.cos(np.pi * N * t / Tc + np.angle(Ck[half]))
    return s_t

Tc = 1.0
t_continuous = np.linspace(0, Tc, 1000)
s_reconstructed_analog = reconstruct_analog_signal(t_continuous, Tc, Ck_analog, N_samples)

plt.figure(figsize=(10,5))
plt.plot(t_continuous, s_reconstructed_analog, linewidth=2, label='Відтворений аналоговий сигнал')
plt.stem(np.linspace(0, Tc, N_samples, endpoint=False), s_n, 'ro', basefmt=' ', label='Дискретні відліки')
plt.title('Відтворення аналогового сигналу з дискретних відліків (8 точок)')
plt.xlabel('Час (t)')
plt.ylabel('Амплітуда')
plt.legend()
plt.grid(True)
plt.show()
print("\nАНАЛІТИЧНИЙ ВИГЛЯД (через експоненту):\n")

for k in range(N_samples):
    expr_terms = []
    for n_ in range(N_samples):
        exp_term = f"e^(-j·2π·{k}·{n_}/{N_samples})"
        term_str = f"{s_n[n_]:.0f}·{exp_term}"
        expr_terms.append(term_str)
    expr_str = " + ".join(expr_terms)
    print(f"C[{k}] = (1/{N_samples})·({expr_str}) "
          f"= {Ck_analog[k].real:.4f} {'+' if Ck_analog[k].imag >= 0 else '-'} j{abs(Ck_analog[k].imag):.4f}")


# =============================================================================
# ЧАСТИНА III: ОДПФ (зворотнє)
# =============================================================================
print("\n" + "=" * 60)
print("ЧАСТИНА III: ОДПФ")
print("=" * 60)

s_reconstructed = np.fft.ifft(Ck_analog * N_samples)

print("Відліки (відновлені):")
for i in range(N_samples):
    print(f"s({i}T) = {s_reconstructed[i].real:.6f} (оригінал: {s_n[i]:.0f})")

# АНАЛІТИЧНИЙ РОЗРАХУНОК для n=0,1
print("\n--- АНАЛІТИЧНИЙ РОЗРАХУНОК ---")
for n_idx in [0, 1]:
    s_calc = 0+0j
    for k in range(N_samples):
        angle = 2 * np.pi * k * n_idx / N_samples
        s_calc += Ck_analog[k] * (np.cos(angle) + 1j * np.sin(angle))
    print(f"s({n_idx}) аналітичне = {s_calc.real:.6f} (оригінал: {s_n[n_idx]:.0f}), похибка = {abs(s_n[n_idx] - s_calc.real):.2e}")

