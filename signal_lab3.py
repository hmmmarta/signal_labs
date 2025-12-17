import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# -----------------------
# Початкові параметри
# -----------------------
n = 100
N = 10 + n
np.random.seed(42)
f = np.random.rand(N)
print(f"Вхідний сигнал f (N={N}):\n{f}\n")

# -----------------------
# Повільне ДПФ з підрахунком операцій
# -----------------------
def dft_slow_with_ops(f, N):
    start_time = time.time()
    Ck = np.zeros(N, dtype=complex)
    mul_ops = 0  # реальні множення
    add_ops = 0  # реальні додавання

    for k in range(N):
        acc = 0 + 0j
        for i in range(N):
            angle = -2 * np.pi * k * i / N
            c = np.cos(angle)
            s = np.sin(angle)
            # комплексне множення: 4 множення, 2 додавання
            mul_ops += 4
            add_ops += 2
            acc += f[i] * (c + 1j * s)
            # комплексне додавання: 2 додавання
            add_ops += 2
        Ck[k] = acc / N
    time_elapsed = time.time() - start_time
    return Ck, time_elapsed, mul_ops, add_ops

# -----------------------
# Реалізація radix-2 FFT з підрахунком операцій
# -----------------------
def next_pow2(x):
    return 1 << (x - 1).bit_length()

def bit_reverse_indices(n):
    bits = int(np.log2(n))
    indices = np.arange(n)
    rev = np.zeros(n, dtype=int)
    for i in range(n):
        b = '{:0{width}b}'.format(i, width=bits)
        rev[i] = int(b[::-1], 2)
    return rev

def fft_radix2_with_ops(x):
    n = len(x)
    assert (n & (n - 1)) == 0, "Довжина повинна бути степенем двійки!"
    X = x.copy().astype(complex)

    mul_ops = 0
    add_ops = 0

    rev = bit_reverse_indices(n)
    X = X[rev]

    m = 1
    while m < n:
        m2 = 2 * m
        for k in range(0, n, m2):
            for j in range(m):
                angle = -2 * np.pi * j / m2
                W = np.cos(angle) + 1j * np.sin(angle)
                # комплексне множення
                mul_ops += 4
                add_ops += 2
                t = W * X[k + j + m]
                u = X[k + j]
                # два комплексних додавання
                add_ops += 4
                X[k + j] = u + t
                X[k + j + m] = u - t
        m = m2

    X = X / n  # нормалізація
    return X, mul_ops, add_ops

# -----------------------
# Обчислення DFT і FFT
# -----------------------
Ck_slow, time_slow, mul_slow, add_slow = dft_slow_with_ops(f, N)

M = next_pow2(N)
if M != N:
    x_padded = np.zeros(M, dtype=complex)
    x_padded[:N] = f
else:
    x_padded = f.astype(complex)

start_time = time.time()
Ck_fft, mul_fft, add_fft = fft_radix2_with_ops(x_padded)
time_fft = time.time() - start_time

# -----------------------
# Порівняння з NumPy FFT
# -----------------------
start_time = time.time()
Ck_np = np.fft.fft(f) / N
time_np = time.time() - start_time

# -----------------------
# Результати
# -----------------------
summary = pd.DataFrame({
    "Метод": ["Повільне ДПФ", f"Radix-2 ШПФ (M={M})", "NumPy FFT"],
    "Час, с": [time_slow, time_fft, time_np],
    "Множення (≈)": [mul_slow, mul_fft, "—"],
    "Додавання (≈)": [add_slow, add_fft, "—"]
})

print("=== Порівняння результатів ===")
print(summary.to_string(index=False))
print(f"\nМакс. різниця між ДПФ та ШПФ: {np.max(np.abs(Ck_slow - Ck_fft[:N])):.3e}")
print(f"Макс. різниця між ДПФ та NumPy FFT: {np.max(np.abs(Ck_slow - Ck_np)):.3e}\n")

# -----------------------
# Побудова спектрів
# -----------------------
plt.figure(figsize=(9, 5))
plt.stem(range(N), np.abs(Ck_slow), linefmt='b-', markerfmt='bo', basefmt=' ')
plt.stem(range(N), np.abs(Ck_fft[:N]), linefmt='r--', markerfmt='ro', basefmt=' ')
plt.title('Порівняння спектрів ДПФ та ШПФ')
plt.xlabel('k')
plt.ylabel('|Ck|')
plt.legend(['ДПФ (повільне)', 'ШПФ (radix-2)'])
plt.grid(True)
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import time
#
# import math
#
# # ------------------------------------------------------------
# # УТИЛІТИ
# # ------------------------------------------------------------
# def dft_slow(f):
#     """Повільна ДПФ з нормалізацією 1/N (O(N^2)). Повертає (Ck, час, лічильники)."""
#     N = len(f)
#     t0 = time.perf_counter()
#     Ck = np.zeros(N, dtype=complex)
#
#     mul_cnt = 0  # приблизна кількість комплексних множень
#     add_cnt = 0  # приблизна кількість комплексних додавань
#
#     for k in range(N):
#         acc = 0j
#         for n in range(N):
#             angle = -2 * np.pi * k * n / N
#             w = np.cos(angle) + 1j * np.sin(angle)   # e^{-j 2πkn/N}
#             acc += f[n] * w
#             mul_cnt += 1
#             if n > 0:
#                 add_cnt += 1
#         Ck[k] = acc / N
#     dt = time.perf_counter() - t0
#     return Ck, dt, mul_cnt, add_cnt
#
#
# # def dft_fft(f):
# #     """Швидка ДПФ через numpy з тією ж нормою 1/N."""
# #     t0 = time.perf_counter()
# #     Ck = np.fft.fft(f) / len(f)
# #     return Ck, time.perf_counter() - t0
# def dft_fft(f):
#     """Швидка ДПФ через numpy з тією ж нормою 1/N + підрахунок операцій."""
#     N = len(f)
#     t0 = time.perf_counter()
#     Ck = np.fft.fft(f) / N
#     t = time.perf_counter() - t0
#
#     # Теоретичні оцінки кількості операцій
#     additions = int(N * math.log2(N))            # додавання
#     multiplications = int((N / 2) * math.log2(N))  # множення
#
#     print(f"N = {N}")
#     print(f"Теоретичні операції FFT:")
#     print(f"  Додавань:     {additions}")
#     print(f"  Множень:      {multiplications}")
#     print(f"  Разом:        {additions + multiplications}")
#     print(f"Час виконання: {t:.8f} c")
#
#     return Ck, t
#
# def trig_term_for_k(f, k):
#     """
#     Тригонометрична форма ОДНОГО k:
#       A_k = (2/N) sum f[n] cos(2πkn/N)
#       B_k = (2/N) sum f[n] sin(2πkn/N)
#     Повертає (A_k, B_k).
#     """
#     N = len(f)
#     n = np.arange(N)
#     Ak = (2.0 / N) * np.sum(f * np.cos(2 * np.pi * k * n / N))
#     Bk = (2.0 / N) * np.sum(f * np.sin(2 * np.pi * k * n / N))
#     return float(Ak), float(Bk)
#
#
# def reconstruct_analog_from_Ck(t, Tc, Ck):
#     """
#     Реальна косинусна форма (з нормою 1/N).
#     Для парного N додаємо термін k=N/2 окремо.
#     """
#     N = Ck.size
#     s = Ck[0].real
#     half = N // 2
#     for k in range(1, half):
#         s += 2 * np.abs(Ck[k]) * np.cos(2 * np.pi * k * t / Tc + np.angle(Ck[k]))
#     if N % 2 == 0:
#         s += np.abs(Ck[half]) * np.cos(np.pi * N * t / Tc + np.angle(Ck[half]))
#     return s
#
#
# # ------------------------------------------------------------
# # ЧАСТИНА I: ДПФ (довільний вектор, N=10+n)
# # ------------------------------------------------------------
# print("=" * 70)
# print("ЧАСТИНА I: ДПФ")
# print("=" * 70)
#
# n = 5
# N = 10 + n
# rng = np.random.default_rng(42)
# f = rng.random(N)  # довільний вхідний вектор
#
# # 1) Тригонометрична форма для одного k (приклад: k=3)
# k_demo = 3
# Ak, Bk = trig_term_for_k(f, k_demo)
# print(f"[I.1] Триг-форма для k={k_demo}:  A_k={Ak:.6f}, B_k={Bk:.6f}")
#
# # 2) Обчислення всіх коефіцієнтів Ck (повільно) і порівняння з FFT
# Ck_slow, t_slow, mul_cnt, add_cnt = dft_slow(f)
# Ck_fft, t_fft = dft_fft(f)
# max_diff = np.max(np.abs(Ck_slow - Ck_fft))
# print(f"[I.2] Макс. різниця DFT_slow vs FFT: {max_diff:.2e}")
#
# # 5a) Час обчислення
# print(f"[I.5a] Час: повільна ДПФ = {t_slow:.6f}s, FFT = {t_fft:.6f}s, прискорення ×{t_slow/t_fft:.2f}")
#
# # 5b) Оціночна кількість операцій
# print(f"[I.5b] Операції (приблизно, комплексні): множень ~{mul_cnt}, додавань ~{add_cnt}")
# if (N & (N - 1)) == 0:
#     butterflies = (N // 2) * int(np.log2(N))
#     print(f"       Для FFT (N=2^m): ~{butterflies} ''")
# else:
#     print("       FFT-оцінка коректна лише для N=2^m")
#
#
# # 3) Спектри амплітуд і фаз
# amp = np.abs(Ck_fft)
# phs = np.angle(Ck_fft)
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.stem(range(N), amp)
# plt.title('Спектр амплітуд (N=10+n)')
# plt.xlabel('k'); plt.ylabel('|Ck|'); plt.grid(True)
#
#
# plt.subplot(2, 1, 2)
# plt.stem(range(N), phs)
# plt.title('Спектр фаз (N=10+n)')
# plt.xlabel('k'); plt.ylabel('arg(Ck)'); plt.grid(True)
# plt.tight_layout(); plt.show()
#
#
# # ------------------------------------------------------------
# # ЧАСТИНА II: Відтворення аналогового сигналу (N=8 відліків)
# # ------------------------------------------------------------
# print("\n" + "=" * 70)
# print("ЧАСТИНА II: Відтворення аналогового сигналу (N=8)")
# print("=" * 70)
#
# # 1) Генеруємо 8-бітну послідовність з N_binary = 96+n, виставляємо 8-й РОЗРЯД (MSB)
# N_binary = 96 + n                   # 105
# binary_repr = format(N_binary, '08b')  # рівно 8 біт
# bits = list(binary_repr)
# # 8-й розряд (MSB): для НЕпарного n = 1, для парного = 0
# bits[0] = '1' if (n % 2 == 1) else '0'
# s_bits = ''.join(bits)
# s_n = np.array([int(b) for b in s_bits], dtype=float)  # 8 рівновіддалених відліків
# N8 = len(s_n)
# print(f"[II.1] 8-бітна послідовність (MSB виправлено): {s_bits}")
# print(f"        Відліки s[n], n=0..7: {s_n.astype(int)}")
#
# # 2) ДПФ цих 8 відліків (норма 1/N): модулі й фази ВСІ
# Ck8 = np.fft.fft(s_n) / N8
# print("\n[II.2] Коефіцієнти C_k (усі 8):")
# for k in range(N8):
#     print(f" k={k}: Ck={Ck8[k]: .6f}, |Ck|={np.abs(Ck8[k]):.6f}, arg={np.angle(Ck8[k]):.6f}")
#
# # 3) Явний вираз s(t) і графік
# Tc = 1.0
# print("\n[II.3] Явний вираз s(t) у косинусній формі для N=8:")
# # s(t) = C0 + 2 Σ_{k=1..3} |Ck| cos(2πk t/Tc + arg Ck) + |C4| cos(4π t/Tc + arg C4)
# for k in range(N8):
#     print(f"   k={k}: |Ck|={np.abs(Ck8[k]):.6f}, φ_k={np.angle(Ck8[k]):.6f}")
# print("\nАНАЛІТИЧНИЙ ВИГЛЯД (через експоненту):\n")
# for k in range(N8):
#     expr_terms = []
#     for n_ in range(N8):
#         angle_str = f"-j·2π·{k}·{n_}/{N8}"
#         term_str = f"{s_n[n_]:.0f}·e^({angle_str})"
#         expr_terms.append(term_str)
#     expr_str = " + ".join(expr_terms)
#     print(f"C[{k}] = (1/{N8})·({expr_str})")
#
# t = np.linspace(0, Tc, 1000, endpoint=False)
# s_t = reconstruct_analog_from_Ck(t, Tc, Ck8)
#
# plt.figure(figsize=(12, 6))
# plt.plot(t, s_t, linewidth=2, label='Відтворений s(t)')
# ts = np.linspace(0, Tc, N8, endpoint=False)
# (markerline, stemlines, baseline) = plt.stem(ts, s_n,  label='s[n]')
# plt.setp(baseline, visible=False)
# plt.title('Відтворення аналогового сигналу з 8 відліків')
# plt.xlabel('t'); plt.ylabel('Амплітуда'); plt.grid(True); plt.legend()
# plt.show()
#
# # ------------------------------------------------------------
# # ЧАСТИНА III: ОДПФ (зворотне перетворення)
# # ------------------------------------------------------------
# print("\n" + "=" * 70)
# print("ЧАСТИНА III: ОДПФ")
# print("=" * 70)
#
# # 1) Зворотне перетворення (узгоджене з нормою 1/N): ifft(Ck * N)
# s_rec = np.fft.ifft(Ck8 * N8)
# print("[III.1] Відновлені відліки s(nTδ) (реальна частина):")
# for i in range(N8):
#     print(f" n={i}: {s_rec[i].real:.6f}  (оригінал: {int(s_n[i])})")
#
# # 2) Аналітичні вирази для n=0,1 і порівняння
# #    s[0] = sum Ck
# s0 = np.sum(Ck8)
# #    s[1] = sum Ck * e^{j 2π k / 8}
# k = np.arange(N8)
#
# s1 = np.sum(Ck8 * np.exp(1j * 2 * np.pi * k / N8))
#
# print("\n[III.2] Аналітичні значення:")
# print(f" s[0] = Σ Ck = {s0.real:.6f}   (похибка: {abs(s0.real - s_n[0]):.2e})")
#
# print(f" s[1] = Σ Ck e^(j2πk/8) = {s0.real:.6f}   (похибка: {abs(s1.real - s_n[1]):.2e})")
# print(f" s[1] = Σ Ck e^(j2πk/8) = {s1.real:.6f}   (похибка: {abs(s1.real - s_n[1]):.2e})")
#
