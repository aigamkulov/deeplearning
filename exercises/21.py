import math

def central(f, t, eps):
    return (f(t + eps) - f(t - eps)) / (2 * eps)

f = math.sin
t = 1.0

print("t =", t)
print("Exact derivative cos(t) =", math.cos(t))
print("\n k        eps             |central-cos|")
print("-" * 45)

best_eps = None
best_err = float("inf")

for k in range(1, 13):
    eps = 10.0 ** (-k)
    numerical = central(f, t, eps)
    err = abs(numerical - math.cos(t))
    print(f"{k:2d}   {eps:10.1e}      {err:.3e}")
    if err < best_err:
        best_err = err
        best_eps = eps

print("\nEng yaxshi markaziy ayirma eps =", best_eps)
print("Eng kichik xato =", best_err)
print("\nNazariya: eps juda katta bo'lsa truncation error, juda kichik bo'lsa floating-point cancellation error kuchayadi.")
