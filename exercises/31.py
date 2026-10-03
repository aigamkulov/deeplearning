import numpy as np

def sensor_samples(mu, sigma_r, n, rng, g=1.0):
    return g * rng.poisson(mu, n) + rng.normal(0.0, sigma_r, n)

rng = np.random.default_rng(42)
print(" mu    var_meas  mu+sr^2  SNR_meas  SNR_theory  sqrt(mu)")
for mu in (5, 50, 500, 5000):
    x = sensor_samples(mu, 4.0, 100_000, rng)
    print(f"{mu:5d} {x.var():9.2f} {mu + 16:8d} {x.mean() / x.std():9.3f} "
          f"{mu / np.sqrt(mu + 16):11.3f} {np.sqrt(mu):9.3f}")
