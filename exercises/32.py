import numpy as np

def photon_transfer(g, sigma_r, mus, rng, shape=(200, 200)):
    means, variances = [], []
    for mu in mus:
        a = g * rng.poisson(mu, shape) + g * rng.normal(0, sigma_r, shape)
        b = g * rng.poisson(mu, shape) + g * rng.normal(0, sigma_r, shape)
        means.append(0.5 * (a.mean() + b.mean()))
        variances.append((a - b).var() / 2)
    slope, icpt = np.polyfit(means, variances, 1)
    return np.array(means), np.array(variances), slope, np.sqrt(max(icpt, 0)) / slope

rng = np.random.default_rng(42)
m, v, gain, sr = photon_transfer(0.5, 3.0, [20, 50, 100, 200, 400, 800, 1600], rng)
print("mean_DN =", np.round(m, 1).tolist())
print("var_DN  =", np.round(v, 1).tolist())
print(f"estimated gain = {gain:.4f} DN/e (true 0.5), read noise = {sr:.2f} e (true 3.0), "
      f"electrons at 4095 DN = {4095 / gain:.0f} e")
