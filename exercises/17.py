import numpy as np
import torch
import torch.nn.functional as F

def activations(z, alpha=0.01):
    sigmoid = 1 / (1 + np.exp(-z))
    sigmoid_prime = sigmoid * (1 - sigmoid)

    tanh = np.tanh(z)
    tanh_prime = 1 - tanh ** 2

    relu = np.maximum(0, z)
    relu_prime = np.where(z > 0, 1.0, 0.0)

    leaky_relu = np.where(z > 0, z, alpha * z)
    leaky_relu_prime = np.where(z < 0, alpha, 1.0)

    softplus = np.logaddexp(0, z)
    softplus_prime = 1 / (1 + np.exp(-z))

    return {
        "sigmoid": (sigmoid, sigmoid_prime),
        "tanh": (tanh, tanh_prime),
        "relu": (relu, relu_prime),
        "leaky_relu": (leaky_relu, leaky_relu_prime),
        "softplus": (softplus, softplus_prime),
    }

z = np.linspace(-3, 3, 7)
results = activations(z)

t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
torch_activations = {
    "sigmoid": torch.sigmoid(t),
    "tanh": torch.tanh(t),
    "relu": torch.relu(t),
    "leaky_relu": F.leaky_relu(t, negative_slope=0.01),
    "softplus": F.softplus(t),
}

for name, (phi, phi_prime) in results.items():
    torch_phi = torch_activations[name]
    if t.grad is not None:
        t.grad.zero_()
    torch_phi.sum().backward()
    torch_phi_prime = t.grad.detach().numpy()

    value_match = np.allclose(phi, torch_phi.detach().numpy(), atol=1e-12)
    derivative_match = np.allclose(phi_prime, torch_phi_prime, atol=1e-12)

    print("=" * 60)
    print("Funksiya:", name)
    print("phi(z):       ", np.round(phi, 4))
    print("phi'(z):      ", np.round(phi_prime, 4))
    print("Autograd phi':", np.round(torch_phi_prime, 4))
    print("Qiymat match: ", value_match)
    print("Hosila match: ", derivative_match)
