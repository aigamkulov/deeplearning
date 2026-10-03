import torch
import torch.nn as nn

def rosen(theta):
    t1, t2 = theta[0], theta[1]
    return (1 - t1)**2 + 100 * (t2 - t1**2)**2

theta0 = torch.tensor([-1.2, 1.0], dtype=torch.float64)

def run(name):
    theta = nn.Parameter(theta0.clone())
    if name == "SGD":
        opt = torch.optim.SGD([theta], lr=3e-3)
    elif name == "Momentum":
        opt = torch.optim.SGD([theta], lr=3e-3, momentum=0.9)
    else:
        opt = torch.optim.Adam([theta], lr=0.1)

    for _ in range(500):
        opt.zero_grad()
        loss = rosen(theta)
        loss.backward()
        opt.step()

    with torch.no_grad():
        final_loss = rosen(theta).item()
    return theta.detach(), final_loss

for name in ["SGD", "Momentum", "Adam"]:
    theta, loss = run(name)

    # Float64 PyTorch reference with matching optimizer family.
    ref = nn.Parameter(theta0.clone())
    if name == "SGD":
        opt = torch.optim.SGD([ref], lr=3e-3)
    elif name == "Momentum":
        opt = torch.optim.SGD([ref], lr=3e-3, momentum=0.9)
    else:
        opt = torch.optim.Adam([ref], lr=0.1)

    for _ in range(500):
        opt.zero_grad()
        rosen(ref).backward()
        opt.step()

    print(f"{name:10s} theta500=({theta[0].item():.4f}, {theta[1].item():.4f})  f={loss:.2e}  reference_match={torch.allclose(theta, ref, atol=1e-10)}")
