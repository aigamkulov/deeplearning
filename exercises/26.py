import torch
import torch.nn as nn

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

def run(std, max_norm):
    torch.manual_seed(42)
    m = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, std)

    X = torch.randn(64, 64)
    y = torch.randint(0, 10, (64,))
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    opt.zero_grad()
    loss = criterion(m(X), y)
    loss.backward()
    before = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm)
    opt.step()
    return float(loss), float(before)

settings = [
    ("std=0.25, no clipping", 0.25, float("inf")),
    ("std=0.25, max_norm=1", 0.25, 1.0),
    ("std=sqrt(2/64), max_norm=1", (2/64)**0.5, 1.0),
]

for name, std, max_norm in settings:
    loss, norm_before = run(std, max_norm)
    print(f"{name:38s} loss={loss:.4f}  first_grad_norm={norm_before:.4f}")
