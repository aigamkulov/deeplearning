import math
import torch
import torch.nn as nn

torch.manual_seed(42)

# 23-topshiriqdagi kabi model
model = nn.Sequential(
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
X = torch.randn(128, 64)
y = torch.randint(0, 10, (128,))

def train_variant(use_softmax):
    torch.manual_seed(42)
    m = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    for _ in range(20):
        logits = m(X)
        pred = torch.softmax(logits, dim=1) if use_softmax else logits
        # Ikkinchi variantda Softmax CrossEntropyLoss ga noto'g'ri qo'shiladi.
        # Maqsad: CE qiymati va gradientlarni taqqoslash.
        loss = ce(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return ce(m(X) if not use_softmax else torch.softmax(m(X), dim=1), y).item()

correct = train_variant(False)
double_softmax = train_variant(True)

bound = math.log(math.e + 9) - 1
print("To'g'ri logits -> CrossEntropyLoss CE:", correct)
print("Logits -> Softmax -> CrossEntropyLoss CE:", double_softmax)
print("Ikkinchi variant uchun nazariy yuqori chegara:", bound)
print("ln(e+9)-1 =", bound)
