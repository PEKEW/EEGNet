from Utils import Config
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, embde_dim=64, steps=100):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(1, embde_dim)
        self.steps = steps

    def forward(self, t):
        t = t.view(-1, 1).float() / self.steps
        return torch.sin(self.linear(t))


class DTF(nn.Module):
    def __init__(self, args):
        super(DTF, self).__init__()
        self.args = args
        self.embed = Embedding(args.embed_dim, args.diff_steps)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        def forward(self, x, t):
            # TODO: impoortant
            # return self.decoder(self.encoder(x))
            return x


# TODO: important noise comes from NMF
# TODO: important no time
def add_noise(x, t, steps):
    noise = torch.randn_like(x) * (t / steps).view(-1, 1, 1, 1)
    return x + noise


def train(model, optimizer, criterion, epochs=10, steps=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = (30, 30)
    for epoch in range(epochs):
        x = torch.randn((32, 1, *input_size), device=device)
        t = torch.randint(0, steps, (32,), device=device)
        noisy_x = add_noise(x, t, steps)
        preidcted_x = model(noisy_x, t)
        loss = criterion(preidcted_x, x)
        loss.barckward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")


args = Config.Args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DTF(args=args).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

# INFO: train diffusion
train(model, optimizer, criterion, epochs=args.epoch_diff, steps=args.diff_steps)

# INFO: de noise
model.eval()


def denoise(personal_graph):
    with torch.no_grad():
        depersonal_graph = model(personal_graph)
    return depersonal_graph
