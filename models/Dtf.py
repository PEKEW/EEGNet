from Utils import Config
import torch
import torch.nn as nn
from torch.nn import functional as F


class Embedding(nn.Module):
    def __init__(self, embde_dim=64):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(1, embde_dim)

    def forward(self, t):
        t = t.view(-1, 1).float()
        return torch.sin(self.linear(t))


class DTF(nn.Module):
    def __init__(self, args):
        super(DTF, self).__init__()
        self.args = args
        self.steps = args.diff_steps
        self.embed_dim = args.embed_dim
        self.embed = Embedding(args.embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128 + args.embed_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def get_noise_level(self, diff):
        noise_level = torch.norm(diff, p=2, dim=(1,2,3))
        noise_level = noise_level / noise_level.max()
        return noise_level

    def forward(self, x):
        noise_level = self.get_noise_level(x)
        features = self.encoder(x)
        
        noise_embed = self.embed(noise_level)
        noise_embed = noise_embed.view(-1, self.embed_dim, 1, 1)
        noise_embed = noise_embed(-1, -1, features.shape[2], features.shape[3])
        combined = torch.cat([features, noise_embed], dim=1)
        return self.decoder(combined)



# INFO: train diffusion
def train_step(model, optimizer, person, depersonal):
    optimizer.zero_grad()
    pred_clean = model(person)
    loss = F.mse_loss(pred_clean, depersonal)
    loss.backward()
    optimizer.step()
    return loss.item()

# INFO: de noise
@torch.no_grad()
def denoise(model, person):
    model.eval()
    return model(person)


def train(model, graph_list, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for person, deperson in graph_list:
            loss = train_step(model, optimizer, person, deperson)
            total_loss += loss
        avg_loss = total_loss / len(graph_list)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')



# TODO: important 预训练一个dft
def main():
    args = Config.Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DTF(args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

