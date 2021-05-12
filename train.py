"""Trains IMLE on the MNIST dataset."""

import torch
from torch import optim, nn
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF
from tqdm import tqdm

from vgg_loss import vgg_loss

BATCH_SIZE = 50
BIG_BATCH_SIZE = 500
EPOCHS = 100
LATENT_SIZE = 32
PREFIX = 'mnist_imle_vgg'


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(0)

    tf = transforms.ToTensor()
    train_set = datasets.MNIST('data/mnist', download=True, transform=tf)
    train_dl = data.DataLoader(train_set, BATCH_SIZE, shuffle=True,
                               num_workers=1, pin_memory=True)

    model = nn.Sequential(
        nn.Linear(LATENT_SIZE, 16 * 7 * 7),
        nn.Unflatten(-1, (16, 7, 7)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        ConvBlock(16, 16),
        ConvBlock(16, 16),
        nn.Upsample(scale_factor=2),
        ConvBlock(16, 16),
        nn.Conv2d(16, 1, 3, padding=1),
        nn.Sigmoid(),
    ).to(device)
    print('Parameters:', sum(map(lambda x: x.numel(), model.parameters())))

    vgg = vgg_loss.VGGLoss(layer=8).to(device)

    def crit(fakes, reals):
        out = fakes.unsqueeze(0) - reals.unsqueeze(1)
        out = out.pow(2).mean([2, 3, 4])
        return out.min(1).values.mean()

    opt = optim.Adam(model.parameters(), lr=1e-3)

    def train():
        with tqdm(total=len(train_set), unit='samples', dynamic_ncols=True) as pbar:
            model.train()
            losses = []
            i = 0
            for x, _ in train_dl:
                i += 1
                x = x.to(device, non_blocking=True)
                xf = vgg.get_features(torch.cat([x, x, x], dim=1))
                z = torch.randn([BIG_BATCH_SIZE, LATENT_SIZE], device=device)
                gz = model(z)
                gzf = vgg.get_features(torch.cat([gz, gz, gz], dim=1))
                opt.zero_grad()
                loss = crit(gzf, xf)
                losses.append(loss.item())
                loss.backward()
                opt.step()
                pbar.update(len(x))
                if i % 50 == 0:
                    tqdm.write(f'{i * BATCH_SIZE} {sum(losses[-50:]) / 50:g}')
                if i % 500 == 0:
                    demo()
                    model.train()

    @torch.no_grad()
    @torch.random.fork_rng()
    def demo():
        model.eval()
        torch.manual_seed(0)
        z = torch.randn([100, LATENT_SIZE], device=device)
        grid = utils.make_grid(model(z), 10).cpu()
        TF.to_pil_image(grid).save('demo.png')
        tqdm.write('Wrote examples to demo.png.')

    def save():
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict()}, PREFIX + '.pth')
        print(f'Wrote checkpoint to {PREFIX}.pth.')

    try:
        for epoch in range(1, EPOCHS + 1):
            print('Epoch', epoch)
            train()
            demo()
            save()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
