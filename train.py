import random
from pathlib import Path

import torch
import numpy as np
from einops import asnumpy
from tqdm.auto import tqdm

from metrics import Metrics


def train(model, trainloader, testloader, epochs, optimizer, criterion, device, scheduler=None, ck_dir="checkpoint", flooding=0):
    torch.manual_seed(777)
    np.random.seed(777)
    random.seed(777)

    train_loss_per_epoch = []
    test_loss_per_epoch = []
    dice_per_epoch = []
    vs_per_epoch = []
    hd_per_epoch = []
    asd_per_epoch = []
    for epoch in range(epochs):
        model.train()
        train_loss, test_loss = 0, 0
        for i, (img, mask) in enumerate(trainloader, start=1):
            optimizer.zero_grad()
            pred = model(img.to(device))
            if isinstance(pred, dict):
                pred = pred["out"]
            loss = criterion(pred, mask.to(device))
            loss = torch.abs(loss-flooding) + flooding
            loss.backward()
            train_loss = train_loss + 1/i*(asnumpy(loss)-train_loss)
            optimizer.step()
        if scheduler:
            scheduler.step()

        model.eval()
        avgDice, avgVS, avgAsd, avgHd = 0, 0, 0, 0
        for i, (img, mask) in enumerate(testloader, start=1):
            with torch.no_grad():
                pred = model(img.to(device))
                loss = criterion(pred, mask.to(device))
                test_loss = test_loss + 1/i*(asnumpy(loss)-test_loss)
                quality = Metrics.eval(pred, mask)
                dice, vs, asd, hd = \
                    quality['Dice'], quality['Volume Similarity'], \
                    quality['Average Surface Distance'], quality['Hausdorff Distance']
                avgDice = avgDice + 1/i*(dice - avgDice)
                avgVS = avgVS + 1/i*(vs - avgVS)
                avgAsd = avgAsd + 1/i*(asd - avgAsd)
                avgHd = avgHd + 1/i*(hd - avgHd)

        print(
            f"Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")
        path = Path(ck_dir)/f"check{epoch}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if epoch == 0:
            torch.save(model.state_dict(), path)
            print(f"Saving model to {path}")
        elif (best := test_loss) < (prev := min(test_loss_per_epoch)):
            torch.save(model.state_dict(), path)
            print(
                f"val_loss improved from {prev:.4f} to {best:.4f}, saving model to {path}")

        train_loss_per_epoch.append(train_loss)
        test_loss_per_epoch.append(test_loss)
        dice_per_epoch.append(avgDice)
        vs_per_epoch.append(avgVS)
        asd_per_epoch.append(avgAsd)
        hd_per_epoch.append(avgHd)

    return {
        'train_loss': train_loss_per_epoch,
        'test_loss': test_loss_per_epoch,
        'dice': dice_per_epoch,
        'vs': vs_per_epoch,
        'asd': asd_per_epoch,
        'hd': hd_per_epoch
    }
