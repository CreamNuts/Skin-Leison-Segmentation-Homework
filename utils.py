import matplotlib.pyplot as plt

from metrics import Metrics


def save_result(model, loader):
    avgDice, avgHausdorff, avgMean = 0, 0, 0
    for i, (img, target) in enumerate(loader):
        if i % 4 == 0:
            fig, axes = plt.subplots(ncols=4, figsize=(14, 8))
        pred = model(img.cuda())
        # cal score
        score = Metrics.eval(pred, target)
        dice, hd, msd = score['dice'], score['hd'], score['msd']
        avgDice, avgHausdorff, avgMean = avgDice+dice, avgHausdorff+hd, avgMean+msd

        # make image with pred mask and target mask
        pred = Morphology.preprocess(pred)
        pred_mask = np.array([np.zeros_like(pred), np.zeros_like(pred), pred])
        target_mask = np.array([np.zeros_like(pred), asnumpy(
            target.squeeze()), np.zeros_like(pred)])

        img = rearrange(img, '1 c h w -> h w c')
        pred_mask = rearrange(pred_mask, 'c h w -> h w c')
        target_mask = rearrange(target_mask, 'c h w -> h w c')

        pred_result = 0.7*img + 0.3*pred_mask
        target_result = 0.7*img + 0.3*target_mask
        result = np.concatenate([pred_result, target_result], axis=1)

        # visualize
        axes[i % 4].imshow(result)
        axes[i % 4].set_title(
            f"Dice: {dice:.2f}, Hausdorff: {hd:.2f}, MSD: {msd:.2f}")
        fig.tight_layout()
    avgDice, avgHausdorff, avgMean = avgDice / \
        (i+1), avgHausdorff/(i+1), avgMean/(i+1)
    return avgDice, avgHausdorff, avgMean
