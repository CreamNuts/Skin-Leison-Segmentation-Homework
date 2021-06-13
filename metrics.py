import numpy as np
import SimpleITK as sitk
from einops import asnumpy


class Metrics:
    class decorator:
        def __init__(self, func) -> None:
            self._func = func

        def __call__(self, *args):
            args = [np.round(asnumpy(arg.squeeze())).astype(np.int64)
                    for arg in args]
            return self._func(*args)

    @staticmethod
    def surface_distance(pred, target):
        labelPred = sitk.GetImageFromArray(pred, isVector=False)
        labelTrue = sitk.GetImageFromArray(target, isVector=False)
        hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
        try:
            hausdorffcomputer.Execute(labelTrue, labelPred)
        except:
            return 100, 100
        avgHausdorff = hausdorffcomputer.GetAverageHausdorffDistance()
        Hausdorff = hausdorffcomputer.GetHausdorffDistance()
        return avgHausdorff, Hausdorff

    @staticmethod
    def dice(pred, target):
        dice = (target & pred).sum()*2/(pred.sum()+target.sum())
        return dice

    @staticmethod
    def volume_similarity(pred, target):
        # labelPred = sitk.GetImageFromArray(pred, isVector=False)
        # labelTrue = sitk.GetImageFromArray(target, isVector=False)
        # dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
        # dicecomputer.Excute(labelTrue, labelPred)
        # vs = dicecomputer.GetVolumeSimilarity()
        vs = 1 - abs(target.sum()-pred.sum())/(pred.sum()+target.sum())
        return vs

    @staticmethod
    @decorator
    def eval(pred, target):
        dice = Metrics.dice(pred, target)
        vs = Metrics.volume_similarity(pred, target)
        asd, hd = Metrics.surface_distance(pred, target)
        return {
            'Dice': dice, 'Volume Similarity': vs,
            'Average Surface Distance': asd, 'Hausdorff Distance': hd
        }
