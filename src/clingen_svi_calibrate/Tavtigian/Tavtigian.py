import csv
import argparse
import os
import numpy as np
import math
import time
import bisect
from multiprocessing.pool import Pool

from ..LocalCalibration.LocalCalibration import LocalCalibration

from scipy.optimize import fsolve
import numpy as np

def evidence_to_plr(opvst, npsu, npm, npst, npvst):
    return opvst**( (npsu/8) + (npm/4) + (npst/2) + npvst )

def odds_to_postP(plr, priorP):
    postP = plr * priorP / ((plr - 1) * priorP + 1)
    return postP

def get_postP(c, prior, npsu, npm, npst, npvst):
    plr = evidence_to_plr(c, npsu, npm, npst, npvst)
    return odds_to_postP(plr, prior)

def get_postP_moderate(c, prior):
    return get_postP(c,prior,2,0,1,0) - 0.9

def get_tavtigian_c(prior):
    return fsolve(get_postP_moderate, 300 , args=(prior))

def get_tavtigian_thresholds(c, alpha):

    Post_p = np.zeros(5)
    Post_b = np.zeros(5)

    Post_p[0] = c ** (8 / 8) * alpha / ((c ** (8 / 8) - 1) * alpha + 1);
    Post_p[1] = c ** (4 / 8) * alpha / ((c ** (4 / 8) - 1) * alpha + 1);
    Post_p[2] = c ** (3 / 8) * alpha / ((c ** (3 / 8) - 1) * alpha + 1);
    Post_p[3] = c ** (2 / 8) * alpha / ((c ** (2 / 8) - 1) * alpha + 1);
    Post_p[4] = c ** (1 / 8) * alpha / ((c ** (1 / 8) - 1) * alpha + 1);

    Post_b[0] = (c ** (8 / 8)) * (1 - alpha) /(((c ** (8 / 8)) - 1) * (1 - alpha) + 1);
    Post_b[1] = (c ** (4 / 8)) * (1 - alpha) /(((c ** (4 / 8)) - 1) * (1 - alpha) + 1);
    Post_b[2] = (c ** (3 / 8)) * (1 - alpha) /(((c ** (3 / 8)) - 1) * (1 - alpha) + 1);
    Post_b[3] = (c ** (2 / 8)) * (1 - alpha) /(((c ** (2 / 8)) - 1) * (1 - alpha) + 1);
    Post_b[4] = (c ** (1 / 8)) * (1 - alpha) /(((c ** (1 / 8)) - 1) * (1 - alpha) + 1);

    #for j in range(4):
    #    Post_p[j] = c ** (1 / 2 ** (j)) * alpha / ((c ** (1 / 2 ** (j)) - 1) * alpha + 1);
    #    Post_b[j] = (c ** (1 / 2 ** (j))) * (1 - alpha) /(((c ** (1 / 2 ** (j))) - 1) * (1 - alpha) + 1);

    return Post_p, Post_b



class LocalCalibrateThresholdComputation:


    def __init__(self, alpha,c, reverse = None, windowclinvarpoints = 100, 
                 windowgnomadfraction = 0.03, gaussian_smoothing = True, pu_smoothing=False):

        self.alpha = alpha
        self.c = c
        self.reverse = reverse
        self.windowclinvarpoints = windowclinvarpoints
        self.windowgnomadfraction = windowgnomadfraction
        self.gaussian_smoothing = gaussian_smoothing
        self.pu_smoothing = pu_smoothing


    @staticmethod
    def initialize(x_, y_, g_, w_, thrs_, minpoints_, gft_, B_, gaussian_smoothing_, pu_smoothing_):
        global x
        global y
        global g
        global w
        global thrs
        global minpoints
        global gft
        global B
        global gaussian_smoothing
        global pu_smoothing
        x = x_
        y = y_
        g = g_
        w = w_
        thrs = thrs_
        minpoints  = minpoints_
        gft = gft_
        B = B_
        gaussian_smoothing = gaussian_smoothing_
        pu_smoothing = pu_smoothing_


    @staticmethod
    def get_both_bootstrapped_posteriors(seed):
        np.random.seed(seed)
        qx = np.random.randint(0,len(x),len(x))
        qg = np.random.randint(0,len(g),len(g))
        posteriors_p = LocalCalibration.get_both_local_posteriors(x[qx], y[qx], g[qg], thrs, w, minpoints, gft, gaussian_smoothing, pu_smoothing)
        return posteriors_p
        


    def get_both_bootstrapped_posteriors_parallel(self, x, y, g, B, alpha = None, thresholds = None):

        if alpha is None:
            alpha = self.alpha
        if self.pu_smoothing:
            assert g is not None
        w = ( (1-alpha)*((y==1).sum()) ) /  ( alpha*((y==0).sum()) )

        x,y,g = LocalCalibration.preprocess_data(x, y, g, self.reverse)

        if thresholds is None:
            xg = np.concatenate((x,g))
            thresholds = LocalCalibration.compute_thresholds(xg)

        ans = None
        with Pool(192,initializer = LocalCalibrateThresholdComputation.initialize, initargs=(x, y, g, w, thresholds, self.windowclinvarpoints, self.windowgnomadfraction, B, self.gaussian_smoothing, self.pu_smoothing),) as pool:
            items = [i for i in range(B)]
            ans = pool.map(LocalCalibrateThresholdComputation.get_both_bootstrapped_posteriors, items, 64)

        return thresholds, np.array(ans)



    @staticmethod
    def get_all_thresholds(posteriors, thrs, Post):
    
        thresh = np.zeros((posteriors.shape[0], len(Post)))

        for i in range(posteriors.shape[0]):
            posterior = posteriors[i]
            for j in range(len(Post)):
                idces = np.where(posterior < Post[j])[0]
                if len(idces) > 0 and idces[0] > 0:
                    thresh[i][j] = thrs[idces[0]-1]
                else:
                    thresh[i][j] = np.nan

        return thresh


    @staticmethod
    def get_discounted_thresholds(thresh, Post, B, discountonesided, tp):
        threshT = thresh.T
        DiscountedThreshold = np.zeros(len(Post))
        for j in range(len(Post)):
            threshCurrent = threshT[j]
            invalids = np.count_nonzero(np.isnan(threshCurrent))
            if invalids > discountonesided*B:
                DiscountedThreshold[j] = np.nan
            else:
                valids = threshCurrent[~np.isnan(threshCurrent)]
                valids = np.sort(valids)
                if tp == 'pathogenic':
                    valids = np.flip(valids)
                else:
                    assert tp == 'benign'
                    
                DiscountedThreshold[j] = valids[np.floor(discountonesided * B).astype(int) - invalids];

        return DiscountedThreshold


        

    @staticmethod
    def convertProbToPoint(prob, alpha, c):
        
        c0 = c
        c1 = np.sqrt(c0)
        c2 = np.sqrt(c1)
        c3 = np.sqrt(c2)

        
        op = prob * (1-alpha)/ ((1-prob)*alpha)
        opsu = c3

        points = np.log10(op)/np.log10(opsu)

        return points
