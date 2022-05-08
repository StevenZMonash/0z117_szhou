import gw_ml_priors
import sys
import multiprocessing
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import Interped
from joblib import Parallel, delayed
from numpy.random import uniform as unif
from tqdm.auto import tqdm

from gw_ml_priors.conversions import calc_a2

NUM_CORES = multiprocessing.cpu_count()
MCMC = int(1e5)

def get_a1_prior(xeff, q, mcmc_n=MCMC):
    a1s = np.linspace(0, 1, 500)
    da1 = a1s[1] - a1s[0]
    p_a1 = Parallel(n_jobs=NUM_CORES, verbose=0)(
        delayed(get_p_a1_given_xeff_q)(a1, xeff, q, mcmc_n)
        for a1 in a1s
    )
    p_a1 = p_a1 / np.sum(p_a1) / da1
    data = pd.DataFrame(dict(a1=a1s, p_a1=p_a1))
    a1 = data.a1.values
    p_a1 = norm_values(data.p_a1.values, a1)
    min_b, max_b = find_boundary(a1, p_a1)
    return Interped(
        xx=a1, yy=p_a1, minimum=min_b, maximum=max_b, name="a_1", latex_label=r"$a_1$"
    )


def get_p_a1_given_xeff_q(a1, xeff, q, n=int(1e4)):
    cos1, cos2 = unif(-1, 1, n), unif(-1, 1, n)
    a2 = calc_a2(xeff=xeff, q=q, cos1=cos1, cos2=cos2, a1=a1)
    integrand = a2_interpreter_function(a2)
    return np.mean(integrand)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_boundary_idx(x):
    """finds idx where data is non zero (assumes that there wont be gaps)"""
    non_z = np.nonzero(x)[0]
    return non_z[0], non_z[-1]


def norm_values(y, x):
    return y / np.trapz(y, x)


def find_boundary(x, y):
    b1, b2 = find_boundary_idx(y)
    vals = [x[b1], x[b2]]
    start, end = min(vals), max(vals)
    return start, end


def a2_interpreter_function(a2):
    return np.where(((0 < a2) & (a2 < 1)), 1, 0)

def main():
    q_id = int(sys.argv[1])
    q_range = np.linspace(0, 1, 100)   # q range 0 to 1
    xeff_range = np.linspace(-1, 1, 100)  # xeff range -1 to 1
    # remove zero
    q_range = np.delete(q_range, np.where(q_range == 0))
    q = q_range[q_id]
    # rm zero and -1
    xeff_range = np.delete(xeff_range, np.where(xeff_range == 0))
    xeff_range = np.delete(xeff_range, np.where(xeff_range == -1))
    outdir = 'out'
    out_png = 'out_png'
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(out_png, exist_ok=True)
    print(f"Starting data generation for q={q}")
    df = []
    counter = 0
    for xeff in tqdm(xeff_range, desc="xeff"):
        a1_prior = get_a1_prior(q=q, xeff=xeff)
        if counter%20 == 0:
            plt.plot(a1_prior.xx, a1_prior.yy)
            plt.xlabel("a1")
            plt.ylabel(f"p(a1|q={q:.2f},xeff={xeff:.2f})")
            plt.savefig(f"{out_png}/p_a1_given_q_xeff_{q:.2f}_{xeff:.2f}.png")
            plt.close()
        counter +=1
        # save to data
        df.append([q,xeff,a1_prior.xx,a1_prior.yy])
    df = pd.DataFrame(df)
    df.columns = ["q","xeff","a1","p_a1"]
    # save the dataframe to pickle
    with open(f"{outdir}/p_a1_given_q_xeff_qid_{q_id}.pkl", "wb") as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()