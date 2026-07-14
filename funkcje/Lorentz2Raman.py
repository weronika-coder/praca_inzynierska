import numpy as np
import matplotlib.pyplot as plt 
import statistics as st
from lmfit import Parameters, Minimizer
from lmfit.models import LorentzianModel
import os
from joblib import Parallel, delayed
from pathlib import Path




#wczytanie pojedynczego spektrum
def importSpectrum(nazwa):
    data = np.loadtxt(nazwa, skiprows=1)
    rs = data[:, 0]
    intensity = data[:, 1]
    return rs, intensity




#wczytywanie mapy 
def importMap(nazwa):
    data = np.loadtxt(nazwa, skiprows=1)
    x = data[:,0]
    y = data[:,1]
    rs= data[:,2]
    intensity= data[:,3]

    xy = np.column_stack([x,y])
    changexy = np.where(np.any(np.diff(xy, axis=0) != 0, axis=1))[0] + 1 #nowe widmo zaczyna się indeks=i+1 gdzie zmiana, diff=wiersz(i+1)-wiersz(i)
    splits = np.split(np.arange(len(data)), changexy) #rozdziela indeksy dla widm
 
    rs_all = [rs[idx] for idx in splits]
    I_all  = [intensity[idx] for idx in splits]
    return rs_all, I_all
 

#dopasowanie lorentza do pojedynczego piku
def fitLorentz(args):
    rs, intensity, xmin, xmax = args
    mask = (rs >= xmin) & (rs <= xmax)
    if mask.sum() < 5:
        return (np.nan, np.nan, np.nan)
    rs_ = rs[mask]
    I_  = intensity[mask]

    idx_max = np.argmax(I_)
    center0 = rs_[idx_max] #pik tam gdzie intensywność największa
    amp0= I_[idx_max] * np.pi * 0.05  # przybliżenie amplitudy w Lorentzie? (tu amplituda to pole pod pikiem a nie intensywność!!)
    try:
        l1 = LorentzianModel(prefix='l1_')

        params = l1.make_params()
        params['l1_center'].set(center0, min=xmin, max=xmax)
        params['l1_amplitude'].set(amp0, min=0)
        params['l1_sigma'].set(0.03, min=0.001) #minimum, bo jak =0, to program ma problem czasem
        
        result = l1.fit(I_, params, x=rs_)
        center= result.params['l1_center'].value
        fwhm = result.params['l1_fwhm'].value
        amplitude = result.params['l1_height'].value  #wysokość piku
        return (center, fwhm, amplitude)
    except Exception:
        #jakby coś poszło nie tak zwraca nan
        return (np.nan, np.nan, np.nan)
 
 

#dopasowanie do wszytskich widm z mapy naraz!! jak robiłam po kolei to było strasznie wolne...
def lorentz2map(rs_all, I_all, xmin, xmax, n_jobs=-1):
    args = [(rs, I, xmin, xmax) for rs, I in zip(rs_all, I_all)]
    results = Parallel(n_jobs=n_jobs)(delayed(fitLorentz)(a) for a in args)
    centers, fwhms, amplitudes = zip(*results)
    return np.array(centers), np.array(fwhms), np.array(amplitudes)


#zapis danych do pliku w kolumnach 
def data2file(file_name, centers, fwhms, amplitudes):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write("x_c\tI\tFWHM\n")
        for c, a, fwhm in zip(centers, amplitudes, fwhms):
            f.write(f"{c:.6f}\t{a:.4f}\t{fwhm:.6f}\n")
 


def lorentz(x, I, x0, fwhm):
    return I * (fwhm/2)**2 / ((x - x0)**2 + (fwhm/2)**2)
