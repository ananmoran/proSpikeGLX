from postPhyPipeline import *
from npyx import *
import numpy as np
from matplotlib import pyplot as plt

def calcFR(uid, dpks, T, twin=5*60, overlap=0):

    spt = getSpikesByUnitID(dpks, uid, tosecs=True)
    spk = spt[(spt>= T[0])  & (spt <= T[1])]
    edges = range(T[0], T[1], twin)
    hist, ed = np.histogram(spt, bins=edges)
    return hist, ed


def calcFROE(uid, dpks, T=None, twin=5*60, overlap=0):

    spt = getSpikesByUnitID(dpks, uid, tosecs=True, OE=True)
    if T is None:
        T = [0, int(np.ceil(spt[-1]))]
    spt = spt[(spt>= T[0])  & (spt <= T[1])]
    edges = range(T[0], T[1], twin)
    hist, ed = np.histogram(spt, bins=edges)
    return hist, ed

# def plotSampFR():
#    # fill
#
# def plotMeanFR():
#     # fill

if __name__ == '__main__':

    baseDir = r'D:\GLXData\TB27'
    sesName = 'test'
    g = '0'
    imec = '0'
    ksdir = 'kilosort3'

    dp = combineToGLXData(baseDir, sesName, g, imec)
    dpks = combineToGLX_KS_path(baseDir, sesName, g, imec, ksdir)

    dp = r'F:\GLXData\TBR40\laser_2023-09-12_15-20-59\Record Node 114\experiment1\recording1'
    dpks = r'F:\GLXData\TBR40\laser_2023-09-12_15-20-59\Record Node 114\experiment1\recording1\continuous\Neuropix-PXI-110.0\kilosort3'

    gu = getGoodUnits(dpks)
    gudata = getUnitsData(dpks, gu)
    # meta = read_metadata(dp)
    endT = 1*3600; #meta['recording_length_seconds']

    cm = 1 / 2.54  # centimeters in inches
    savefigdir = os.path.join(dpks, 'FRfigs2')
    plt.figure(1)
    twin = 1*60
    frarrr = []
    for i, u in enumerate(gu):
        hist, edges  = calcFROE(u, dpks, twin=twin)
        frarrr.append(hist/twin)
        edges *= 3
        # if i%5 ==1:
        plt.plot(edges[0:-1]/3600-0.4, hist/twin)
        # plt.xticks(range(0, 1))
        plt.savefig(os.path.join(savefigdir, str(u)+'.jpg'))
        plt.savefig(os.path.join(savefigdir, str(u) + '.svg'))
        plt.clf()
    a=1
    gg = np.array(frarrr)
    # np.save('frarr.npy', gg)
    mgg = [np.mean(gg[i]) for i in range(len(gg))]
    for j in range(len(gg)):
        gg[j] = gg[j]/mgg[j]


    plt.figure(2, figsize=(35*cm, 12*cm))
    t = [x*600/3600 for x in range(gg.shape[1])]
    plt.plot(t, gg[::10].T)
    plt.xticks(range(0,24,8))
    plt.xlabel('Post-CTA time (H)')
    plt.ylabel('Normalized FR')
    plt.savefig(dpks + '\\FRunitsSample.svg')
    plt.show()


    plt.figure(3, figsize=(35*cm, 12*cm))
    M_vec = np.average(gg, axis=0)
    std = np.std(gg, axis=0)
    plt.plot(t, M_vec)
    lower_bound = M_vec - 1.96*std
    upper_bound = M_vec + 1.96 * std
    plt.fill_between(t, lower_bound, upper_bound, alpha=.3)
    # plt.xticks(range(0, len(M_vec), 6*3600/twin), range(0, int(len(M_vec)/6), 6))
    plt.xticks(range(0, 24, 8))
    plt.xlabel('Post-CTA time (H)')
    plt.ylabel('Mean normalized FR')
    plt.savefig(dpks + '\\MeanFR.svg')
    plt.show()





