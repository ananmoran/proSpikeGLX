import numpy as np
import os
import subprocess
import time
from npyx.gl import get_units
from npyx import read_metadata
from npyx.spk_t import trn
import pandas as pd


# this is a test

def combineToGLXData(baseDir, sesName, g , imec)->str:
    dirg = os.path.join(baseDir, sesName + '_g' + g, sesName + '_g' + g + '_imec' + imec)
    return dirg

def combineToGLX_KS_path(baseDir, sesName, g , imec, ksdir)->str:
    dirg = os.path.join(combineToGLXData(baseDir, sesName, g , imec), ksdir)
    return dirg

def getGoodUnits(dp):
    gu = get_units(dp, quality='good')
    print('hello')
    return gu


    # if postphy:
    #     file = os.path.join(sdir, 'cluster_group.tsv')
    #     df_clusgroup = pd.read_csv(file, sep='\t', header=0)
    #
    #     df1 = df_clusgroup[df_clusgroup['group'] == 'good']['cluster_id'].tolist()
    #     return df1

def getUnitsData(dpks, units=None):
    # sdir = combineToGLX_KS_path(baseDir, sesName, g , imec, ksdir)
    df_clusinfo = pd.read_csv(dpks + r'\cluster_info.tsv', sep='\t', header=0)
    if (units is None):
        return df_clusinfo
    else:
        return df_clusinfo[df_clusinfo['cluster_id'].isin(units)]

def getSpikesByUnitID(dp, uid, tosecs=False):
    t = trn(dp, uid)  # gets all spikes from unit uid, in samples
    if tosecs:
        meta = read_metadata(dp)
        fs = meta['highpass']['sampling_rate']
        t = t/fs
    return t

def convertSampToSecs(dp, dpks):
    meta = read_metadata(dp)
    rate = meta['highpass']['sampling_rate']
    timef = os.path.join(dpks, 'spike_times.npy')
    trnInSamp = np.load(timef)
    trnInSecs = trnInSamp/rate
    np.save(r'{}\spike_seconds.npy'.format(dpks), trnInSecs)

def writeSTtoFile(units, tofile, betTimes=' ', betUnits=' '):
    thefile = open(tofile, 'a')
    for i in range(len(units)):
        arr = units[i]
        # thefile.write("%1.6f\n" % (arr))
        np.savetxt(thefile, arr, fmt='%.8f', delimiter=betTimes, newline=betUnits)
        thefile.write(';')
    thefile.close()

def extractDigitalEvents(baseDir, sesName, g):

    p = subprocess.run("CatGT -no_tshift -dir="+baseDir + " -run=" + sesName + " -g=" +g + " -t=0 -ni -XD=0,0,0 -XD=0,1,0 -XD=0,2,0 -XD=0,3,0 -XD=0,4,0 -XD=0,5,0 -XD=0,6,0 -XD=0,7,0", capture_output=True)


def extractSyncSignal(baseDir, sesName, g):

    p = subprocess.run("CatGT -no_tshift -dir="+baseDir + " -run=" + sesName + " -g=" +g + " -t=0 -prb_fld -prb=0 -ap -SY=0,384,6,500", capture_output=True)

def alignEventstoSync(baseDir, sesName, g , imec):

    command = "TPrime -syncperiod=1.000000 -tostream=" + os.path.join(baseDir, sesName + '_g' + g, sesName + '_g' + g + '_imec' + imec, sesName + "_g" +g + "_tcat.imec" + imec + ".ap.SY_384_6_500.txt") + \
    " -fromstream=2," + os.path.join(baseDir,sesName + "_g" + g, sesName + "_g" + g + "_tcat.nidq.XD_0_0_0.txt ") + \
    " -events=2," + "{}\{}_g{}\{}_g{}_tcat.nidq.XD_0_1_0.txt,{}\{}_g{}\{}_g{}_tcat.nidq.XD_0_1_0_corr.txt".format(baseDir,sesName,g, sesName, g, baseDir,sesName,g, sesName, g) +\
    " -events=2," + "{}\{}_g{}\{}_g{}_tcat.nidq.XD_0_2_0.txt,{}\{}_g{}\{}_g{}_tcat.nidq.XD_0_2_0_corr.txt".format(baseDir,sesName,g, sesName, g, baseDir,sesName,g, sesName, g) +\
    " -events=2," + "{}\{}_g{}\{}_g{}_tcat.nidq.XD_0_3_0.txt,{}\{}_g{}\{}_g{}_tcat.nidq.XD_0_3_0_corr.txt".format(baseDir,sesName,g, sesName, g, baseDir,sesName,g, sesName, g) +\
    " -events=2," + "{}\{}_g{}\{}_g{}_tcat.nidq.XD_0_4_0.txt,{}\{}_g{}\{}_g{}_tcat.nidq.XD_0_4_0_corr.txt".format(baseDir,sesName,g, sesName, g, baseDir,sesName,g, sesName, g)

    p = subprocess.run(command, capture_output=True)
    print("here2 " +str(p))





# -events=2,D:\GLXData\ND10\ND10_HAB3_1_g0\ND10_HAB3_1_g0_tcat.nidq.XD_0_1_0.txt,D:\GLXData\ND10\ND10_HAB3_1_g0\ND10_HAB3_1_g0_tcat.nidq.XD_0_1_0_corr.txt ^
# -events=2,D:\GLXData\ND10\ND10_HAB3_1_g0\ND10_HAB3_1_g0_tcat.nidq.XD_0_2_0.txt,D:\GLXData\ND10\ND10_HAB3_1_g0\ND10_HAB3_1_g0_tcat.nidq.XD_0_2_0_corr.txt ^
# -events=2,D:\GLXData\ND10\ND10_HAB3_1_g0\ND10_HAB3_1_g0_tcat.nidq.XD_0_3_0.txt,D:\GLXData\ND10\ND10_HAB3_1_g0\ND10_HAB3_1_g0_tcat.nidq.XD_0_3_0_corr.txt ^
# -events=2,D:\GLXData\ND10\ND10_HAB3_1_g0\ND10_HAB3_1_g0_tcat.nidq.XD_0_4_0.txt,D:\GLXData\ND10\ND10_HAB3_1_g0\ND10_HAB3_1_g0_tcat.nidq.XD_0_4_0_corr.txt

if __name__ == '__main__':
    baseDir = r'D:\GLXData\ND10'
    sesName = 'ND10_HAB1_bank0_fromE40up_extref_noapfilt_2'
    g = '0'
    imec = '0'
    # ksdir = 'kilosort3'
    # ksdir = 'Analyses'
    ksdir = 'kilosort3'

    # meta = npyx.read_metadata(combineToGLXData(baseDir, sesName, g , imec))
    #
    # rate = meta['highpass']['sampling_rate']
    # rate = float(29999.634454)
    #
    # dp = combineToGLXData(baseDir, sesName, g , imec)
    dpks = combineToGLX_KS_path(baseDir, sesName, g , imec, ksdir)
    goodu = getGoodUnits(dpks)
    writeSTtoFile(goodu, 'tofile', betTimes=' ', betUnits=';')

    # # print('There are {} good unints in session {}'.format(goodu, sesName))
    # convertSampToSecs(dp, dpks)
    extractDigitalEvents(baseDir, sesName, g)
    extractSyncSignal(baseDir, sesName, g)
    alignEventstoSync(baseDir, sesName, g , imec)
