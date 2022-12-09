import numpy as np
import os
import subprocess
import time
import pandas as pd

# this is a test

def combineToGLXpath(baseDir, sesName, g , imec, ksdir)->str:
    dirg = os.path.join(baseDir, sesName + '_g' + g, sesName + '_g' + g + '_imec' + imec, ksdir)
    return dirg

def getGoodUnits(baseDir, sesName, g='0' , imec='0', ksdir='3', postphy=True):
    sdir = combineToGLXpath(baseDir, sesName, g, imec, ksdir)
    if postphy:
        file = os.path.join(sdir, 'cluster_group.tsv')
        df_clusgroup = pd.read_csv(file, sep='\t', header=0)

        df1 = df_clusgroup[df_clusgroup['group'] == 'good']['cluster_id'].tolist()
        return df1

def getGoodUnitsData(baseDir, sesName, g='0' , imec='0', ksdir='3', g_unitss=None):
    sdir = combineToGLXpath(baseDir, sesName, g, imec, ksdir)
    df_clusinfo = pd.read_csv(sdir + r'\cluster_info.tsv', sep='\t', header=0)
    if (gu==None):
        return df_clusinfo
    else:
        return df_clusinfo[df_clusinfo['cluster_id'].isin(g_unitss)]


def convertSapToSecs(baseDir, sesName, g , imec, ksdir):
    dirg = os.path.join(baseDir, sesName + '_g' + g, sesName + '_g' + g + '_imec' + imec, ksdir)
    timef = r'{}\spike_times.npy'.format(dirg)
    f = np.load(timef)
    print(f[0])
    ff = f/rate
    np.save(r'{}\spike_seconds.npy'.format(dirg), ff)

def extractDigitalEvents(baseDir, sesName, g):

    p = subprocess.run("CatGT -dir="+baseDir + " -run=" + sesName + " -g=" +g + " -t=0 -ni -XD=0,0,0 -XD=0,1,0 -XD=0,2,0 -XD=0,3,0 -XD=0,4,0 -XD=0,5,0 -XD=0,6,0 -XD=0,7,0", capture_output=True)


def extractSyncSignal(baseDir, sesName, g):

    p = subprocess.run("CatGT -dir="+baseDir + " -run=" + sesName + " -g=" +g + " -t=0 -prb_fld -prb=0 -ap -SY=0,384,6,500", capture_output=True)

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
    baseDir = r'f:\GLXData\ND11'
    sesName = 'ND11_Hab1'
    g = '0'
    imec = '0'
    ksdir = 'kilosort3'
    rate = float(29999.634454)

    goodu = countGoodUnits(baseDir, sesName, g, imec, ksdir)
    print('There are {} good unints in session {}'.format(goodu, sesName))
    convertSapToSecs(baseDir, sesName, g , imec, ksdir)
    extractDigitalEvents(baseDir, sesName, g)
    extractSyncSignal(baseDir, sesName, g)
    alignEventstoSync(baseDir, sesName, g , imec)
