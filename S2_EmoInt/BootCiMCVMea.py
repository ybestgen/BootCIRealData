
import pprint
import os
import collections
import numpy as np

from statistics import mean

import sys

from scipy.stats import pearsonr

sys.path.insert(1, '../')

import paired_bootstrap_interval

np.random.seed(1234)

def stat(pred, rep):
  return(pearsonr(pred,rep)[0])


def main():
  emotion=['anger','fear','joy','sadness']
  co=['Fu','A1','A2','A3']
  basemo=[0,1,2,3]
  mapping = {"Fu":"Full_model ","A1":"Without_FC ","A2":"Without_CNN","A3":"Without_LE "} 
  text_file = open("Res/ResTestBootCIAll.txt", "a")
  text_file_sum = open("Res/ResTestBootCISum.txt", "a")
  text_file_sum.write(
    "%9s %11s   %11s   %5s  %5s  %5s  |   %5s  %5s  %5s  |   %5s  %5s  %5s  | %7s  %7s  %7s  | %7s  %7s  %7s  | %7s  %7s  %7s  | %4s\n" % 
    ("Emotion","Cond1","Cond2","Min1","Mea1","Max1","Min2","Mean2","Max2","Min12","Mea12","Max12",
    "Min-","Mean-","Max-","MinMe","MeanMe","MaxMe","Min+","Mean+","Max+","#sig"))
  n_iter=10000
  alpha=0.05
  ncv=20
  r12 = np.zeros(ncv)
  r13 = np.zeros(ncv)
  r23 = np.zeros(ncv)
  diff = np.zeros(ncv)
  BCaLow = np.zeros(ncv)
  BCaHigh = np.zeros(ncv)
  for basemo1 in basemo:
    emotion1 = emotion[basemo1]
    datai = np.loadtxt('Data/CVFu_'+emotion1+str(basemo1)+'.csv',delimiter=',')    
    text_file.write("\nTable %ld : %s N = %6d Niter = %ld Alpha = %lf \n" % ((basemo1+1),emotion1,len(datai[1]),n_iter,alpha))
    text_file.write("Cond1         Cond2         Corr1  Corr2  Corr12 | BCaLow    MeanDiff  BCaHigh\n")
    for co1 in list(range(3)):   
      for co2 in list(range(co1+1,4)):
        print(emotion1,mapping[co[co1]],mapping[co[co2]])
        nsig=0
        for cv in range(ncv):
          which =  basemo1 + (cv*4)
          data1 = np.loadtxt('Data/CV'+co[co1]+'_'+emotion1+str(which)+'.csv' ,delimiter=',') 
          r12[cv]=stat(data1[0],data1[1])
          data2 = np.loadtxt('Data/CV'+co[co2]+'_'+emotion1+str(which)+'.csv' ,delimiter=',') 
          r13[cv]=stat(data2[0],data2[1])
          r23[cv]=stat(data1[0],data2[0])
          diff[cv]=r12[cv]-r13[cv]
          bs = paired_bootstrap_interval.Bootstrap(data1[0],data2[0],data1[1], stat=stat,n_iter=n_iter)
          BCaLow[cv], BCaHigh[cv] = bs.get_confidence_interval(alpha, method='bias_corrected') #method='percentile')
          if BCaLow[cv]>0 or BCaHigh[cv]<0:
            nsig+=1
          text_file.write("%11s   %11s   %5.3lf  %5.3lf  %5.3lf  | [%7.4lf  %7.4lf  %7.4lf]  \n" % 
            (mapping[co[co1]],mapping[co[co2]],r12[cv],r13[cv],r23[cv],BCaLow[cv], diff[cv],BCaHigh[cv]))   
        text_file_sum.write(
            "%9s %11s   %11s   %5.3lf  %5.3lf  %5.3lf  |   %5.3lf  %5.3lf  %5.3lf  |   %5.3lf  %5.3lf  %5.3lf  | %7.4lf  %7.4lf  %7.4lf  | %7.4lf  %7.4lf  %7.4lf  | %7.4lf  %7.4lf  %7.4lf |  %3ld\n" % 
            (emotion1,mapping[co[co1]],mapping[co[co2]],
            np.min(r12),np.mean(r12),np.max(r12),
            np.min(r13),np.mean(r13),np.max(r13),
            np.min(r23),np.mean(r23),np.max(r23),
            np.min(BCaLow),np.mean(BCaLow),np.max(BCaLow), 
            np.min(diff),np.mean(diff),np.max(diff),
            np.min(BCaHigh),np.mean(BCaHigh),np.max(BCaHigh),
            nsig ))  

main()

