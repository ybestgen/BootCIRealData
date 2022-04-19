
import pprint
import os
import collections
import numpy as np
import sys

from scipy.stats import pearsonr

from statistics import mean

sys.path.insert(1, '../')

from Fisher_Pitman_paired_test import fpreptest

np.random.seed(1234)

def stat(pred, rep):
  return(pearsonr(pred,rep)[0])
  
def main():
  #global text_file
  emotion=['anger','fear','joy','sadness']
  co=['Fu','A1','A2','A3']
  basemo=[0,1,2,3]
  mapping = {"Fu":"Full model ","A1":"Without FC ","A2":"Without CNN","A3":"Without LE "} 
  text_file = open("Res/ResTestPermAll.txt", "a")
  text_file_sum = open("Res/ResTestPermSum.txt", "a")
  n_iter=10000
  alpha=0.05
  ncv=20
  r12 = np.zeros(ncv)
  r13 = np.zeros(ncv)
  r23 = np.zeros(ncv)
  diff = np.zeros(ncv)
  p_value = np.zeros(ncv)
  for basemo1 in basemo:
    emotion1 = emotion[basemo1]
    text_file.write("\nTable %ld : %s Niter = %ld Alpha = %lf \n" % ((basemo1+1),emotion1,n_iter,alpha))
    text_file.write("Cond1         Cond2         Corr1  Corr2  Corr12 |    diff   p_value\n") 
    text_file_sum.write("\nTable %ld : %s Niter = %ld Alpha = %lf \n" % ((basemo1+1),emotion1,n_iter,alpha))
    text_file_sum.write(
      "%9s %11s   %11s   %5s  %5s  %5s  |   %5s  %5s  %5s  |   %5s  %5s  %5s  | %7s  %7s  %7s  | %7s  %7s  %7s | %4s\n" % 
      ("Emotion","Cond1","Cond2","Min1","Mea1","Max1","Min2","Mean2","Max2","Min12","Mea12","Max12",
      "MinDiff","MeanDif","MaxDiff","Minp","Meanp","Maxp","#sig"))
    for co1 in list(range(3)):   
      for co2 in list(range(co1+1,4)):
        nsigt=0
        nsigp=0
        for cv in range(ncv):
          which =  basemo1 + (cv*4)
          data1 = np.loadtxt('Data/CV'+co[co1]+'_'+emotion1+str(which)+'.csv' ,delimiter=',') 
          #data1 = (data1 < 0.5 )
          r12[cv]=stat(data1[0],data1[1])
          data2 = np.loadtxt('Data/CV'+co[co2]+'_'+emotion1+str(which)+'.csv' ,delimiter=',') 
          #data2 = (data2 < 0.5 )
          r13[cv]=stat(data2[0],data2[1])
          r23[cv]=stat(data1[0],data2[0])
          diff[cv]=r12[cv]-r13[cv]
          p_value[cv]=fpreptest(data1[0], data2[0], data1[1], stat, n_iter)
          if p_value[cv] <= 0.05:
           nsigp+=1                     
          text_file.write("%11s   %11s   %5.3lf  %5.3lf  %5.3lf  | %7.4lf  %7.4lf  \n" % 
            (mapping[co[co1]],mapping[co[co2]],r12[cv],r13[cv],r23[cv],diff[cv],p_value[cv]))   
          
        print(emotion1,mapping[co[co1]],mapping[co[co2]])
        text_file_sum.write(
          "%9s %11s   %11s   %5.3lf  %5.3lf  %5.3lf  |   %5.3lf  %5.3lf  %5.3lf  |   %5.3lf  %5.3lf  %5.3lf  | %7.4lf  %7.4lf  %7.4lf  | %7.4lf  %7.4lf  %7.4lf | %3ld\n" % 
          (emotion1,mapping[co[co1]],mapping[co[co2]],
          np.min(r12),np.mean(r12),np.max(r12),
          np.min(r13),np.mean(r13),np.max(r13),
          np.min(r23),np.mean(r23),np.max(r23),
          np.min(diff),np.mean(diff),np.max(diff),
          np.min(p_value),np.mean(p_value),np.max(p_value), nsigp ))  
  
main()

