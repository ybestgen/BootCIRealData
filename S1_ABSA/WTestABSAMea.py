
import pprint
import os
import collections
import numpy as np
import sys

from sklearn.metrics import f1_score,accuracy_score

sys.path.insert(1, '../')

from Fisher_Pitman_paired_test import fpreptest

np.random.seed(1234)

measure='Accuracy'; 
#measure='F1_score'; 
def stat(pred, rep):
  return(accuracy_score(rep, pred))
  #return(f1_score(rep, pred, average='macro'))

def main():
  system=['aen_bert','bert_spc','memnet','atae_lstm','td_lstm']
  text_file = open("Res/ResFPTestABSA.txt", "a")
  n_iter=10000
  alpha=0.05
  text_file.write("\nTable 1 : %s Niter = %ld Alpha = %lf \n" % (measure,n_iter,alpha))
  text_file.write("    Cond1         Cond2       M1    M2    |   diff   p_value\n") 
  for s1 in range(len(system)-1):
    data1 = np.loadtxt('Data/'+system[s1]+'.csv' ,delimiter=',') 
    m1=stat(data1[0],data1[1])
    for s2 in range(s1+1,len(system)):      
      data2 = np.loadtxt('Data/'+system[s2]+'.csv' ,delimiter=',')  
      m2=stat(data2[0],data2[1])
      diff=m1-m2
      p_value=fpreptest(data1[0], data2[0], data1[1], stat, n_iter)
      print(system[s1],system[s2],m1,m2,diff,p_value)
      text_file.write("%11s   %11s   %5.3lf  %5.3lf  | %7.4lf  %7.4lf\n" % 
        (system[s1],system[s2],m1,m2,diff,p_value))   
  
main()

