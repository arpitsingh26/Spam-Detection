import math
import random
import string
import csv
import numpy as np

random.seed(0)

def rand(x, y):
    return (y-x)*np.random.random_sample() + x

def sigmoid(x):
    return 1/(1+math.exp(-x))
    
def dsigmoid(x):
    return (1-x)*x
    
class NN:
    def __init__(self,nhidden):
        self.nhidden=nhidden        
        self.ai=np.ones(58)
        self.ai=np.matrix(self.ai)
        self.ah=np.ones(self.nhidden)
        self.ah=np.matrix(self.ah)
        self.ao=np.ones(1)
        self.ao=np.matrix(self.ao)
        self.wi=np.zeros(shape=(58,self.nhidden))
        self.wo=np.zeros(shape=(self.nhidden,1))
        for i in range(58):
            for j in range(self.nhidden):
                self.wi[i,j]=rand(-1,1)
        for i in range(self.nhidden):
            self.wo[i,0]=rand(-1,1)
       
    def update(self,inputs):
        vec=np.vectorize(sigmoid)
        for i in range(57):
            self.ai[0,i]=inputs[0,i]
        self.ah=np.matrix.dot(self.ai,self.wi)
        self.ah=vec(self.ah)
        self.ao=np.matrix.dot(self.ah,self.wo)
        self.ao[0,0]=sigmoid(self.ao[0,0])
        return self.ao[:]
    
    def backPropagate(self,inputs,N=0.5):
        output=self.ao[0,0]
        dsigm=dsigmoid(output)
        del1=np.zeros(1)
        del1[0]=(self.ao[0,0]-inputs[0,57])*dsigm
        del11=del1[0]
          
        del2=np.zeros(shape=(self.nhidden,1))
        del2=(del11*dsigm)*self.wo
           
          
        for i in range(self.nhidden):
            self.wo[i,0]=self.wo[i,0]-N*del11*self.ah[0,i]
                
        err=0.0
        self.wi=np.subtract(self.wi,N*np.matrix.dot(np.transpose(self.ai),np.transpose(del2)))
        err=err+0.5*((output-inputs[0,57])**2)
        return err
                
    def test(self,patterns):
        self.update(patterns)
        if self.ao[0,0]>0.5 :
            return 1
        else :
            return 0

    def train(self,patterns,iterations=1000,lr=0.5):
        for i in range(iterations):
            for p in patterns:
                err=0.0
                self.update(p)
                err=err+self.backPropagate(p,lr)
                
def demo():
    inp1=np.genfromtxt('Train.csv',delimiter=',')
    inp1=np.matrix(inp1)
    rows=inp1.shape[0]
    avg=np.zeros(57)
    for i in range(rows):
        for j in range(57):
            avg[j]=avg[j]+inp1[i,j]
    avg=avg/rows
    max1=np.zeros(57)
    for i in range(rows):
        for j in range(57):
            max1[j]=max(inp1[i,j],max1[j])
  
    for i in range(rows):
        for j in range(57):
            inp1[i,j]=(inp1[i,j])/max1[j]
            
       
        
      
    n=NN(57)
    n.train(inp1)
    
    out1=np.genfromtxt('TestX.csv',delimiter=',')
    out1=np.matrix(out1)
    rowstr=out1.shape[0]
    max1tr=np.zeros(57)
    for i in range(rowstr):
        for j in range(57):
            max1tr[j]=max(out1[i,j],max1tr[j])
    for i in range(rowstr):
        for j in range(57):
            out1[i,j]=(out1[i,j])/max1tr[j]
    output_file=open("TestY.csv",'w')
    writer=csv.writer(output_file, dialect='excel', lineterminator='\n')
    writer.writerow(['Id','Label',])
    i=0
    for j in out1:
        cont=n.test(j)
        writer.writerow([i,cont])
        i=i+1
            
if __name__ == '__main__':
    demo()
      
           
            
                
                    
                
        
                
                
        
        
                        
        
                
                
        
    


