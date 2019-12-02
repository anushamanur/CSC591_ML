#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold   
from nltk.stem.porter import PorterStemmer                                                                 
from copy import copy
from sklearn.manifold import TSNE
from nltk import word_tokenize
import matplotlib.pyplot as plt

from sklearn import svm
from metric_learn import LMNN
import sys
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
#from modshogun import RealFeatures,BinaryLabels,LMNN,MulticlassLabels
from metric_learn import ITML
from scipy.spatial.distance import pdist
from sklearn.model_selection import ShuffleSplit

randomState=13204
data=np.array([])
labels=np.array([])

print ('data.shape is ',data.shape)
print ('labels.shape is ',labels.shape)

print ('labels are ',labels)
        
class GraphBasedLearn:
    
    def __init__(self,X_train,y_train,x_test,y_test):
        self.x_train=copy(X_train)
        self.y_train=copy(y_train)
        self.x_test=copy(x_test)
        self.y_test=copy(y_test)
        self.y_train=copy(self.y_train.reshape(-1,))
        self.y_test=copy(self.y_test.reshape(-1,))
        
        #self.data=copy(np.vstack((self.x_train,self.x_test)))
        self.data=np.concatenate((self.x_train, x_test), axis=None)
        self.labels=copy(np.hstack((self.y_train,self.y_test)))
        print ('labels are ',self.labels)
        self.Vectorize()
  
        
        #print 'before PCA SVM Accuracy is ',self.compareWithSvm(self.trainVectors,self.testVectors)
        print ('now computing pca ',self.computePca())
        print ('after PCA svm accuracy is ',self.compareWithSvm(self.trainVectorsPCA,self.testVectorsPCA))
        #self.constructSimilartyMatrixITML()
        ks=[3,65]
        for k in ks:
            self.constructSimilartyMatrixLMNN(k)
        self.constructEucleadianGaussianKernel()
        self.constructEucleadianGaussianKernelNoPca()
    
     
    def constructEucleadianGaussianKernel(self):
        
        self.pwdis=pairwise_distances(self.allDataPCA)
        
        maccs=[]
        ks=[3]
        for k in ks:
            sigmas=[1]
            accs=[]
            for sigma in sigmas:
                self.pwdis=-1*self.pwdis/(2*sigma*sigma)
                self.pwdis=np.exp(self.pwdis)
                self.D=np.zeros(self.pwdis.shape)
                for i in range(0,self.pwdis.shape[0]):
                    l1=self.pwdis[i].tolist()
                    #print 'l1 is ',l1,'\n\n'
                    allnearestNeighbours=sorted(range(len(l1)),key=lambda i : l1[i])
                    #now set the all the weights except for k+1 to 0
                    self.pwdis[i,allnearestNeighbours[k:]]=0
                    self.D[i,i]=sum(self.pwdis[i])
                    
                    #here we make no trnasformation on the dataset, as this is simply the 
                print ('accuracy for constructEucleadianGaussianKernel with k=',k,' and sigma =',sigma,' is \n')
                accs.append(self.labelPropogation())
            maccs.append(np.mean(accs))
        
    def constructEucleadianGaussianKernelNoPca(self):
        
        self.pwdis=pairwise_distances(self.allVectors)
        
        maccs=[]
        ks=[3,65]
        for k in ks:
            sigmas=[1]
            accs=[]
            for sigma in sigmas:
                self.pwdis=-1*self.pwdis/(2*sigma*sigma)
                self.pwdis=np.exp(self.pwdis)
                self.D=np.zeros(self.pwdis.shape)
                for i in range(0,self.pwdis.shape[0]):
                    l1=self.pwdis[i].tolist()
                    #print 'l1 is ',l1,'\n\n'
                    allnearestNeighbours=sorted(range(len(l1)),key=lambda i : l1[i])
                    #now set the all the weights except for k+1 to 0
                    self.pwdis[i,allnearestNeighbours[k:]]=0
                    self.D[i,i]=sum(self.pwdis[i])
                    
                    #here we make no trnasformation on the dataset, as this is simply the 
                print ('accuracy for constructEucleadianGaussianKernel with k=',k,' and sigma =',sigma,' is \n')
                accs.append(self.labelPropogation())
            maccs.append(np.mean(accs))
    
    def convertToDenseMatrix(self):
        #transform the trainVectors to dense
        self.trainVectors=self.trainVectors.todense()
        temp=copy(np.zeros(self.trainVectors.shape))
        for i in range(0,self.trainVectors.shape[0]):
            for j in range(0,self.trainVectors.shape[1]):
                temp[i,j]=self.trainVectors[i,j]
        
        #transform the testVectors to dense
        self.trainVectors=copy(temp)
        self.testVectors=self.testVectors.todense()
        temp=copy(np.zeros(self.testVectors.shape))
        for i in range(0,self.testVectors.shape[0]):
            for j in range(0,self.testVectors.shape[1]):
                temp[i,j]=self.testVectors[i,j]    
        self.testVectors=copy(temp)
        
        self.allVectors=copy(np.vstack((self.trainVectors,self.testVectors)))
        
    def Vectorize(self):
        self.vectorizer = TfidfVectorizer(decode_error='replace',analyzer='word',stop_words='english',lowercase=True,tokenizer=StemmerTokenizer())
     
        self.x2=[]
        for doc in self.x_train:
            #print 'doc is ',doc[0],'\n\n'
            self.x2.append(doc[0])
        self.data2=[]
        for doc in self.data:   
            self.data2.append(doc[0])
        self.xtest2=[]
        for doc in self.x_test:  
            self.xtest2.append(doc[0])
        
        self.vectorizer.fit(self.x2)
        #print 'self.x2 is ',len(self.x2)
        self.trainVectors=self.vectorizer.transform(self.x2)
        print ('train vectors are ',self.trainVectors.shape)
      
        self.testVectors=self.vectorizer.transform(self.xtest2)
        self.allVectors=self.vectorizer.transform(self.data2)
        self.convertToDenseMatrix()
        print ('allVectors are ',self.allVectors.shape)
        
    
    
    def checkAccuracy(self,predicted,goldset):
        predicted=predicted.tolist()
        goldset=goldset.tolist()
        correct=0
        for i in range(0,len(predicted)):
            #print 'predicted is ',predicted[i],' goldset is ',goldset[i]
            if goldset[i]==predicted[i]:
                correct+=1
        
        return (float(correct)/len(predicted))*100
    
    def computePca(self):
        
        pca=PCA(n_components=7)
        pca.fit(self.trainVectors)
        self.trainVectorsPCA=copy(pca.transform(self.trainVectors))
        self.testVectorsPCA=copy(pca.transform(self.testVectors))
        print ('\ndata ',self.trainVectorsPCA,'\n')
        #print 'the explained variance is ',np.cumsum(pca.explained_variance_ratio_)
        self.allDataPCA=copy(np.vstack((self.trainVectorsPCA,self.testVectorsPCA)))
    
    def constructSimilartyMatrixLMNN(self,ks):
        
        
        print ('now doing LMNN for k= ',ks)
        self.y_train=self.y_train.reshape(-1,)
        lmnn=LMNN(k=ks, learn_rate=1e-7,max_iter=1000)
        lmnn.fit(self.trainVectorsPCA, self.y_train)
        self.L_lmnn = lmnn.transformer()
        name='lmnn/LMNN transformer matrix with dataset shape '+str(self.trainVectorsPCA.shape)
        np.save(name,self.L_lmnn)
        print ('L.shape is ',self.L_lmnn.shape,'\n\n')
        # Input data transformed to the metric space by X*L.T
        self.transformedTrainLMNN=copy(lmnn.transform(self.trainVectorsPCA))
        self.transformedTestLMNN=copy(lmnn.transform(self.testVectorsPCA))
        self.transformedAllLMNN=copy(lmnn.transform(self.allDataPCA)) #we compute the pairwise distance on this now 
        projectedDigits = TSNE(random_state=randomState).fit_transform(self.transformedAllLMNN)
        
        self.pwdis=copy(pairwise_distances(self.transformedAllLMNN,metric='euclidean'))
        self.D=np.zeros(self.pwdis.shape)
        for i in range(0,self.pwdis.shape[0]):
            l1=self.pwdis[i].tolist()
            #print 'l1 is ',l1,'\n\n'
            allnearestNeighbours=sorted(range(len(l1)),key=lambda i : l1[i])
            #now set the all the weights except for k+1 to 0
            self.pwdis[i,allnearestNeighbours[ks:]]=0
            self.D[i,i]=sum(self.pwdis[i])
        
        print ('accuracy for LMNN for k= ',ks,'\n')
        self.labelPropogation()
    
    def transformPairwiseDistanceToGaussian(self,sigma):
        
        self.pwdis=copy(-1*self.pwdis/(2*sigma*sigma))
        self.pwdis=copy(np.exp(self.pwdis))
        
            
    def labelPropogation(self):
        #Algorithm 11.1 Label propagation (Zhu and Ghahramani, 2002)
        self.y_test=self.y_test.reshape(-1,1)
        self.y_train=self.y_train.reshape(-1,1)
        
        self.yUnlabeled=np.zeros(self.y_test.shape)
        self.y_labeled=copy(self.y_train)
        
        self.ypred=copy(np.vstack((self.y_labeled,self.yUnlabeled)))
        
        #now to do the label propogation 
        
        for i in range(0,50):   
            self.ypred=np.dot(np.linalg.inv(self.D),np.dot(self.pwdis,self.ypred))
            #now need to relabel all the labeled points
            for i in range(0,self.y_labeled.shape[0]):
                self.ypred[i,0]=self.y_labeled[i,0]
            
            
            
        #now label propogation is complete
        numTrain=self.y_train.shape[0]
        self.predicted1=self.ypred[numTrain:,0]
        #now we need to rethreshold them to 1 and -1
        
        for i in range(self.predicted1.shape[0]):
            if self.predicted1[i]>0:
                self.predicted1[i]=1
            else:
                self.predicted1[i]=-1
        
        self.predicted1=self.predicted1.reshape(-1,)
        self.y_test=self.y_test.reshape(-1) 
        print ('self.predicted1 is ',self.predicted1.shape)
        print ('self.y_test is ',self.y_test.shape)       
        acc=self.checkAccuracy(self.predicted1,self.y_test)
        print ('the accuracy is ',acc)
        return acc
                    
            
         
    def compareWithSvm(self,datasetTrain,datasetTest):
        C=[0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
        print ('\n')
        print ('dataset shape is ',datasetTrain.shape)
        self.y_train=self.y_train.reshape(-1,)
        for c in C:
            self.Svm=svm.LinearSVC(C=c)
            self.Svm.fit(datasetTrain,self.y_train)
            labels=self.Svm.predict(datasetTest)
            print ('accuracy with c=',c,'  is  ',self.checkAccuracy(labels,self.y_test),'% ','\n')   
        


#for graph based reasoning , replace every 0 with -1

newLabels=[]
for label in labels:
    if label==1:
        newLabels.append(label)
    else:
        newLabels.append(-1)

#print 'newLabels are ',newLabels

newLabels=np.asarray(newLabels)
np.save('NewLabels',newLabels)


# In[ ]:


#Uncomment this call to use it
#ob1=GraphBasedLearn(X_train,y_train,X_test,Y_test)

