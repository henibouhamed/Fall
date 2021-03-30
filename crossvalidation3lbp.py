# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:58:54 2021

@author: ASUS
"""


import cv2
import numpy
#from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern
for a in range(1,723):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\722\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit=hist
      
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit=numpy.vstack([voit,hist])
import pandas
c=pandas.read_csv("D:\\img1\\722\\labels.csv", index_col=0,sep=',')
d=c.to_numpy() 
d=numpy.subtract(d,1)
t=numpy.hstack((voit,d))

for a in range(1,1302):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\1301\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit1=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit1=numpy.vstack([voit1,hist])
import pandas
c1=pandas.read_csv("D:\\img1\\1301\\labels.csv", index_col=0,sep=',')
d1=c1.to_numpy() 
d1=numpy.subtract(d1,1)
t1=numpy.hstack((voit1,d1))

for a in range(1,1791):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\1790\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit2=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit2=numpy.vstack([voit2,hist])
import pandas
c2=pandas.read_csv("D:\\img1\\1790\\labels.csv", index_col=0,sep=',')
d2=c2.to_numpy() 
d2=numpy.subtract(d2,1)
t2=numpy.hstack((voit2,d2))

for a in range(1,490):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\489\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit3=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit3=numpy.vstack([voit3,hist])
import pandas
c3=pandas.read_csv("D:\\img1\\489\\labels.csv", index_col=0,sep=',')
d3=c3.to_numpy() 
d3=numpy.subtract(d3,1)
t3=numpy.hstack((voit3,d3))

for a in range(1,571):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\569\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit4=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit4=numpy.vstack([voit4,hist])
import pandas
c4=pandas.read_csv("D:\\img1\\569\\labels.csv", index_col=0,sep=',')
d4=c4.to_numpy() 
d4=numpy.subtract(d4,1)
t4=numpy.hstack((voit4,d4))

for a in range(1,582):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\581\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit5=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit5=numpy.vstack([voit5,hist])
import pandas
c5=pandas.read_csv("D:\\img1\\581\\labels.csv", index_col=0,sep=',')
d5=c5.to_numpy() 
d5=numpy.subtract(d5,1)
t5=numpy.hstack((voit5,d5))

for a in range(1,732):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\731\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit6=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit6=numpy.vstack([voit6,hist])
import pandas
c6=pandas.read_csv("D:\\img1\\731\\labels.csv", index_col=0,sep=',')
d6=c6.to_numpy() 
d6=numpy.subtract(d6,1)
t6=numpy.hstack((voit6,d6))

for a in range(1,759):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\758\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit7=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit7=numpy.vstack([voit7,hist])
import pandas
c7=pandas.read_csv("D:\\img1\\758\\labels.csv", index_col=0,sep=',')
d7=c7.to_numpy() 
d7=numpy.subtract(d7,1)
t7=numpy.hstack((voit7,d7))

for a in range(1,808):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\807\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit8=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit8=numpy.vstack([voit8,hist])
import pandas
c8=pandas.read_csv("D:\\img1\\807\\labels.csv", index_col=0,sep=',')
d8=c8.to_numpy() 
d8=numpy.subtract(d8,1)
t8=numpy.hstack((voit8,d8))

for a in range(1,1220):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\1219\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit9=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit9=numpy.vstack([voit9,hist])
import pandas
c9=pandas.read_csv("D:\\img1\\1219\\labels.csv", index_col=0,sep=',')
d9=c9.to_numpy() 
d9=numpy.subtract(d9,1)
t9=numpy.hstack((voit9,d9))

for a in range(1,1261):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\1260\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit10=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit10=numpy.vstack([voit10,hist])
import pandas
c10=pandas.read_csv("D:\\img1\\1260\\labels.csv", index_col=0,sep=',')
d10=c10.to_numpy() 
d10=numpy.subtract(d10,1)
t10=numpy.hstack((voit10,d10))

for a in range(1,1379):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\1378\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit11=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit11=numpy.vstack([voit11,hist])
import pandas
c11=pandas.read_csv("D:\\img1\\1378\\labels.csv", index_col=0,sep=',')
d11=c11.to_numpy() 
d11=numpy.subtract(d11,1)
t11=numpy.hstack((voit11,d11))

for a in range(1,1844):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\1843\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit12=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit12=numpy.vstack([voit12,hist])
import pandas
c12=pandas.read_csv("D:\\img1\\1843\\labels.csv", index_col=0,sep=',')
d12=c12.to_numpy() 
d12=numpy.subtract(d12,1)
t12=numpy.hstack((voit12,d12))

for a in range(1,1955):
   b=str(a)
   b=b.zfill(4)
   st="D:\\img1\\1954\\rgb\\"+"rgb_"+b+".png"
   print(st)
   im = cv2.imread(st,0)
   lbp = local_binary_pattern(im, 8, 1,method='nri_uniform')
   (hist, _) = numpy.histogram(lbp.ravel(),bins=numpy.arange(60))
   hist = hist.astype("float")
   hist /= (hist.sum())
   
   lbp2 = local_binary_pattern(im, 8, 2,method='nri_uniform')
   (hist2, _) = numpy.histogram(lbp2.ravel(),bins=numpy.arange(60))
   hist2 = hist2.astype("float")
   hist2 /= (hist2.sum())
   
   lbp3 = local_binary_pattern(im, 8, 3,method='nri_uniform')
   (hist3, _) = numpy.histogram(lbp3.ravel(),bins=numpy.arange(60))
   hist3 = hist3.astype("float")
   hist3 /= (hist3.sum())
   if a == 1:
       hist=numpy.hstack([hist,hist2,hist3])
       voit13=hist
   else:
       hist=numpy.hstack([hist,hist2,hist3])
       voit13=numpy.vstack([voit13,hist])
import pandas
c13=pandas.read_csv("D:\\img1\\1954\\labels.csv", index_col=0,sep=',')
d13=c13.to_numpy() 
d13=numpy.subtract(d13,1)
t13=numpy.hstack((voit13,d13))

data=numpy.concatenate((t,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13),axis=0)



X = data[:,:177]
Y = data[:,177]
from keras.models import Sequential 
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
Y = np_utils.to_categorical(Y)
def baseline_model():
    model = Sequential()
    model.add(Dense(units=500, input_dim=177, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(500, activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(Dense(1000, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(units=6, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy','Recall','Precision'])
    #model.fit(X[train], Y[train], epochs=1000, batch_size=100)
    return model
estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=500)
kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))  
plt.plot(results.history['accuracy'])
plt.plot(results.history['Recall'])
plt.plot(results.history['Precision'])
plt.title('model accuracy')
#plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'recall','precision'])
plt.show() 
plt.savefig("D:\\all.jpeg")