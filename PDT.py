import pandas as pd
import itertools
from lib import DUMMY
from lib import DECTREE
from lib import datasets

from sklearn.pipeline import Pipeline

f = open("result/PDT-accuracy.csv", 'w')
f.write("dataset,accuracy\n")
accFold=0.0
for dataset in datasets.datasets:  
    for fold in range(0,10):
        trainX=pd.read_csv("/home/tomas/Dropbox/KonferenceAakce/JMLR15_EM_my/eval_final/folds_nodiscr/train/" +  dataset["filename"]+str(fold)+".csv",delimiter=",",index_col=False).drop(dataset["targetvariablename"],1)
        trainY=pd.read_csv("/home/tomas/Dropbox/KonferenceAakce/JMLR15_EM_my/eval_final/folds_nodiscr/train/" +  dataset["filename"]+str(fold)+".csv",delimiter=",",index_col=False, usecols=[dataset["targetvariablename"]])
        testX=pd.read_csv("/home/tomas/Dropbox/KonferenceAakce/JMLR15_EM_my/eval_final/folds_nodiscr/test/" +  dataset["filename"]+str(fold)+".csv",delimiter=",",index_col=False).drop(dataset["targetvariablename"],1)
        testY=pd.read_csv("/home/tomas/Dropbox/KonferenceAakce/JMLR15_EM_my/eval_final/folds_nodiscr/test/" +  dataset["filename"]+str(fold)+".csv",delimiter=",",index_col=False, usecols=[dataset["targetvariablename"]])
        clf = Pipeline([ ('dummyConv', DUMMY.ConvertCategoricalToDummies()),('autdectree', DECTREE.AutoTunedDecisionTreeClassifier())])
        clf.fit(trainX,trainY)
        prediction=clf.predict(trainY)
        p=pd.DataFrame(data=prediction).values==trainY.values
        p = list(itertools.chain(*p))
        acc=sum(p)/(len(p)*1.0)
        accFold=accFold+acc
        print(str(fold) + ":" + str(acc))
    f.write(dataset["filename"]+"," + str(accFold/10) + "\n")
    accFold=0.0
    f.flush()
        
