import numpy as np
from os.path import join

pospath='../data/POS'
newpath='../data/newdata'

postrfile=join(pospath,'train')
postefile=join(pospath,'dev.out')
newtefile=join(newpath,'test.in.txt')

def loadtrain(trainfile):         #load POS training data file and process the initial data
    f=open(trainfile)
    s=f.read()
    f.close()
    lines=s.split('\n')
    sentences=[[]]
    labels=[[]]
    sentenceConstructIndex=0
    for line in lines:
        if line=='':
            sentences.append([])
            labels.append([])
            sentenceConstructIndex+=1
            continue
        word,label=line.split(' ')
        word=word.lower()           #all words are trained in lowercase, to reduce vacabulary size, reduce unseen event
        try:                        #turn all numbers into '0', for better training capability
            float(word[0])
            word='0'
        except:
            pass
        if 'http' in word:          #fix all words starting with 'http' to be just 'http'
            word='http'             #in result we can reduce unseen events, increase training capability
        if word[0]=='@' and len(word)>1:  #fix all words with '@' to be '@something'
            word='@something'
        if word[0]=='#' and len(word)>1:  #fix all words with '#' to be '#something'
            word='#something'
        sentences[sentenceConstructIndex].append(word)
        labels[sentenceConstructIndex].append(label)  
    for i in range(len(sentences)):           #delete empty sentences in the list
        if sentences[len(sentences)-i-1]==[]:
            del sentences[len(sentences)-i-1]
        if labels[len(labels)-i-1]==[]:
            del labels[len(labels)-i-1]
    return sentences,labels

def loadWords(file):                #only load POS training data file, no processing of data
    f=open(file)
    s=f.read()
    f.close()
    lines=s.split('\n')
    sentences=[[]]
    sentenceConstructIndex=0
    for line in lines:
        if line=='':
            sentences.append([])
            sentenceConstructIndex+=1
            continue
        word=line.split()[0].strip()
        sentences[sentenceConstructIndex].append(word)
    for i in range(len(sentences)):
        if sentences[len(sentences)-i-1]==[]:
            del sentences[len(sentences)-i-1]
    return sentences
    
def loadWords2(file):               #only load newest POS testing data file, no processing of data
    f=open(file)
    s=f.read()
    f.close()
    lines=s.split('\n')
    sentences=[[]]
    sentenceConstructIndex=0
    for line in lines:
        if line=='\r':
            sentences.append([])
            sentenceConstructIndex+=1
            continue
        word=line.strip()
        sentences[sentenceConstructIndex].append(word)
    for i in range(len(sentences)):
        if sentences[len(sentences)-i-1]==['']:
            del sentences[len(sentences)-i-1]
    return sentences

def loadWords3(file):               #load newest POS testing data file and process the initial data
    f=open(file)
    s=f.read()
    f.close()
    lines=s.split('\n')
    sentences=[[]]
    sentenceConstructIndex=0
    for line in lines:
        if line=='\r':
            sentences.append([])
            sentenceConstructIndex+=1
            continue
        word=line.strip().lower()
        try:
            float(word[0])
            word='0'
        except:
            pass
        if 'http' in word:
            word='http'
        if '@' in word and len(word)>1:
            word='@something'
        if '#' in word and len(word)>1:
            word='#something'
        sentences[sentenceConstructIndex].append(word)
    for i in range(len(sentences)):
        if sentences[len(sentences)-i-1]==['']:
            del sentences[len(sentences)-i-1]
    return sentences
    
def stripList(List):                    #make 2D list to 1D list, unfold the list
    ListPlain=[]
    for i in range(len(List)):
        for j in range(len(List[i])):
            ListPlain.append(List[i][j])
    return ListPlain

def createTransitionCountDict(trainsentences,trainlabels):   #function to count all required counts, and store them in different dictionaries
    transitCountDict=dict()                                  #dictionary for transition count, conut(y1,y2)
    yCountDict=dict()                                        #dictionary for the counts of different states
    emissionCountDict=dict()                                 #dictionary for emission counts
    for i in range(len(trainsentences)):
        for j in range(len(trainsentences[i])):
            if j < len(trainsentences[i])-1:
                try:
                    transitCountDict[trainlabels[i][j],trainlabels[i][j+1]]+=1
                except:
                    transitCountDict[trainlabels[i][j],trainlabels[i][j+1]]=1
            try:
                yCountDict[trainlabels[i][j]]+=1
            except:
                yCountDict[trainlabels[i][j]]=1
            try:
                emissionCountDict[trainlabels[i][j],trainsentences[i][j]]+=1
            except:
                emissionCountDict[trainlabels[i][j],trainsentences[i][j]]=1
        try:
            transitCountDict['START',trainlabels[i][0]]+=1          #initialize the base case
        except:
            transitCountDict['START',trainlabels[i][0]]=1           #initialize the base case
            
        try:
            transitCountDict[trainlabels[i][-1],'STOP']+=1          #initialize the base case
        except:
            transitCountDict[trainlabels[i][-1],'STOP']=1           #initialize the base case
            
    yCountDict['START']=len(trainsentences)
        
    return yCountDict,transitCountDict,emissionCountDict

def estTransitionP(yCountDict,transitCountDict,y1,y2):     #function to calculate transition probability based on count dictionary
    try:                                                   #using Laplace Smoothing
        transitPr=(transitCountDict[y1,y2]+1)/float((yCountDict[y1])+len(posStateList))
    except:
        #transitPr=np.exp(-100)
        transitPr=1/float(yCountDict[y1]+len(posStateList))
    return transitPr

def estEmissionPara(trainsentencesPlain,yCountDict,emissionCountDict,y,x):
    emissionPr=0
    if x not in trainsentencesPlain:
        #emissionPr=1/float(yCountDict[y]+1)
        emissionPr=float(yCountDict[y]+1)
    else:
        try:                    #using Absolute Discount
            emissionPr=emissionCountDict[y,x]/float(yCountDict[y])-smoothingPlist[posStateList.index(y)]
        except:
            #emissionPr=absoluteDiscListV[posStateList.index(y)]*smoothingPlist[posStateList.index(y)]/(len(postrainsentencesPlain)-absoluteDiscListV[posStateList.index(y)])         
            emissionPr=np.exp(-100)
    return emissionPr
    
def probabilityDict(yCountDict,transitCountDict,emissionCountDict,trainsentencesPlain,testsentences,state):
    emissionPrDict=dict()           #calculate all emissions, and store the probability in a dictionary
    transitPrDict=dict()             #calculate all transition, and store the probability in a dictionary
    for i in range(len(testsentences)):
        for j in testsentences[i]:
            for s in state:
                emissionPrDict[s,j]=estEmissionPara(trainsentencesPlain,yCountDict,emissionCountDict,s,j)
    for m in state:
        transitPrDict['START',m]=estTransitionP(yCountDict,transitCountDict,'START',m)
        transitPrDict[m,'STOP']=estTransitionP(yCountDict,transitCountDict,m,'STOP')
        for n in state:
            transitPrDict[m,n]=estTransitionP(yCountDict,transitCountDict,m,n)
    return transitPrDict,emissionPrDict
    
def Viterbi(transitPrDict,emissionPrDict,trainsentencesPlain,testsentences,state):
    testlabels=[]
    for i in range(len(testsentences)):
        testlabels.append([])
        piList=dict()
        for n in state:         
            piList['1',n]=np.log(transitPrDict['START',n])+np.log(emissionPrDict[n,testsentences[i][0]])

        for j in range(len(testsentences[i])-1):     #forward algorithm
            for k in state:
                temp=[piList[str(j+1),m]+np.log(transitPrDict[m,k])+np.log(emissionPrDict[k,testsentences[i][j+1]]) for m in state]
                piList[str(j+2),k]=max(temp)
                
        predEndLabel = state[0]                     #backward algorithm
        bestEndBackPara=piList[str(len(testsentences[i])),state[0]]+np.log(transitPrDict[state[0],'STOP'])
        for o in state[1::]:
            backEndPara=piList[str(len(testsentences[i])),o]+np.log(transitPrDict[o,'STOP'])
            if backEndPara>bestEndBackPara:
                bestEndBackPara=backEndPara
                predEndLabel=o
        testlabels[i].insert(0,predEndLabel)
        for p in range(len(testsentences[i])):
            if p==len(testsentences[i])-1:
                continue    
            predLabel = state[0]
            bestBackPara=piList[str(len(testsentences[i])-p-1),state[0]]+np.log(transitPrDict[state[0],testlabels[i][0]])
            for q in state[1::]:
                backPara=piList[str(len(testsentences[i])-p-1),q]+np.log(transitPrDict[q,testlabels[i][0]])
                if backPara>bestBackPara:
                    bestBackPara=backPara
                    predLabel=q
            testlabels[i].insert(0,predLabel)
    return testlabels

def outputAnalyse(trainsentencePlain,trainlabelsPlain,testsentence,postag): #function to correct obvious mistakes in output
    for i in range(len(testsentence)):
        for j in range(len(testsentence[i])):
            if testsentence[i][j] not in trainsentencePlain:
                if testsentence[i][j][-3:]=='ing':
                    for k in postrainsentencesPlain:
                        if testsentence[i][j][:-3] in k:
                            postag[i][j] = 'VBG'
                            continue
                elif testsentence[i][j][-2:]=='in':
                    for k in postrainsentencesPlain:
                        if testsentence[i][j][:-2] in k:
                            postag[i][j] = 'VBG'
                            continue
                elif testsentence[i][j][-2:]=='ed' and len(testsentence[i][j])>=5:
                    postag[i][j]='VBD'
                    for k in postrainsentencesPlain:
                        if testsentence[i][j][:-2] in k:
                            postag[i][j]=trainlabelsPlain[trainsentencePlain.index(k)]
                            continue  
                elif testsentence[i][j][-1]=='s':
                    for k in postrainsentencesPlain:
                        if testsentence[i][j][:-2] in k:
                            postag[i][j]=trainlabelsPlain[trainsentencePlain.index(k)]
                            continue

            if postag[i][j]=='VB' and j!=0:
                if postag[i][j-1]=='PRP':
                    postag[i][j]='VBP'
            elif postag[i][j]=='VBP' and j!=0:
                if postag[i][j-1]!='PRP':
                    postag[i][j]='VB'
            if postag[i][j][0:2]=='VB':
                if 'ed' in testsentence[i][j][-2::] and postag[i][j] !='VBN' and testsentence[i][j][-3:]!='eed':
                    postag[i][j]='VBD'
                elif (testsentence[i][j][-3::]=='ing' or testsentence[i][j][-2::]=='in') and testsentence[i][j][-3]!='a':
                    postag[i][j]='VBG'
                elif testsentence[i][j][-2::]=='en':
                    postag[i][j]='VBN'
                elif (testsentence[i][j][-1]=='s' or testsentence[i][j][-1::]=='z') and testsentence[i][j][-2]!='a' and testsentence[i][j][-2]!='s':
                    postag[i][j]='VBZ'
            if postag[i][j]=='VBG' and j!=0:
                if postag[i][j-1]=='PRP$':
                    postag[i][j]='NN'
            
            if (postag[i][j]=='NN' or postag[i][j]=='NNS') and j!=0 and postag[i][j-1]!='.' and postag[i][j-1]!='!' and postag[i][j-1]!=':':
                if postestwords[i][j][0].isupper() and postestwords[i][j][1:].islower:
                    postag[i][j]='NNP'
            
            elif (postag[i][j]=='NNP' or postag[i][j]=='NNPS') and j!=0:
                if postestwords[i][j].islower():
                    postag[i][j]='NN'
            
            if postag[i][j]=='TO' and j!=len(testsentence)-1:
                if postag[i][j+1]=='NN':
                    postag[i][j+1]='VB'       
            if postag[i][j]=='NN':
                if testsentence[i][j][-2:]=='ly':
                    postag[i][j]='RB'
                if testsentence[i][j][-1]=='s' and testsentence[i][j][-2]!='s':
                    postag[i][j]='NNS'
                
    return postag
    
def calcaccuracy(taglabelsPlain,testlabelsPlain):
    count=0
    for i in range(len(testlabelsPlain)):
        if taglabelsPlain[i]==testlabelsPlain[i]:
            count+=1
    accuracy=float(count)/len(taglabelsPlain)
    return accuracy
    
postrainsentences,postrainlabels=loadtrain(postrfile)
#postestsentences,postestlabels=loadtrain(postefile)
#postestwords=loadWords(postefile)

postestwords=loadWords2(newtefile)
postestsentences=loadWords3(newtefile)

postrainlabelsPlain=[]
postrainsentencesPlain=[]
postestsentencesPlain=[]
postrainlabelsPlain=stripList(postrainlabels)
postrainsentencesPlain=stripList(postrainsentences)
postestsentencesPlain=stripList(postestsentences)
#postestlabelsPlain=stripList(postestlabels)
posStateList=list(set(postrainlabelsPlain))
posyCountDict,postransitCountDict,posemissionCountDict=createTransitionCountDict(postrainsentences,postrainlabels)


##calculate the p-value for absolute discount and laplace smoothing
absoluteDiscListV=[]
smoothingPlist=[]
for i in range(len(posStateList)):
    absoluteDiscListV.append(0)
    for j in range(len(postrainsentencesPlain)):
        if postrainlabelsPlain[j]==posStateList[i]:
            absoluteDiscListV[i]+=1
    smoothingPlist.append(1.0/(posyCountDict[posStateList[i]]+absoluteDiscListV[i]))

for i in range(len(postestsentences)):
    for j in range(len(postestsentences[i])):
        if postestsentences[i][j] not in postrainsentencesPlain:
            if postestsentences[i][j][-3:]=='ful':
                postestsentences[i][j]='beautiful'
            if postestsentences[i][j][-2:]=='ly':
                postestsentences[i][j]='totally'
            if postestsentences[i][j][0]=='.':
                postestsentences[i][j]='.'
    
    
    
    
postransitPrDict,posemissionPrDict=probabilityDict(posyCountDict,postransitCountDict,posemissionCountDict,postrainsentencesPlain,postestsentences,posStateList)
postag2=Viterbi(postransitPrDict,posemissionPrDict,postrainsentencesPlain,postestsentences,posStateList)
postag2=outputAnalyse(postrainsentencesPlain,postrainlabelsPlain,postestsentences,postag2)
postag2Plain=stripList(postag2)
#posAccuracy=calcaccuracy(postag2Plain,postestlabelsPlain)
#print posAccuracy
f=open("test.out",'w')
for i in range(len(postag2)):
    for j in range(len(postag2[i])):
        f.write(postestwords[i][j]+' '+postag2[i][j]+'\n')
    f.write('\n')
f.close()
