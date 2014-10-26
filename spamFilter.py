import sys, os
import pprint
import re
import numpy
from operator import add
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

from pyspark import SparkContext

use_lexicon = True
use_hash = None
hashtable_size = 10000

def remPlural( word ):
    word = word.lower()
    if word.endswith('s'):
        return word[:-1]
    else:
        return word


def vector(tupleList,lexicon):
    #input: list of tuples [(word,count),(word,count)...]
    #return: vector representing word counts in lexicon [0,1,4,2,..]
    vector = [0]*(len(lexicon))
    for (x,y) in tupleList:
        #print ("x:",x, " y:",y, "lexicon(x): ",lexicon.index(x))
        try:
            idx  = lexicon.index(x)
        except:
            continue
        vector[idx] = y
    return vector


def hashVector(tupleList):
    #input: list of tuples [(word,count),(word,count)...]
    #return: hashTable
    hash_table = [0]*hashtable_size
    for (x,y)in tupleList:
        x = hash(x) % hashtable_size
        hash_table[x]+=1

    return hash_table



def wordCountPerFile(rdd):
    #input: rdd of (file,word) tuples
    #return: rdd of (file, [(word, count),(word, count)...]) tuples
    print ("##### BUILDING  ((file,word),1) tuples #####")
    rdd = rdd.map(lambda (x):((x[0],x[1]),  1))
    print('##### GETTING THE  ((file,word),n) WORDCOUNT PER (DOC, WORD) #####')
    rdd = rdd.reduceByKey(add)
    print('##### REARRANGE AS  (file, [(word, count)])  #####')
    rdd = rdd.map (lambda (a,b) : (a[0],[(a[1],b)]))
    print ('##### CONCATENATE (WORD,COUNT) LIST PER FILE AS  (file, [(word, count),(word, count)...])  #####')
    rdd = rdd.reduceByKey(add)
    return rdd


def vectorise(rdd,lexicon):
    #input: rdd of (file, [(word, count),(word, count)...]) tuples
    #return: rdd of (file,[vector]) tuples
    print('##### CREATE A DOC VECTOR AGAINST THE LEXICON   #####')
    rdd = rdd.map (lambda (f,wc): ( f,vector(wc,lexicon)))
    return rdd


def confusionMatrix (tupleList):
    mx = [0,0,0,0]
    for (x,y)in tupleList:
        mx[((x<<1) + y)] += 1
    return mx

def confusionDict (tupleList):
    mx =[0,0,0,0]
    for (x,y)in tupleList:
        mx[((x<<1) + y)] += 1
    dict = {'TN':mx[0],'FP':mx[1],'FN':mx[2],'TP':mx[3]}

    dict['TotalTrue']     = dict['TP'] + dict['FN']
    dict['TotalFalse']    = dict['TN'] + dict['FP']
    dict['TotalSamples']  = len(tupleList)
    dict['TotalPositive'] = dict['TP'] + dict['FP']
    dict['TotalNegative'] = dict['TN'] + dict['FN']
    dict['TotalCorrect']  = dict['TP'] + dict['TN']
    dict['TotalErrors']   = dict['FN'] + dict['FP']
    dict['Recall']        = float(dict['TP'])/dict['TotalTrue']
    dict['Precision']     = float(dict['TP'])/dict['TotalPositive']
    dict['Sensitivity']   = float(dict['TP'])/dict['TotalSamples']
    dict['Specificity']   = float(dict['TN'])/dict['TotalSamples']
    dict['ErrorRate']     = float(dict['TotalErrors'])/dict['TotalSamples']
    dict['Accuracy']      = float(dict['TotalCorrect'])/dict['TotalSamples'] # = 1 - Error Rate
    dict['Fmeasure']      = 2*float(dict['TP'])/(dict['TotalTrue']+dict['TotalPositive'])
    dict['Fmeasure2']     = 1/((1/dict['Precision']) + (1/dict['Recall']))
    dict['Fmeasure3']     = 2*dict['Precision']*dict['Recall']/(dict['Precision']+dict['Recall'])
    return dict

def printConfusionMatrix(confusionDict):
    print ("            condition\n" \
          "   test    T         F  \n"\
          "    T %6i    %6i    \n"\
          "    F %6i    %6i    \n"\
            % ( confusionDict['TP'], confusionDict['FP'],\
                confusionDict['FN'], confusionDict['TN']))

def printConfusionDict(confusionDict):
    print ("                  relevant       \n" \
          " retreived     yes       no  \n"\
          "   yes  %6i TP %6i FP   \n"\
          "   no   %6i FN %6i TN   \n"\
    \
          % ( confusionDict['TP'], confusionDict['FP'],\
                confusionDict['FN'], confusionDict['TN']))
    print ("                  truth       \n" \
           " prediction   spam       ham  \n"\
           "    spam %6i TP %6i FP   \n"\
           "     ham %6i FN %6i TN   \n"\
        "\n"\
        "classifier stats (classes spam and ham) \n"
        "total samples: %i \n"\
        "     accuracy: %.3f    TP+TN/total \n"
        "   error rate: %.3f    FN+FP/total \n"
        "\n"
        "class-specific stats (class spam)\n"
            "  sensitivity: %.3f   TP/total\n"\
        "  specificity: %.3f   FN/total\n"\
        "       recall: %.3f   TP/totalTrue TP/TP+TN \n"\
        "    precision: %.3f   TP/totalPos  TP/TP+FP\n"\

        "    f-measure: %.3f   2*TP/(totalTrue+totalPos) 2TP/(TP+TN+TP+FP)\n"\
        "   f-measure2: %.3f   1/(1/precision + 1/recall) \# this one looks wrong\n"\

        #http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html#10657
        "   f-measure3: %.3f   2 * precision * recall / (precision + recall)\n"\
    \
            % ( confusionDict['TP'], confusionDict['FP'],\
                confusionDict['FN'], confusionDict['TN'],\
                confusionDict['TotalSamples'],      \
                confusionDict['Accuracy'],\
                confusionDict['ErrorRate'],\
                confusionDict['Sensitivity'],\
                confusionDict['Specificity'],\
                confusionDict['Recall'],\
                confusionDict['Precision'],\

                confusionDict['Fmeasure'],\
                confusionDict['Fmeasure2'],\
                confusionDict['Fmeasure3'],\

        ))




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: spanfolder <folder> testfolder <folder> stoplist<file>"
        exit(-1)
    sc = SparkContext(appName="spamFilter")


    #1 Start by loading the files from part1 with wholeTextFiles.
    trainingSet = sc.wholeTextFiles(sys.argv[1], 1)
    testSet     = sc.wholeTextFiles(sys.argv[2],1)
    stopfile    = sc.textFile(sys.argv[3],1)
    stoplist    = stopfile.flatMap (lambda x: re.split('\W+',x)).collect()
    hashTableSize = 10000


    #2 (A)  Use the code from last time to generate the [(word,count), ...] list per file.
    #could use os.basename here

    print "\n##### BUILDING (file,word) tuples #####\n"

    train1 = trainingSet.flatMap(lambda (file,word):([(file[file.rfind("/")+1:],remPlural(word)) \
                                                   for word in re.split('\W+',word) \
                                                   if len(word)>0]))
    print("training set: ",train1.takeSample(True,4,0))

    test_1 =     testSet.flatMap(lambda (file,word):([(file[file.rfind("/")+1:],remPlural(word)) \
                                                   for word in re.split('\W+',word) \
                                                   if len(word)>0]))
    print("test set    : ",test_1.takeSample(True,4,0))

    train1.cache()
    test_1.cache()

    ##### BUILDING THE LEXICON #####
    if use_lexicon:
        print "\n\n  ##### BUILDING THE LEXICON #####\n"
        training_words = train1.map (lambda(f,x):x)
        print ("training_words: ",  training_words.count())
        training_lexicon = training_words.distinct()
        print ("training_lexicon: ",  training_lexicon.count())
        lexicon = training_lexicon.collect()


    ##### PROCESS THE RDDs #####
    ##### (file, [(word, count),(word, count)...]) tuples #####
    train5 = wordCountPerFile(train1)
    test_5 = wordCountPerFile(test_1)


    ##### CREATE A DOC VECTOR AGAINST THE LEXICON   #####



    #train6 = train5.map (lambda (f,x): ( f,vector(x,lexicon)))


    if use_hash:
        print('##### CREATE A DOC VECTOR OF HASHES  #####')
        hashtrain6 = train5.map(lambda(f,x):(f,hashVector(x)))
        #print ("hashtrain6 sample:", hashtrain6.takeSample(True,4,0))
        hashtest6  = test_5.map (lambda(f,x):(f,hashVector(x)))


    if use_lexicon:
        print('##### CREATE A DOC VECTOR AGAINST THE LEXICON   #####')
        train6=vectorise(train5,lexicon)
        #print ("traint6 sample:", train6.takeSample(True,4,0))
        test_6=vectorise(test_5,lexicon)

    # 3 Test whether the file is spam (i.e. the path contains spmsg) and replace the filename
    # by a 1 (spam) or 0 (ham) accordingly. Use map() to create an RDD of LabeledPoint objects.
    # See here http://spark.apache.org/docs/latest/mllib-naive-bayes.html for an example,
    # and here http://spark.apache.org/docs/latest/api/python/pyspark.mllib.regression.LabeledPoint-class.html
    # for the LabelledPoint documentation.

    print('#####      TEST WHETHER FILE IS SPAM       #####')
    ##### REPLACE FILENAME BY 1 (spam) 0 (ham) #####

    if use_lexicon:
        train7 = train6.map (lambda(f,x):(1 if 'spmsg' in f else 0, x))
        #print ("train7 sample",train7.take(2))
    if use_hash:
        hashtrain7 = hashtrain6.map (lambda(f,x):(1 if 'spmsg' in f else 0, x))
        #print ("hashtrain7 sample",hashtrain7.take(2))



    print('#####      MAP TO LABELLED POINTS      #####')
    if use_lexicon:
        train8 = train7.map (lambda (f,x):LabeledPoint(f,x))
    if use_hash:
        hashtrain8 = hashtrain7.map (lambda (f,x):LabeledPoint(f,x))


    #4 Use the created RDD of LabelledPoint objects to train the NaiveBayes and save
    # the model as a variable nbModel (again, use this example
    # http://spark.apache.org/ docs/latest/mllib-naive-bayes.html and here is the documentation
    # http://spark. apache.org/docs/latest/api/python/pyspark.mllib.regression.LabeledPoint-class. html).

    print('#####      TRAIN THE NAIVE BAYES      #####')
    if use_lexicon:
        nbModel = NaiveBayes.train(train8, 1.0)
    if use_hash:
        hashnbModel =  NaiveBayes.train(hashtrain8, 1.0)

    # 5 Use the files from /data/extra/spam/bare/part2 and prepare them like in task 3).
    # Then use nbModel to predict the label for each vector you have and compare it to the original,
    # to test the performance of your classifier.

    #          """
    print('#####      RUN THE PREDICTION      #####')
    if use_lexicon:
        test_7 = test_6.map(lambda (f,x):(1 if 'spmsg' in f else 0,int(nbModel.predict(x).item())))
        print ("prediction sample: ",test_7.takeSample(False,20,0))

    if use_hash:
        hashtest7 = hashtest6.map(lambda (f,x):(1 if 'spmsg' in f else 0,int(nbModel.predict(x).item())))
        print ("prediction sample: ",hashtest7.takeSample(False,20,0))


    if use_lexicon:
        print('#####      EVALUATION      #####')
        print "\n"
        printConfusionDict(confusionDict(test_7.collect()))

    if use_hash:
        print('#####    HASH  EVALUATION      #####')
        print "\n"
        printConfusionDict(confusionDict(hashtest7.collect()))



