import sys, os
import pprint
import re
import numpy
from operator import add
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from datetime import datetime, time, timedelta

from pyspark import SparkContext

use_lexicon = 0
use_hash = 1
#use_hash_signing = 1
use_log = 0
#hashtable_size = 6001

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

def sign_hash(x):
    return 1 if len(x)%2 == 1 else -1

def hashVectorSigned(tupleList):
    #input: list of tuples [(word,count),(word,count)...]
    #return: hashTable
     hash_table = [0]*hashtable_size
     for (word,count)in tupleList:
         x = hash(word) % hashtable_size
         hash_table[x] = hash_table[x] + sign_hash(word) * count
     return map(lambda x:abs(x),hash_table)

def hashVectorUnsigned(tupleList):
    #input: list of tuples [(word,count),(word,count)...]
    #return: hashTable
     hash_table = [0]*hashtable_size
     for (word,count)in tupleList:
         x = hash(word) % hashtable_size
         hash_table[x] = hash_table[x] + count
     return map(lambda x:abs(x),hash_table)

def hashVector(tupleList,use_hash_signing):
    return hashVectorSigned(tupleList) if use_hash_signing else hashVectorUnsigned(tupleList)

def wordCountPerFile(rdd):
    #input: rdd of (file,word) tuples
    #return: rdd of (file, [(word, count),(word, count)...]) tuples
    #print ("##### BUILDING  ((file,word),1) tuples #####")
    rdd = rdd.map(lambda (x):((x[0],x[1]),  1))
    #print('##### GETTING THE  ((file,word),n) WORDCOUNT PER (DOC, WORD) #####')
    rdd = rdd.reduceByKey(add)
    #print('##### REARRANGE AS  (file, [(word, count)])  #####')
    rdd = rdd.map (lambda (a,b) : (a[0],[(a[1],b)]))
    #print ('##### CONCATENATE (WORD,COUNT) LIST PER FILE AS  (file, [(word, count),(word, count)...])  #####')
    rdd = rdd.reduceByKey(add)
    return rdd


def vectorise(rdd,lexicon):
    #input: rdd of (file, [(word, count),(word, count)...]) tuples
    #return: rdd of (file,[vector]) tuples
    #print('##### CREATE A DOC VECTOR AGAINST THE LEXICON   #####')
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
    dict['Recall']        = float(dict['TP'])/dict['TotalTrue'] if dict['TotalTrue']>0 else 0
    dict['Precision']     = float(dict['TP'])/dict['TotalPositive'] if dict['TotalPositive']>0 else 0
    dict['Sensitivity']   = float(dict['TP'])/dict['TotalSamples'] if dict['TotalSamples']>0 else 0
    dict['Specificity']   = float(dict['TN'])/dict['TotalSamples'] if dict['TotalSamples']>0 else 0
    dict['ErrorRate']     = float(dict['TotalErrors'])/dict['TotalSamples'] if dict['TotalSamples']>0 else 0
    dict['Accuracy']      = float(dict['TotalCorrect'])/dict['TotalSamples'] if dict['TotalSamples']>0 else 0
    dict['Fmeasure']      = 2*float(dict['TP'])/(dict['TotalTrue']+dict['TotalPositive']) \
                                                    if (dict['TotalTrue']+dict['TotalPositive']>0) else 0
    dict['Fmeasure2']     = 1/((1/dict['Precision']) + (1/dict['Recall'])) \
                                                    if dict['Precision']>0 and dict['Recall']>0 else 0
    dict['Fmeasure3']     = 2*dict['Precision']*dict['Recall']/(dict['Precision']+dict['Recall']) \
                                                    if (dict['Precision']+dict['Recall']>0) else 0
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
          "   yes  %6i TP %6i FP %6i  \n"\
          "   no   %6i FN %6i TN %6i   \n"\
          "        %6i    %6i    %6i   \n"\
          % ( confusionDict['TP'], confusionDict['FP'],confusionDict['TotalPositive'],\
                confusionDict['FN'], confusionDict['TN'],confusionDict['TotalNegative'],\
                confusionDict['TotalTrue'], confusionDict['TotalFalse'], confusionDict['TotalSamples']))
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


def logTimeInterval(start_time):
    if (use_log):
        timedelta = (datetime.now() - start_time)
        print ("log:",timedelta.seconds)

def logTimeIntervalWithMsg(start_time, msg):
    if (use_log):
        message = msg if msg else ""
        timedelta = (datetime.now() - start_time)
        print ("log:", timedelta.seconds,message)

def logPrint(string):
    if use_log:
        print (string)
    else:
        print '.',


if __name__ == "__main__":
    start_time = datetime.now()

    if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: spanfolder <folder> testfolder <folder> stoplist<file>"
        exit(-1)
    sc = SparkContext(appName="spamFilter")
    logTimeIntervalWithMsg(start_time,"spark initialised, resetting timer")
    start_time = datetime.now()


    #1 Start by loading the files from part1 with wholeTextFiles.
    trainingSet = sc.wholeTextFiles(sys.argv[1], 1)
    logTimeInterval(start_time)
    testSet     = sc.wholeTextFiles(sys.argv[2],1)
    logTimeInterval(start_time)
    stopfile    = sc.textFile(sys.argv[3],1)
    stoplist    = stopfile.flatMap (lambda x: re.split('\W+',x)).collect()
    logTimeInterval(start_time)


    #2 (A)  Use the code from last time to generate the [(word,count), ...] list per file.
    #could use os.basename here

    logPrint("\n##### BUILDING (file,word) tuples #####\n")

    train1 = trainingSet.flatMap(lambda (file,word):([(file[file.rfind("/")+1:],remPlural(word)) \
                                                   for word in re.split('\W+',word) \
                                                   if len(word)>0]))
    logPrint("training set: " % train1.takeSample(True,4,0))

    test_1 =     testSet.flatMap(lambda (file,word):([(file[file.rfind("/")+1:],remPlural(word)) \
                                                   for word in re.split('\W+',word) \
                                                   if len(word)>0]))

    logPrint("test set    : {}".format(test_1.takeSample(True,4,0)))

    train1.cache()
    test_1.cache()
    logTimeInterval(start_time)

    ##### BUILDING THE LEXICON #####
    if use_lexicon:
        logPrint("\n\n  ##### BUILDING THE LEXICON #####\n")
        training_words = train1.map (lambda(f,x):x)
        logPrint("training_words: %i" %  training_words.count())
        training_lexicon = training_words.distinct()
        logPrint("training_lexicon: %i" % training_lexicon.count())
        lexicon = training_lexicon.collect()


    ##### PROCESS THE RDDs #####
    ##### (file, [(word, count),(word, count)...]) tuples #####
    train5 = wordCountPerFile(train1)
    test_5 = wordCountPerFile(test_1)


    ##### CREATE A DOC VECTOR AGAINST THE LEXICON   #####


for hashtable_size in range (1000,21000,1000):
    for use_hash_signing in range (0,1,1):


        #train6 = train5.map (lambda (f,x): ( f,vector(x,lexicon)))
        logTimeInterval(start_time)

        if use_hash:
            #logPrint('##### CREATE A DOC VECTOR OF HASHES  #####')
            hashtrain6 = train5.map(lambda(f,x):(f,hashVector(x,use_hash_signing)))
            #print ("hashtrain6 sample:", hashtrain6.takeSample(True,4,0))
            hashtest6  = test_5.map (lambda(f,x):(f,hashVector(x,use_hash_signing)))


        if use_lexicon:
            #logPrint('##### CREATE A DOC VECTOR AGAINST THE LEXICON   #####')
            train6=vectorise(train5,lexicon)
            #print ("traint6 sample:", train6.takeSample(True,4,0))
            test_6=vectorise(test_5,lexicon)

        # 3 Test whether the file is spam (i.e. the path contains spmsg) and replace the filename
        # by a 1 (spam) or 0 (ham) accordingly. Use map() to create an RDD of LabeledPoint objects.
        # See here http://spark.apache.org/docs/latest/mllib-naive-bayes.html for an example,
        # and here http://spark.apache.org/docs/latest/api/python/pyspark.mllib.regression.LabeledPoint-class.html
        # for the LabelledPoint documentation.

        #logPrint('#####      TEST WHETHER FILE IS SPAM       #####')
        ##### REPLACE FILENAME BY 1 (spam) 0 (ham) #####

        if use_lexicon:
            train7 = train6.map (lambda(f,x):(1 if 'spmsg' in f else 0, x))
            #print ("train7 sample",train7.take(2))
        if use_hash:
            hashtrain7 = hashtrain6.map (lambda(f,x):(1 if 'spmsg' in f else 0, x))
            #print ("hashtrain7 sample",hashtrain7.take(2))

        logTimeInterval(start_time)


        #logPrint('#####      MAP TO LABELLED POINTS      #####')
        if use_lexicon:
            train8 = train7.map (lambda (f,x):LabeledPoint(f,x))
        if use_hash:
            hashtrain8 = hashtrain7.map (lambda (f,x):LabeledPoint(f,x))

        logTimeInterval(start_time)

        #4 Use the created RDD of LabelledPoint objects to train the NaiveBayes and save
        # the model as a variable nbModel (again, use this example
        # http://spark.apache.org/ docs/latest/mllib-naive-bayes.html and here is the documentation
        # http://spark. apache.org/docs/latest/api/python/pyspark.mllib.regression.LabeledPoint-class. html).

        #logPrint('#####      TRAIN THE NAIVE BAYES      #####')
        if use_lexicon:
            nbModel = NaiveBayes.train(train8, 1.0)
        if use_hash:
            hashnbModel =  NaiveBayes.train(hashtrain8, 1.0)

        logTimeInterval(start_time)

        # 5 Use the files from /data/extra/spam/bare/part2 and prepare them like in task 3).
        # Then use nbModel to predict the label for each vector you have and compare it to the original,
        # to test the performance of your classifier.

        #          """
        #logPrint('#####      RUN THE PREDICTION      #####')
        if use_lexicon:
            test_7 = test_6.map(lambda (f,x):(1 if 'spmsg' in f else 0,int(nbModel.predict(x).item())))
            if use_log: print ("prediction sample: ",test_7.takeSample(False,20,0))

        if use_hash:
            hashtest7 = hashtest6.map(lambda (f,x):(1 if 'spmsg' in f else 0,int(hashnbModel.predict(x).item())))
            if use_log: print ("prediction sample: ",hashtest7.takeSample(False,20,0))


        logTimeInterval(start_time)

        if 0:
            if use_lexicon:
                print """\
    ____________________________________
    #####      EVALUATION      #####

    """
                printConfusionDict(confusionDict(test_7.collect()))

            if use_hash:
                print('____________________________________')
                print('#####    HASH  EVALUATION      #####')
                print("#####    size %i" % (hashtable_size))
                print "\n"
                printConfusionDict(confusionDict(hashtest7.collect()))

        else:
           
            if use_lexicon:
                cd = confusionDict(test_7.collect())
                print("L\t\t%i\t%i\t%i\t%i\t%.3f\t%.3f\t%.3f" \
                      %(\
                        cd['TP'],cd['FP'],cd['FN'],cd['TN'],\
                        cd['Recall'],cd['Precision'],cd['Fmeasure']))
            if use_hash:
                cd = confusionDict(hashtest7.collect())
                print("%i\t%i\t%i\t%i\t%i\t%i\t%.3f\t%.3f\t%.3f" \
                      %(hashtable_size,use_hash_signing,\
                        cd['TP'],cd['FP'],cd['FN'],cd['TN'],\
                        cd['Recall'],cd['Precision'],cd['Fmeasure']))




