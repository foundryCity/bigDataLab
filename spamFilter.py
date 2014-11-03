import sys
from os import walk
from pprint import pprint
# from os.path import isfile, join
import re
# import numpy
from operator import add
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
# from datetime import datetime  #, time, timedelta
from time import time
from pyspark import SparkContext
import inspect
import collections




''' INITIALISING '''

def initialiseSpark():
    global s_time
    sc = SparkContext(appName="spamFilter")
    logTimeIntervalWithMsg("spark initialised, resetting timer")
    s_time = time()
    return sc


""" INPUT  """

def validateInput():
    if len(sys.argv) != 3:
        print >> sys.stderr, "Usage: spamPath <folder> stoplist<file>"
        exit(-1)

# '''
# def buildRDDs(path, validation_index):
#     fileRDDs = arrayOfFileRDDs(path)
#     validationPath = path+'part'+str(validation_index+1)
#     result = trainingAndTestRDDs(fileRDDs,validation_index)
#     result.append(validationPath)
#     return result
# '''


' PROCESS RDDS '

def remPlural(word):
    word = word.lower()
    return word[:-1] if word.endswith('s') else word

def lexiconArray(rdd):
    # input: rdd of (file,word) tuples
    # output: [word1,word2,word3] array of distinct words
    logTimeIntervalWithMsg("##### BUILDING THE LEXICON #####")
    training_words = rdd.map(lambda(f, x): x)
    logTimeIntervalWithMsg("training_words: %i" % training_words.count())
    training_lexicon = training_words.distinct()
    logTimeIntervalWithMsg("training_lexicon: %i" % training_lexicon.count())
    return training_lexicon.collect()


def wordCountPerFile(rdd):
    logfuncWithArgs()
    # input: rdd of (file,word) tuples
    # return: rdd of (file, [(word, count),(word, count)...]) tuples
    #logTimeIntervalWithMsg("##### BUILDING wordCountPerFile #####")
    result = rdd.map(lambda(x): ((x[0], x[1]), 1))
    # print('##### GETTING THE  ((file,word),n)
    # WORDCOUNT PER (DOC, WORD) #####')
    result = result.reduceByKey(add)
    # print('##### REARRANGE AS  (file, [(word, count)])  #####')
    result = result.map(lambda (a, b): (a[0], [(a[1], b)]))
    # print ('##### CONCATENATE (WORD,COUNT) LIST PER FILE
    # AS  (file, [(word, count),(word, count)...])  #####')
    result = result.reduceByKey(add)
    return result




#
# def processRDD_old(rdd, create_lexicon):
#     ''' input: rdd as read from filesystem
#         output: array of [processed RDD,lexicon]
#         or [processed RDD] if create_lexicon is None
#         '''
#
#     logTimeIntervalWithMsg("##### BUILDING (file, word) tuples #####")
#
#     rdd = rdd.flatMap(lambda (file, words):
#                                ([(file[file.rfind("/")+1:], remPlural(word))
#                                  for word in re.split('\W+', words)
#                                  if len(word) > 0]))
#     pprint(rdd.takeSample(True,4,0))
#
#     logTimeIntervalWithMsg("processRDD {}".format(rdd.takeSample(True,1,0)))
#     lexicon = lexiconArray(rdd) if create_lexicon else None
#     processedRDD = wordCountPerFile(rdd)
#     logTimeIntervalWithMsg(rdd.take(1))
#     return [rdd, lexicon]


def processRDD(rdd, stop_list):
    '''
    :param rdd:  rdd as read from filesystem ('filename','file_contents')
    :param stop_list: [list, of, stop, words]
    :return:wordCountPerFileRDD [(filename,[(word,count)][(word,count)]...)]
    '''
    logfuncWithArgs()


    #logTimeIntervalWithMsg("##### BUILDING (file, word) tuples #####")


    flatmappedRDD = rdd.flatMap(lambda (file, words):
                                ([(file[file.rfind("/")+1:], remPlural(word))
                                 for word in re.split('\W+', words)
                                 if len(word) > 0
                                 and word not in stop_list
                                 and remPlural(word) not in stop_list]))

    # logTimeIntervalWithMsg("flatmappedRDD {}".format(flatmappedRDD.take(1)))
    wordCountPerFileRDD = wordCountPerFile(flatmappedRDD)
    # logTimeIntervalWithMsg("wordCountPerFileRDD {}".format(wordCountPerFileRDD.take(1)))
    return wordCountPerFileRDD

def processRDDWithPath(path, rdd, stop_list):
    '''
    :param path: folder name of rdd
    :param rdd:  rdd as read from filesystem ('filename','file_contents')
    :param stop_list: [list, of, stop, words]
    :return:wordCountPerFileRDD [((path, filename),[(word,count)][(word,count)]...)]
    '''
    logfuncWithArgs()
   # print("rdd: {}".format(rdd.collect()))


    #logTimeIntervalWithMsg("##### BUILDING (file, word) tuples #####")


    flatmappedRDD = rdd.flatMap(lambda (file, words):
                                ([((path, file[file.rfind("/")+1:]), remPlural(word))
                                 for word in re.split('\W+', words)
                                 if len(word) > 0
                                 and word not in stop_list
                                 and remPlural(word) not in stop_list]))

    # logTimeIntervalWithMsg("flatmappedRDD {}".format(flatmappedRDD.take(1)))
    wordCountPerFileRDD = wordCountPerFile(flatmappedRDD)
    # logTimeIntervalWithMsg("wordCountPerFileRDD {}".format(wordCountPerFileRDD.take(1)))
    return wordCountPerFileRDD



''' TRAINING and  PREDICTION  '''


def vector(tupleList, lexicon):
    '''
    :param tupleList: list of tuples [(word,count),(word,count)...]
    :param lexicon: dictionary of all words in the test sets
    :return:vector representing word counts in lexicon [0,1,4,2,..]
    '''
    vector = [0] * (len(lexicon))
    for (x, y) in tupleList:
        # print ("x:",x, " y:",y, "lexicon(x): ",lexicon.index(x))
        try:
            idx = lexicon.index(x)
        except Exception:
            continue
        vector[idx] = y
    return vector


def sign_hash(x):
    return 1 if len(x) % 2 == 1 else -1


def hashVectorSigned(tupleList, hashtable_size):
    '''
    :param tupleList: list of tuples [(word,count),(word,count)...]
    :param hashtable_size: int size of hash array
    :return:hashTable
    '''
    #logfuncWithArgs()
    hash_table = [0] * hashtable_size
    for (word, count)in tupleList:
        x = (hash(word) % hashtable_size) if hashtable_size else 0
        hash_table[x] = hash_table[x] + sign_hash(word) * count
    return map(lambda x: abs(x), hash_table)



def hashVectorUnsigned(tupleList, hashtable_size):
    '''
    :param tupleList: list of tuples [(word,count),(word,count)...]
    :param hashtable_size: int size of hash array
    :return:hashTable
    '''
    #logfuncWithArgs()
    hash_table = [0] * hashtable_size
    for (word, count)in tupleList:
        x = (hash(word) % hashtable_size) if hashtable_size else 0
        hash_table[x] = hash_table[x] + count
    return map(lambda x: abs(x), hash_table)


def hashVector(tupleList, hashtable_size, use_hash_signing):
    if use_hash_signing:
        return hashVectorSigned(tupleList, hashtable_size)
    else:
        return hashVectorUnsigned(tupleList, hashtable_size)



def vectoriseWithHashtable(rdd, hashtable_size):
    '''
    :param rdd:  ((path,file), [(word, count),(word, count)...]) tuples
    :param hashtable_size: int size of hash array
    :return: rdd of (<1|0>,[vector]) tuples
    '''
    #logfuncWithVals()
    #logTimeIntervalWithMsg('##### CREATE A DOC VECTOR OF HASHES  #####')
    #logTimeIntervalWithMsg('##### AND  REPLACE FILENAME WITH 1(SPAM) or 0(NOT SPAM)       #####')

    result = rdd.map(lambda(f, x):
                     (1 if 'spmsg' in f else 0
                      , hashVector(x, hashtable_size, use_hash_signing)))
    return result


def vectoriseWithLexicon(rdd, lexicon):
    '''
    :param rdd: (file, [(word, count),(word, count)...]) tuples
    :param lexicon: array of words in the lexicon to match against
    :return: rdd of (<1|0>,[sparse vector]) tuples
    '''
    logTimeIntervalWithMsg(
        '##### CREATE A DOC VECTOR AGAINST THE LEXICON   #####')
    result = rdd.map(lambda(f, wc): (1 if 'spmsg' in f else 0
                                     , vector(wc, lexicon)))
    return result


def trainBayes(rdd):
    '''
    :param rdd:rdd of vectors (1,[1,0,0,3,4,2,...],0,[1,0,0,3,4,2,...],1,[1,0,0,3,4,2,...]..)
    :return:naive bayes model to use in predict(rdd,nbModel)
    '''
    logTimeIntervalWithMsg('#####      MAP TO LABELLED POINTS      #####')
    processedRDD = rdd.map(lambda(f, x): LabeledPoint(f, x))
    logTimeIntervalWithMsg('#####      TRAIN THE NAIVE BAYES      #####')
    nbModel = NaiveBayes.train(processedRDD, 1.0)
    return nbModel


def predict(rdd, nbModel):
    '''
    :param rdd:rdd of test vectors (<1|0>,[],<1|0>,[]
    :param nbModel: naive bayes model derived from naiveBayes.train()
    :return:
    '''
    result = rdd.map(lambda (f, x): (f,
                                     int(nbModel.predict(x).item())))
    return result


def trainAndTest(trainingSet, testSet, use_lexicon, use_hash):
    logfuncWithArgs()

    if use_lexicon:

        logTimeIntervalWithMsg('##### T R A I N I N G  #####')
        trainingSet = vectoriseWithLexicon(trainingSet, lexicon)
        nbModel = trainBayes(trainingSet)

        logTimeIntervalWithMsg('##### T E S T I N G  #####')
        testSet = vectoriseWithLexicon(testSet, lexicon)
        lexPrediction = predict(testSet, nbModel)
    else:
        lexPrediction = None

    if use_hash:

        logTimeIntervalWithMsg('##### T R A I N I N G  #####')
        trainingSet = vectoriseWithHashtable(trainingSet, hashtable_size)
        nbModel = trainBayes(trainingSet)

        logTimeIntervalWithMsg('##### T E S T I N G  #####')
        testSet = vectoriseWithHashtable(testSet, hashtable_size)
        hashPrediction = predict(testSet, nbModel)
    else:
        hashPrediction = None

    return [lexPrediction, hashPrediction]


def arrayOfFileRDDs(path):
    '''
    :param path: path to email folders
    :return:array of rdds, one per folder
    '''

    folders = []
    rdds = []
    (_, folders, _) = walk(path).next()
    # http://stackoverflow.com/a/3207973, see comment from misterbee
    for folder in folders:
        rdd = sc.wholeTextFiles(path+folder)
        rdds.append(rdd)
    return rdds


def dictOfFileRDDs(path):
    '''
    :param path: path to email folders
    :return:dictionary {path, rdd}, one rdd per folder
    '''

    rddDict = {}
    (_, folders, _) = walk(path).next()
    # http://stackoverflow.com/a/3207973, see comment from misterbee
    for folder in folders:
        rdd = sc.wholeTextFiles(path+folder)
        rddDict[str(folder)] = rdd
    return rddDict



def trainingAndTestRDDs (rddArray, testIdx):
    trainingRDD = None
    for k,rdd in enumerate(rddArray):
        if k != testIdx:
            if (trainingRDD):
                trainingRDD = trainingRDD.union(rdd)
            else:
                trainingRDD = rdd

    return [trainingRDD,rddArray[testIdx]];





''' OUTPUT REPORTING '''


def confusionMatrix(tupleList):
    mx = [0, 0, 0, 0]
    for (x, y) in tupleList:
        mx[((x << 1) + y)] += 1
    return mx


def confusionDict(tupleList):
    mx = [0, 0, 0, 0]
    for (x, y) in tupleList:
        mx[((x << 1) + y)] += 1
    dict = {'TN': mx[0], 'FP': mx[1], 'FN': mx[2], 'TP': mx[3]}

    dict['TotalTrue'] = dict['TP'] + dict['FN']
    dict['TotalFalse'] = dict['TN'] + dict['FP']
    dict['TotalSamples'] = len(tupleList)
    dict['TotalPositive'] = dict['TP'] + dict['FP']
    dict['TotalNegative'] = dict['TN'] + dict['FN']
    dict['TotalCorrect'] = dict['TP'] + dict['TN']
    dict['TotalErrors'] = dict['FN'] + dict['FP']
    dict['Recall'] = \
        float(dict['TP']) / dict['TotalTrue'] \
        if dict['TotalTrue'] > 0 else 0
    dict['Precision'] = \
        float(dict['TP']) / dict['TotalPositive'] \
        if dict['TotalPositive'] > 0 else 0
    dict['Sensitivity'] = \
        float(dict['TP']) / dict['TotalSamples'] \
        if dict['TotalSamples'] > 0 else 0
    dict['Specificity'] = \
        float(dict['TN']) / dict['TotalSamples'] \
        if dict['TotalSamples'] > 0 else 0
    dict['ErrorRate'] = \
        float(dict['TotalErrors']) / dict['TotalSamples'] \
        if dict['TotalSamples'] > 0 else 0
    dict['Accuracy'] = \
        float(dict['TotalCorrect']) / dict['TotalSamples'] \
        if dict['TotalSamples'] > 0 else 0
    dict['Fmeasure'] = \
        2 * float(dict['TP']) / (dict['TotalTrue'] + dict['TotalPositive']) \
        if (dict['TotalTrue'] + dict['TotalPositive'] > 0) else 0
    dict['Fmeasure2'] = \
        1 / ((1 / dict['Precision']) + (1 / dict['Recall'])) \
        if dict['Precision'] > 0 and dict['Recall'] > 0 else 0
    dict['Fmeasure3'] = \
        2 * dict['Precision'] * dict['Recall'] / (dict['Precision'] + dict['Recall']) \
        if (dict['Precision'] + dict['Recall'] > 0) else 0
    return dict


def printConfusionMatrix(confusionDict):
    print ('''
            condition
   test    T         F
    T %6i    %6i
    F %6i    %6i'''
           % (confusionDict['TP'], confusionDict['FP'],
              confusionDict['FN'], confusionDict['TN']))


def printConfusionDict(confusionDict):
    print ('''
                  relevant
 retreived     yes       no
   yes  %6i TP %6i FP %6i
   no   %6i FN %6i TN %6i
        %6i    %6i    %6i'''
           % (confusionDict['TP'], confusionDict['FP'],
              confusionDict['TotalPositive'], confusionDict['FN'],
              confusionDict['TN'], confusionDict['TotalNegative'],
              confusionDict['TotalTrue'], confusionDict['TotalFalse'],
              confusionDict['TotalSamples']))
    print ('''
                  truth
    spam %6i TP %6i FP
     ham %6i FN %6i TN

classifier stats (classes spam and ham)
total samples: %i
     accuracy: %.3f    TP+TN/total
   error rate: %.3f    FN+FP/total

class-specific stats (class spam)
  sensitivity: %.3f   TP/total
  specificity: %.3f   FN/total
       recall: %.3f   TP/totalTrue TP/TP+TN
    precision: %.3f   TP/totalPos  TP/TP+FP
    f-measure: %.3f   2*TP/(totalTrue+totalPos) 2TP/(TP+TN+TP+FP)
   f-measure2: %.3f   1/(1/precision + 1/recall) \# this one looks wrong
               # http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html#10657
   f-measure3: %.3f   2 * precision * recall / (precision + recall)
'''
           % (confusionDict['TP'], confusionDict['FP'],
              confusionDict['FN'], confusionDict['TN'],
              confusionDict['TotalSamples'],
              confusionDict['Accuracy'],
              confusionDict['ErrorRate'],
              confusionDict['Sensitivity'],
              confusionDict['Specificity'],
              confusionDict['Recall'],
              confusionDict['Precision'],
              confusionDict['Fmeasure'],
              confusionDict['Fmeasure2'],
              confusionDict['Fmeasure3']))


def reportResults(lexPrediction, hashPrediction, filehandle):
    logTimeIntervalWithMsg('#####      EVALUATE THE RESULTS      #####')

    if 0:  # set to 1 for verbose reporting
        if use_lexicon:
            print('____________________________________')
            print('#####      EVALUATION      #####')
            print "\n"
            printConfusionDict(confusionDict(lexPrediction.collect()))

        if use_hash:
            print('____________________________________')
            print('#####    HASH  EVALUATION      #####')
            print("#####    size %i" % (hashtable_size))
            print "\n"
            printConfusionDict(confusionDict(hashPrediction.collect()))

    else:  # 1-line reporting (for spreadsheets)
        if use_lexicon:
            reportResultsForLexiconOnOneLine(lexPrediction,filehandle)

        if use_hash:
            reportResultsFrorHashOnOneLine(hashPrediction,filehandle)

def reportResultsForLexiconOnOneLine(lexPrediction,filehandle):
    '''
    :param lexPredicion:predicition generate from lexicon
    :param filefilehandle: file to write results to
    :return:None
    '''
    cd = confusionDict(lexPrediction.collect())
    string = "L\t\t%i\t%i\t%i\t%i\t%.3f\t%.3f\t%.3f\t%.3f" % (
        cd['TP'], cd['FP'], cd['FN'], cd['TN'],
        cd['Recall'], cd['Precision'], cd['Fmeasure'], cd['Accuracy'])
    if filehandle:
        filehandle.write(string, "\n")
    print(string)


def reportResultsFrorHashOnOneLine(hash_prediction,hashtable_size,use_hash_signing, filehandle):
    '''
    :param hashPrediction:prediction generate from hashes
    :param filefilehandle: file to write results to
    :return:None
    '''

    cd = confusionDict(hash_prediction.collect())
    global s_time
    time_delta = timeDeltas()
    string = "{0}\t{1}\t{2[TP]}\t{2[FP]}\t{2[FN]}\t{2[TN]}\t" \
             "{2[Recall]:.3f}\t{2[Precision]:.3f}\t{2[Fmeasure]:.3f}\t{2[Accuracy]:.3f}\t" \
             "{3[time_since_start]:.3f} {3[time_since_last]:.3f}".format(
     hashtable_size, use_hash_signing, cd, time_delta)
    if filehandle:
        filehandle.write("{}\n".format(string))
    print(string)




''' LOGGING '''

def timeDeltas():
    global s_time
    global i_time
    d_time = time()
    delta_since_start = d_time - s_time
    delta_since_last = d_time - i_time
    i_time = d_time
    return {"time_since_start":delta_since_start,'time_since_last':delta_since_last}

def logfuncWithArgs(start_time=None):
    stime = start_time if start_time else ""
    return logfunc(stime,"args",'novals',sys._getframe().f_back)

def logfuncWithVals(start_time=None):
    stime = start_time if start_time else ""
    return logfunc(stime,"noargs",'vals',sys._getframe().f_back)


def logfunc(start_time=None, args=None, vals=None, frame=None):
    fargs={}
    time_deltas = timeDeltas()
    frame = frame if frame else sys._getframe().f_back
    line_number = frame.f_code.co_firstlineno
    name = frame.f_code.co_name
    argvals = frame.f_locals if vals is "vals" else ""
    argnames = inspect.getargvalues(frame)[0] if args is "args" else ""
    comments = inspect.getcomments(frame)
    comments = comments if comments else ""
    print ("{comments}{time_s: >9,.3f} {time_i: >7,.3f} {name:} {argmames} {argvalse}".format(
        # comments,elapsed_time,line_number,name,argnames,argvals))
        comments=comments,time_s=time_deltas['time_since_start'],time_i=time_deltas['time_since_last'],line=line_number,name=name,argmames=argnames,argvalse=argvals))



def logTimeIntervalWithMsg(msg, filehandle=None):
    if (use_log):
        time_deltas = timeDeltas()
        message = msg if msg else ""
        delta_since_start = time_deltas['time_since_start']
        delta_since_last = time_deltas['time_since_last']
        string = "log:{0[time_since_start]:7,.3f} log:{0[time_since_last]:7,.3f} {1:}".format(time_deltas, message)
        print (string)
        if filehandle:
            filehandle.write("{}\n".format(string))


def logPrint(string):
    print string if use_log else '.',




def stopList(stop_file):
    '''
    :param stop_file: path to file of stopwords
    :return:python array of stopwords
    '''
    rdd = sc.textFile(stop_file)
    return rdd.flatMap (lambda x: re.split('\W+',x)).collect()


def arrayOfDictsOfRDDs(keys,rdds):
    '''
    :param keys: list of hash table sizes (ints) we want to test against
    :param rdds: list of rdds
    :return:array of dictionarys of vectorised rdds
    '''
    logfuncWithArgs()
    arrayOfHashDicts = []
    for (idx,rdd) in enumerate(rdds):
        #orderedDict = collections.OrderedDict
        dict={}
        arrayOfHashDicts.append(dict)
        for hash_size in keys:
            dict[str(hash_size)] = vectoriseWithHashtable(rdd,hash_size)
    return arrayOfHashDicts

def mergeDicts(dictionary, newDict):
    '''
    :param dictionary: dictionary of key:RDD pairs (possibly empty)
    :param newDict: dictionary of key:RDD pairs
    :return:
    '''
    #logfuncWithArgs()

    dictionary = {key: rdd.union(newDict[key]) for key, rdd in dictionary.iteritems()}
    for key in newDict:
        if key not in dictionary.keys():
            dictionary[key] = newDict[key]

    '''
    for key in dictionary:
        print("{} {}".format(key, len(dictionary[key].collect())))

    for key in newDict:
        if key in dictionary:
            dictionary[key] = dictionary[key].union(newDict[key])
        else:
            dictionary[key] = newDict[key]
    '''
    return dictionary


def mergeArrayOfDicts(arrayOfDicts,idx_to_exclude):
    '''
    :param arrayOfDicts: [  {hash_size:rddVector, hash_size:rddVector, hash_size:rddVector}...},  {hash_size:rddVector, hash_size:rddVector, hash_size:rddVector}...}...]
    :param idx_to_exclude: index of array item to exclude
    :return:single dictionary comprising all array dicts except the idx_to_exclude dict
    '''
    #logfuncWithArgs()
    mergedDict = {}
    for idx,dict in enumerate(arrayOfDicts):
        if idx is not idx_to_exclude:
            mergedDict = mergeDicts(mergedDict,dict)
    return mergedDict


if __name__ == "__main__":

    use_lexicon = 0
    use_hash = 1
    use_hash_signing = 1

    use_log = 1
    global s_time
    global i_time
    s_time = time()
    i_time = s_time
    validateInput()
    sc = initialiseSpark()
    filehandle = open('out.txt', 'a')
    spamPath = sys.argv[1]
    stop_list = stopList(sys.argv[2])
    validation_index = 1

    dict = dictOfFileRDDs(spamPath)
    rdds = []
    paths = []
    for path, rdd in dict.iteritems():
        rdds.append(processRDD(rdd,stop_list))
        paths.append(path)

    hash_table_sizes = [100,300,1000,3000,10000]
    array_of_dicts_of_rdds = arrayOfDictsOfRDDs(hash_table_sizes
                                ,rdds)
    '''
    now we have an array of dictionaries of rdds. One array entry per email directory
    [  {'100':rddVector, '300':rddVector, 1000:rddVector}...}
    ,  {'100':rddVector, '300':rddVector, 1000:rddVector}...}
    ,  {'100':rddVector, '300':rddVector, 1000:rddVector}...}
    ...
    ]
    each rddVector has the form ((<1|0>,[vector]),(<1|0>,[vector]),(<1|0>,[vector]),...)
    then we access as arrayOfDicts[0]['100']...

    which could just as well be a dict of arrays:
    {
    '100':[vector1, vector2, ...]
    '300':[vector1, vector2, ...]
    ...
    }

    then we access as dictOfArrays['100'][0]
    '''
    '''
    for (idx,rddDict) in enumerate(arrayOfHashDicts):
    '''
    logTimeIntervalWithMsg('starting the folds...')

    for (idx,testDict) in enumerate(array_of_dicts_of_rdds):
        use_log = 1
        #print("\n")
        logTimeIntervalWithMsg('\n\n#####  LAP:{} {}\n'.format(idx, paths[idx]))
        logTimeIntervalWithMsg('mergeArrayOfDicts - start')
        training_dict = mergeArrayOfDicts(array_of_dicts_of_rdds,idx)
        logTimeIntervalWithMsg('mergeArrayOfDicts - end')

        test_dict = array_of_dicts_of_rdds[idx]
        #print(testDict['100'].take(3))

        logTimeIntervalWithMsg('array_of_dicts_of_rdds - end')
        use_hash_signing = 1
        use_log = 0
        string = "hSize\tsigned?\tTP\tFP\tFN\tTN\t" \
        "Recall\tPrcsion\tFMeasre\tAcc\tTime"
        print(string)
        #this bits really ugly
        keys = sorted([int(key) for key in training_dict])
        keys = [str(key) for key in keys]
        for hash_table_size in keys:
            logTimeIntervalWithMsg('##### T R A I N I N G  #####')
            nbModel = trainBayes(training_dict[hash_table_size])
            logTimeIntervalWithMsg('##### T E S T I N G  #####')
            hash_prediction = predict(test_dict[hash_table_size], nbModel)
            reportResultsFrorHashOnOneLine(hash_prediction,hash_table_size,use_hash_signing, filehandle)


    filehandle.close()
