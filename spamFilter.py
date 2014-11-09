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







""" INPUT  """

def validateInput():
    if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: spamPath <folder> (optional) stoplist<file>"
        exit(-1)


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


def processRDD(rdd, stop_list=[]):
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
    :return: prediction rdd
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


def reportResults(lexPrediction, hashPrediction, filehandle, verbose=None):
    logTimeIntervalWithMsg('#####      EVALUATE THE RESULTS      #####')

    if verbose:
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




def reportResultsForCollectedHashOnOneLine(hash_prediction,hashtable_size,use_hash_signing, total_time=0, filehandle=None):
    '''
    :param hashPrediction:array of COLLECTED prediction results
    :param filefilehandle: file to write results to
    :return:confusionDict
    '''

    cd = confusionDict(hash_prediction)
    cd['time_since_last'] = float(total_time)

    string = "{0}\t{1}\t{2[TP]}\t{2[FP]}\t{2[FN]}\t{2[TN]}\t" \
             "{2[Recall]:.3f}\t{2[Precision]:.3f}\t{2[Fmeasure]:.3f}\t{2[Accuracy]:.3f}\t" \
             "{2[time_since_last]:0.3f}"\
        .format(hashtable_size, use_hash_signing, cd)
    filePrint(string,filehandle)
    return cd

def reportResultsForHashOnOneLine(hash_prediction,hashtable_size,use_hash_signing, total_time=0, filehandle=None):
    '''
    :param hashPrediction:prediction generate from hashes
    :param filefilehandle: file to write results to
    :return:confusionDict
    '''
    hash_prediction = hash_prediction.collect()
    return reportResultsForCollectedHashOnOneLine(hash_prediction,hashtable_size,use_hash_signing,total_time,filehandle)


def filePrint(string,filehandle=None):
    if filehandle:
        filehandle.write("{}\n".format(string))
    print(string)

def reportResultsForAll(results,use_hash_signing, filehandle=None):
    '''
    :param results:dictionary of results per hash_table_size N-fold array
    :param use_hash_signing: (BOOL) are we using hash_signing?
    :param filehandle: file to write results to
    :return:None
    '''



    '''derive an average from all of the results'''
    ''' {100 => summary for 100
         300 => summary for 300
         ....
         100 => summaryDict}'''




''' LOGGING '''

def timeDeltas():
    global c_time
    global s_time
    global is_time
    ds_time = time()
    deltas_since_start = ds_time - s_time
    deltas_since_last = ds_time - is_time
    is_time = ds_time
    return {"time_since_start":deltas_since_start,'time_since_last':deltas_since_last}

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
        filePrint(string,filehandle)



def logPrint(string):
    print string if use_log else '.',


'NEW FUNCS, UNFILED'

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

def dictOfArrayOfRDDs(keys,rdds):
    '''
    :param keys: list of hash table sizes (ints) we want to test against
    :param rdds: list of rdds
    :return:dictionary of arrays of vectorised rdds
    '''
    logfuncWithArgs()

    dictOfArrays = {}
    for hash_size in keys:
        array = []
        for rdd in rdds:
            vector = vectoriseWithHashtable(rdd,hash_size)
            vector.cache()
            array.append(vector)
        dictOfArrays[str(hash_size)]=array
    return dictOfArrays



def mergeRDDs (rdd1, rdd2):
    return rdd1.union(rdd2) if rdd1 else rdd2

def mergeDicts(dictionary, newDict):
    '''
    :param dictionary: dictionary of key:RDD pairs (possibly empty)
    :param newDict: dictionary of key:RDD pairs
    :return:
    '''
    #logfuncWithVals()
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

def mergeArrayOfRDDsWithPop(arrayOfRDDs, idx_to_pop):
    '''
    :param arrayOfRDDs:array of RDDs, one per mail folder
    :param idx_to_pop: index of array item (folder) to exclude
    :return:single rdd of union'd rdds excluding the popped one.
    '''
    #logfuncWithVals()
    mergedRDD = None
    for idx,rdd in enumerate(arrayOfRDDs):
        if idx is not idx_to_pop:
            mergedRDD = mergedRDD.union(rdd) if mergedRDD else rdd
    return mergedRDD

def mergeArrayOfDictsWithPop(arrayOfDicts,idx_to_exclude):
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


def mergeArrayOfDicts(arrayOfDicts):
    '''
    :param arrayOfDicts: [  {hash_size:rddVector, hash_size:rddVector, hash_size:rddVector}...},  {hash_size:rddVector, hash_size:rddVector, hash_size:rddVector}...}...]
    :param idx_to_exclude: index of array item to exclude
    :return:single dictionary comprising all array dicts except the idx_to_exclude dict
    '''
    #logfuncWithArgs()
    mergedDict = {}
    for idx,dict in enumerate(arrayOfDicts):
            mergedDict = mergeDicts(mergedDict,dict)
    return mergedDict


' THE MAIN LOOP '

if __name__ == "__main__":

    use_lexicon = 0
    use_hash = 1
    use_hash_signing = 1

    use_log = 1
    global s_time
    global is_time

    s_time = time()
    start_s_time = s_time
    is_time = s_time
    validateInput()

    sc = SparkContext(appName="spamFilter")
    #logTimeIntervalWithMsg("spark initialised, resetting timers")
    #s_time = time()

    filehandle = open('out.txt', 'a')
    spamPath = sys.argv[1]
    stop_list = stopList(sys.argv[2]) if len(sys.argv)>2 else []
    test_set = int(sys.argv[3]) if len(sys.argv)>3 else 0
    validation_index = 1

    dict = dictOfFileRDDs(spamPath)
    rdds = []
    paths = []
    for path, rdd in dict.iteritems():
        rdds.append(processRDD(rdd,stop_list))
        paths.append(path)

    hash_table_sizes = [100,300,1000,3000,10000]
    hash_table_strings = map(lambda x: str(x),hash_table_sizes)

    dict_of_arrays_of_rdds = dictOfArrayOfRDDs(hash_table_sizes
                                ,rdds)

    '''
     now we have  a dict of arrays:
    {
    '100':[vector1, vector2, ...]
    '300':[vector1, vector2, ...]
    ...
    }
     access as dictOfArrays['100'][0] .. etc
    '''

    logTimeIntervalWithMsg('starting the folds...',filehandle)
    validation_results = {}
    summary_results = {}
    test_results = {}
    nb_model_for_hash_sizes = {}
    nbModel = ""
    test_dict = {}
    paths.pop(test_set)
    print ("\nrunning time to lap0, including spark initialisation: {:.6f}\n".format(time()-start_s_time))
    #print ("running time to lap0, after spark initialisation: {:.6f}".format(time_b))

    string = "hSize\tsigned?\tTP\tFP\tFN\tTN\tRecall\tPrcsion\tFMeasre\tAcc\tTime"
    filePrint(string,filehandle)
    for hash_table_size in hash_table_strings:

        #use_log = 1
        #logTimeIntervalWithMsg('\n\n#####  HASHTABLE:{0}\n\n'.format(hash_table_size),filehandle)
        #use_log = 0
        #print("HASHSIZE:{}\n".format(hash_table_size))
        array_of_rdds = dict_of_arrays_of_rdds[hash_table_size];
        test_dict[hash_table_size] = array_of_rdds.pop(test_set)
        for (validation_idx, validation_rdd) in enumerate(array_of_rdds):

            lap_time = time()
            #print(array_of_rdds)
            use_log = 1
            #logTimeIntervalWithMsg('#####  LAP:{0}     validation index:{0} path:{1}'.format(validation_idx, paths[validation_idx]),filehandle)
            use_log = 0
            use_hash_signing = 1

            training_rdd = mergeArrayOfRDDsWithPop(array_of_rdds,validation_index)
            #training_dict = mergeArrayOfDictsWithPop(array_of_dicts_of_rdds,validation_idx)



            nbModel = trainBayes(training_rdd)
            nb_model_for_hash_sizes[hash_table_size] = nbModel
            validation_prediction_rdd = predict(validation_rdd, nbModel)
            ltime = time()-lap_time
            validation_prediction_array = validation_prediction_rdd.collect()
            reportResultsForCollectedHashOnOneLine(validation_prediction_array,hash_table_size,use_hash_signing,ltime, filehandle)
            #print ".",
            #accumulate the results (to display average per hash table size after all folds processed


            prediction_dict = {}
            prediction_dict['prediction'] = validation_prediction_rdd
            prediction_dict['lap_time'] = ltime
            prediction_dict['model'] = nbModel
            if hash_table_size not in validation_results:
                validation_results[hash_table_size] = []
            validation_results[hash_table_size].append(prediction_dict)




           # validation_results[hash_table_size][str(validation_idx)]['prediction'] = hash_prediction
           # validation_results[hash_table_size][str(validation_idx)]['time'] = i_time


    'summarise results'
    #print ("validation results")
    #pprint(validation_results)

    for hash_table_size, hash_table_size_results in validation_results.iteritems():
        for hash_prediction in hash_table_size_results:
            if hash_table_size in summary_results:
                summary_results[hash_table_size]['prediction'] = summary_results[hash_table_size]['prediction']\
                                                                           .union(hash_prediction['prediction'])

                summary_results[hash_table_size]['total_time'] += hash_prediction['lap_time']

                #print summary_results[hash_table_size].take(20)
            else:
                summary_results[hash_table_size] = {}
                summary_results[hash_table_size]['prediction'] = hash_prediction['prediction']
                summary_results[hash_table_size]['total_time'] = hash_prediction['lap_time']

                #print summary_results[hash_table_size].take(20)

    print ("\nresults...")
    #pprint(summary_results)


    filePrint ("\n\ncross-validation totals - all folds\n",filehandle)

    string = "hSize\tsigned?\tTP\tFP\tFN\tTN\t" \
        "Recall\tPrcsion\tFMeasre\tAcc\tTime"
    filePrint(string,filehandle)
    for hash_table_size in hash_table_sizes:
        result = summary_results[str(hash_table_size)]
        prediction = result['prediction']
        total_time = result['total_time']
        prediction = prediction.collect()
        reportResultsForCollectedHashOnOneLine(prediction,hash_table_size,use_hash_signing,total_time,filehandle)



    filePrint ("\n\ntest results (test set is set {})\n".format(test_set),filehandle)

    string = "hSize\tsigned?\tTP\tFP\tFN\tTN\t" \
        "Recall\tPrcsion\tFMeasre\tAcc\tTime"
    filePrint(string,filehandle)

    s_time = time()
    sums_time = 0
    totals_time = 0
    cumulative_lap_time = 0
    for hash_table_size in hash_table_strings:
        test_start_s_time = time()
        test_prediction = predict(test_dict[hash_table_size],nb_model_for_hash_sizes[hash_table_size])
        test_prediction = test_prediction.collect()
        test_end_s_time = time()
        laps_time = test_end_s_time - test_start_s_time
        reportResultsForCollectedHashOnOneLine(test_prediction,hash_table_size,use_hash_signing,laps_time,filehandle)


    end_s_time = time()
    runtime_s = end_s_time - start_s_time
    print("\ntotal running time:{}".format(runtime_s))





    filehandle.close()
