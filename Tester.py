import math
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow import keras

from dataclasses import asdict

import StepFn
from ArgParam import ArgParam

from NetParams import NetParams

from Util import log, logRegion, logNodeProcessed

def defaultLossQScoreFn(score, depth):
    return score*(depth+1)

def defaultAccQScoreFn(score, depth):
    return score*(1-depth/5)

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

badModel = None

class Tester:
    def __init__(self, dataset, DB, stepFunctions,
                 scoreFn = defaultLossQScoreFn):
        self.dataset = dataset
        self.stepFunctions = stepFunctions
        self.scoreFn = scoreFn
        self.DB = DB

    def getAllCombinations(self, args, depth=0):
        def genCombination(vals):
            resArgs = {}
            resSubregion = []
            for j in range(argsLen):
                minOrMax = vals & 1
                vals >>= 1
                arg = args[j]

                # In case min and max settled on the same value don't generate
                # duplicate combinations
                if arg.val[0] == arg.val[1] and minOrMax == 1:
                    return None

                stepFn = self.stepFunctions[arg.name]
                resArgs[arg.name] = arg.val[minOrMax]
                resSubregion.append(ArgParam(
                    name=arg.name,
                    val=stepFn(*arg.val, minOrMax),
                ))
            return {'args': resArgs, 'depth': depth, 'region': resSubregion}

        combinations = []
        argsLen = len(args)
        for i in range(2**argsLen):
            res = genCombination(i)
            if res != None:
                combinations.append(res)
        return combinations

    def testModel(self, model):
        dataset = self.dataset
        startTime = timer()
        history = model.fit(
            dataset.X_train, dataset.y_train,
            validation_data=dataset.ds_valid,
            batch_size=dataset.batchSize,
            epochs=dataset.epochs,
            callbacks=[earlyStopping],
            verbose=0,
        )
        time = timer() - startTime

        score = model.evaluate(
            dataset.X_test,
            dataset.y_test,
            batch_size=dataset.batchSize,
            verbose=0
        )

        return (score, history.history, time)

    def processCombination(self, comb):
        netparam = NetParams(**comb['args'])

        npArgs = asdict(netparam)
        constrainedArgs = {key: npArgs[key] for key in comb['args']}

        cachedScore = self.DB.checkCached(constrainedArgs)
        isCached = cachedScore != None

        (score, accuracy, history, time) = (None, None, None, None)
        if isCached:
            score = cachedScore
        else:
            model = netparam.genNetwork(self.dataset)
            (score, history, time) = self.testModel(model)
            if self.dataset.problemType == 'classification':
                [score, accuracy] = score

            if type(score) != float or math.isnan(score):
                badModel = model
                raise Exception(f'Bad score: {score}')

        result = {
            **comb,
            'score': score,
            'qscore': self.scoreFn(score, comb['depth']),
            'accuracy': accuracy,
            'cached': isCached,
            'history': history,
            'time': time,
            'args': constrainedArgs,
        }

        logNodeProcessed(result)

        return result

    # Searches one region, non-recursively
    def searchRegion(self, region, depth=-1):
        log("Searching region:")
        logRegion(region, depth+1)

        startTime = timer()

        combinations = self.getAllCombinations(region, depth+1)
        results = []
        for comb in combinations:
            res = self.processCombination(comb)
            self.DB.saveData(res)
            results.append(res)

        totalTime = timer() - startTime
        log(f'Total region time: {totalTime/60} minute(s).')
        return results

    # Resumes search by going through DB queue
    def resumeSearch(self, cycleCount=math.inf):
        reg = self.DB.getNextRegion()
        while reg != None and cycleCount > 0:
            results = self.searchRegion(reg['region'], reg['depth'])
            reg = self.DB.getNextRegion()
            cycleCount = cycleCount - 1

    # Starts search by giving starting arguments etc.
    def startSearch(self, region, cycleCount=math.inf):
        self.searchRegion(region)
        self.resumeSearch(cycleCount-1)
