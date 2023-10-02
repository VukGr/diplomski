import pandas as pd
import tensorflow as tf
import numpy as np
import sys

from Dataset import Dataset
from NetParams import NetParams
from DBHandler import DBHandler
from ArgParam import ArgParam
from Tester import Tester
import StepFn

dataset = None
pathPrefix = './out'
datasetName = sys.argv[2]
if datasetName == 'ds1':
    pathPrefix='./out/ds1/'

    dataset = Dataset(
        data=pd.read_csv('./datasets/train.csv', na_values='Error'),
        problemType='regression',
        y_field='churn_risk_score',
        y_transform=(lambda y: (y + 1) / 6),
        feature_blacklist=[
            'customer_id', 'Name', 'security_no', 'joining_date', 'referral_id',
            'last_visit_time'
        ],
        batchSize=64,
        epochs=100
    )
elif datasetName == 'radon':
    pathPrefix='./out/radon/'

    dataset = Dataset(
        dsType='tfds',
        data='radon',
        problemType='regression',
        y_field='activity',
        y_transform=(lambda y: (y-y.mean())/y.std()),
        batchSize=64,
        epochs=100,
    )
elif datasetName == 'mnist':
    pathPrefix='./out/mnist/'

    dataset = Dataset(
        dsType='tfds',
        data='mnist',
        problemType='classification',
        y_field='label',
        y_size=10,
        x_transform=(lambda x: np.stack(x.image.apply(lambda arr: arr.flatten()).to_numpy())),
        makePreprocessor=None,
        batchSize=64,
        epochs=100,
    )
else:
    raise Exception('Invalid dataset')

if 'DB' in globals():
    DB.con.close()
DB = DBHandler(
    #':memory:', ':memory:', ':memory:',
    pathPrefix=pathPrefix,
    params=[
        'baseWidth', 'baseHeight',
        'dropoutPeriod', 'dropoutValue', 'normPeriod'
    ]
)

runner = Runner(
    dataset,
    DB=DB,
    stepFunctions={
        'baseWidth': StepFn.intNum,
        'baseHeight': StepFn.intNum,
        'dropoutPeriod': StepFn.realNumWithCutoff(2),
        'dropoutValue': StepFn.realNumWithCutoff(1),
        'normPeriod': StepFn.realNumWithCutoff(2),
    },
    scoreFn=lambda score,depth: score+depth*100
)

if __name__ == '__main__':
    count = int(sys.argv[3])
    if sys.argv[1] == 'start':
        runner.startSearch([
            ArgParam(name='baseWidth', val=(8, 256)),
            ArgParam(name='baseHeight', val=(1, 64)),
            ArgParam(name='dropoutPeriod', val=(0, 1)),
            ArgParam(name='dropoutValue', val=(0.1, 0.8)),
            ArgParam(name='normPeriod', val=(0, 1)),
        ], count)
    elif sys.argv[1] == 'resume':
        runner.resumeSearch(count)
