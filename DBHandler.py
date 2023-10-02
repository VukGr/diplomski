import sqlite3
import pickle

from dataclasses import asdict
from itertools import chain

from ArgParam import ArgParam

class DBHandler:
    CULL_KEEP = 1000

    def __init__(self, mainDbPath='main.db', instanceDbPath='instance.db',
                 historyDbPath='history.db', pathPrefix='', params=[]):
        mainDbPath = pathPrefix + mainDbPath
        instanceDbPath = pathPrefix + instanceDbPath
        historyDbPath = pathPrefix + historyDbPath

        self.con = sqlite3.connect(mainDbPath)
        self.cur = self.con.cursor()
        self.cur.execute("ATTACH DATABASE ? as instance", (instanceDbPath,))
        self.cur.execute("ATTACH DATABASE ? as history", (historyDbPath,))

        self.params = params

        resultStructure = ", ".join(params)
        queueStructure = ", ".join([p + minOrMax for p in params
                                    for minOrMax in ['Min', 'Max']])

        resultPlaceholders = ", ".join(['?'] * len(params))
        queuePlaceholders = ", ".join(['?'] * (2 * len(params)))

        self.cur.executescript(f"""
            CREATE TABLE IF NOT EXISTS
                Result(score REAL, accuracy REAL, time REAL, {resultStructure});

            CREATE TABLE IF NOT EXISTS
                instance.Queue(score REAL, depth INT, {queueStructure});
            CREATE INDEX IF NOT EXISTS instance.iQs ON Queue(score);

            CREATE TABLE IF NOT EXISTS
                history.List(resid INT NOT NULL PRIMARY KEY, history BLOB)
        """)

        self.queryInsertResult = "INSERT INTO Result(score, accuracy, time, " +\
            f"{resultStructure}) VALUES(?, ?, ?, {resultPlaceholders})"
        self.queryInsertQueue = "INSERT INTO instance.Queue(score, depth, " + \
            f"{queueStructure}) VALUES(?, ?, {queuePlaceholders})"

        self.queryFindCachedResult = f"SELECT score FROM Result WHERE " + \
            " AND ".join(map(lambda p: f"{p} = ?", params))

        self.querySelectBestRegion = f"SELECT rowid, score, depth, " + \
            f"{queueStructure} FROM instance.Queue ORDER BY score ASC LIMIT 1"

    def queueDeserialize(self, dbval):
        it = iter(dbval[2:])
        regionVals = list(zip(it,it))
        region = [ArgParam(name=self.params[i], val=regionVals[i])
                  for i in range(len(self.params))]
        return {'qscore': dbval[0], 'depth': dbval[1], 'region': region}

    def resultSerialize(self, obj):
        return (obj['score'], obj['accuracy'], obj['time'],
                *(obj['args'][key] for key in obj['args']))

    def queueSerialize(self, obj):
        return (obj['qscore'], obj['depth'],
                *chain(*(reg.val for reg in obj['region'])))

    def historySerialize(self, resId, hist):
        return (resId,
                sqlite3.Binary(pickle.dumps(hist, pickle.HIGHEST_PROTOCOL)))

    # Return next argparams, region, depth from instance.queue
    def getNextRegion(self):
        # Commit before next pop to prevent potential data loss
        self.cullQueue()
        self.con.commit()
        self.cur.execute(self.querySelectBestRegion)
        result = self.cur.fetchone()
        if result != None:
            resId, res = result[0], result[1:]
            self.cur.execute("DELETE FROM instance.Queue WHERE ROWID = ?",
                             (resId,))
            return self.queueDeserialize(res)

    def cullQueue(self):
        self.cur.execute(
            "DELETE FROM instance.Queue WHERE ROWID NOT IN \
            (SELECT ROWID FROM instance.Queue ORDER BY score ASC LIMIT ?)",
            (self.CULL_KEEP,)
        )

    def saveData(self, result):
        if not result['cached']:
            self.cur.execute(self.queryInsertResult,
                             self.resultSerialize(result))
            resId = self.cur.lastrowid

            self.cur.execute(
                "INSERT INTO history.list(resid, history) VALUES (?, ?)",
                self.historySerialize(resId, result['history']))

        self.cur.execute(self.queryInsertQueue, self.queueSerialize(result))

    def saveDataList(self, results):
        for result in results:
            self.commitResult(result)

    def checkCached(self, args_dict):
        args = tuple(args_dict.values())
        self.cur.execute(self.queryFindCachedResult, args)
        res = self.cur.fetchone()
        if res != None:
            score = res[0]
            return score

    def getResultHistory(self, resId):
        self.cur.execute('SELECT * FROM history.list WHERE resId = ?',
                         (resId,))
        history = pickle.loads(self.cur.fetchone()[1])
        return history
