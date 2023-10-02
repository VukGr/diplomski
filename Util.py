def plotHistory(history):
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()

log = print

def logNodeProcessed(nodeResult):
    log('Processed Node:')
    log('\tArgs: ', end='')
    log(nodeResult['args'])
    log(f"\tCached: {nodeResult['cached']}")
    log(f"\tTime: {nodeResult['time']}")
    log(f"\tScore: {nodeResult['score']}")
    log(f"\tqScore: {nodeResult['qscore']}")
    log(f"\tDepth: {nodeResult['depth']}")
    log(f"\tAccuracy: {nodeResult['accuracy']}")

def logRegion(region, depth):
    for arg in region:
        log('\t', end='')
        log(f'{arg.name}={arg.val}')
    log(f'\tDepth: {depth}')
