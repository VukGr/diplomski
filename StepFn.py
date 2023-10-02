def realNum(minV, maxV, switch):
    mid = (minV+maxV)/2
    if switch:
        return (mid, maxV)
    else:
        return (minV, mid)

# Returns a lambda that cuts off digits past the digitCutoff
def realNumWithCutoff(digitCutoff):
    def stepFn(minV, maxV, switch):
        mid = round((minV+maxV)/2, digitCutoff)
        if switch:
            return (mid, maxV)
        else:
            return (minV, mid)
    return stepFn

def intNum(minV, maxV, switch):
    mid = (minV+maxV)//2
    if switch:
        return (mid, maxV)
    else:
        return (minV, mid)

def none(minV, maxV, switch):
    return (minV, maxV)
