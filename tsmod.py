import numpy
import random

def tabu_search(function, Bounds):
    sBest = [random.uniform(*B) for B in Bounds]
    bestCandidate = sBest
    tabuList = []
    tabuList.append(sBest)
    maxTabuSize = 5
    error = function(sBest)
    count = 0
    
#     while error > 1e-3 and count < 100:
    while count < 100:
        
        sNeighborhood = numpy.array([numpy.linspace(bc-0.02 if bc-0.02 > 0 else bc , bc+0.02) for bc in bestCandidate]).T
        bestCandidate = sNeighborhood[0]
        
        for sCandidate in sNeighborhood:
            error = function(sCandidate)
            if (list(sCandidate) not in tabuList) and (error < function(bestCandidate)):
                bestCandidate = list(sCandidate)
        
        if function(bestCandidate) < function(sBest):
            sBest = bestCandidate
        
        tabuList.append(bestCandidate)
        
        if len(tabuList) > maxTabuSize:
            tabuList = tabuList[1:]
        if count > 99: print('oof')
        count += 1
        
    return sBest