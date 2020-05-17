import numpy
import random

def tabu_search(function, Bounds):
    sBest = [random.uniform(*B) for B in Bounds]
    bestCandidate = sBest
    tabuList = [sBest]
    maxTabuSize = 5
    min_error = error = function(sBest)
    count = 0
    
    while min_error > 1e-3 and count < 1000:
        
        sNeighborhood = numpy.array([numpy.linspace(bc-1, bc+1, 2000) for bc in bestCandidate]).T
        bestCandidate = sNeighborhood[0]
        
        for sCandidate in sNeighborhood:
            error = function(sCandidate)
            if (list(sCandidate) not in tabuList) and (error < function(bestCandidate)):
                bestCandidate = list(sCandidate)
        
        if function(bestCandidate) < function(sBest):
            sBest = bestCandidate
            min_error = function(sBest)
        
        tabuList.append(bestCandidate)
        
        if len(tabuList) > maxTabuSize:
            tabuList = tabuList[1:]
        
        count += 1
        
    return sBest