import numpy
import random

def tabu_search(function, Bounds):
    sBest = numpy.array([random.uniform(*B) for B in Bounds])
    bestCandidate = sBest
    tabuList = [list(sBest)]
    maxTabuSize = 5
    min_error = error = function(sBest)
    count = 0
       
    f = 128

    while min_error > 1e-2 and count < f:
        
        sNeighborhood = numpy.array([numpy.linspace(bc-1, bc+1) for bc in bestCandidate]).T
        bestCandidate = sNeighborhood[0]

        for sCandidate in sNeighborhood:
            error = function(sCandidate)
            if (list(sCandidate) not in tabuList) and (error < function(bestCandidate)):
                bestCandidate = sCandidate

        if function(bestCandidate) < function(sBest):
            sBest = bestCandidate
            min_error = function(sBest)

        tabuList.append(list(bestCandidate))

        if len(tabuList) > maxTabuSize:
            tabuList = tabuList[1:]

        count += 1

    return sBest


    