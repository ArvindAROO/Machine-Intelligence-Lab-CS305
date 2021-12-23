def A_star_Traversal(cost, heuristic, start_point, goals):
    route = []
    visitedArray = []
    route = [start_point]
    temp = [[heuristic[start_point], route]]
    while len(temp) > 0:
        totalCost, totalPath = temp.pop(0)
        lastNode = totalPath[-1]
        totalCost -= heuristic[lastNode]
        if lastNode in goals:
            #success, path found
            return totalPath
        visitedArray.append(lastNode)

        for index in range(len(cost[0])):
            if cost[lastNode][index] not in [0, -1]:
                tempPath = totalPath + [index]
                tempCost = totalCost + cost[lastNode][index] + heuristic[index]
                if index not in visitedArray and tempPath not in [i[1] for i in temp]:
                    temp.append((tempCost, tempPath))
                    temp = sorted(temp)
    if len(route) == 0:
        return route
    else:
        return []



def DFS_Traversal(cost, start_point, goals):
    path = []
    pathList = []
    visitedArray = [0 for _ in range(len(cost))]                
    arrayAsStack = [(start_point, [start_point])]
    while(len(arrayAsStack) != 0):  
        last, currPath = arrayAsStack[-1]
        
        if visitedArray[last] == 1:
            pass
        else:
            visitedArray[last] = 1
           
            if last in goals:
                #success, path found
                return currPath

            for element in range(len(cost)-1, 0, -1):
                if cost[last][element] >= 1:
                    if visitedArray[element] == 1:
                        pass
                    else:
                        tempArray = [i for i in currPath]
                        tempArray.append(element)
                        arrayAsStack.append((element, tempArray))
    if len(path) == 0:
        return path
    else:
        return []
