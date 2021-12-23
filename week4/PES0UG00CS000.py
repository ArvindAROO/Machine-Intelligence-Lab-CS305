import numpy as np


class KNN:
    def __init__(self, k_neigh, weighted=False, p=2):
        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        self.data = data
        self.target = target.astype(np.int64)
        return self

    def find_distance(self, x):
        self.X = x
        sol = []
        for i in self.X:
            ans = []
            for j in self.data:
                t = sum((abs(a-b) ** self.p) for a, b in zip(i,j))**(1/self.p)
                ans.append(t)
            sol.append(ans)
        else:
            sol = np.array(sol)
            return sol
        
    def k_neighbours(self, x):
        Dist = self.find_distance(x)
        sortedDist = np.argsort(Dist, axis = 1)
        neighInfo = [[],[]]
        index = 0
        for element in sortedDist:
            currneighData = [[],[]]
            for j in range(self.k_neigh):
                currneighData[0] += [element[j]]
                currneighData[1] += [Dist[index][element[j]]]
            neighInfo[1] += [currneighData[1]]
            neighInfo[0] += [currneighData[0]]
            index += 1
        else:
            return (neighInfo[1],neighInfo[0])

    def predict(self, x):
        pred = []
        target = self.target
        neighs = self.k_neighbours(x)
        for i in neighs[1]:
            temp = {}
            for ind in i:
                if target[ind] not in temp:
                    temp[target[ind]] = 1
                else: 
                    temp[target[ind]] += 1
            present = target[i[0]]
            for key in temp:
                if temp[key] <= temp[present]:
                    pass
                else:
                    present = key
            else:
                pred += [present]
        else:
            return np.array(pred)

    def evaluate(self, x, y):
        return (100*(np.sum(self.predict(x) == y)/len(y)))
