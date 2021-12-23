import numpy as np
import pandas as pd
import random


def get_entropy_of_dataset(dataframe):
    unique = 2
    numberOfOutPuts = 2
    tempEntropy = 0
    output = dataframe[[dataframe.columns[-1]]].values.tolist()
    uniqueVal, counts = np.unique(output, return_counts=True)
    total = sum(counts)
    for times in counts:
        temp = times/total
        tempEntropy = tempEntropy - temp*(np.log2(temp))
    else:
        return tempEntropy



def get_entropy_of_attribute(dataframe, attribute):
    unique = 2
    headers = dataframe[attribute].values
    newHeaders = np.unique(headers)
    rows = dataframe.shape[0]
    currentEntropy = 0
    for item in newHeaders:
        part = dataframe[dataframe[attribute] == item]
        target = part[[part.columns[-1]]].values.tolist()
        spam, counts = np.unique(target, return_counts=True)
        sums = sum(counts)
        entropy = 0
        for times in counts:
            temp = times/sums
            if temp == 0:
                pass
            else:
                entropy = entropy - temp*(np.log2(temp))
        currentEntropy += entropy*(sum(counts)/rows)
    else:
        if currentEntropy > 0:
            return currentEntropy
        return -currentEntropy




def get_information_gain(dataframe, attribute):

    return abs(get_entropy_of_dataset(dataframe) - get_entropy_of_attribute(dataframe, attribute))



def get_selected_attribute(dataframe):

    atributeWiseGains = {}
    currColumn = None

    possiblemax = -1
    for attribute in dataframe.columns[:-1]:
        currAtttrgain = get_information_gain(dataframe, attribute)
        if currAtttrgain <= possiblemax:
            pass
        else:
            currColumn = attribute
            possiblemax = currAtttrgain
        atributeWiseGains[attribute] = currAtttrgain
    else:
        return (atributeWiseGains, currColumn)



def get_avg_info_of_attribute(dataframe, attribute):
    # TODO
    
    columnsValues = dataframe[attribute].value_counts().to_dict()
    vals = np.unique(dataframe[attribute], return_counts=False)
    
    size = len(dataframe)
    avg = 0
    for currVal in vals:
        temp = (get_entropy_of_dataset(dataframe[dataframe[attribute] == currVal]))
        avg = avg + ((columnsValues[currVal]/size)*temp)
    if avg != 0:
        return avg
    else:
        return float(0)