import math
import time

class DecisionTree:
    def __init__(self):
        self.learnedTree = None
    
    def train(self, x, y):
        features = [i for i in range(len(x[0]))]
        self.learnedTree = self.__trainID3(features,x,y)
        return True
        
    
    def predict(self,x):
        return self.learnedTree(x)
    
    #--- internal functions

    def __trainID3(self,features, x,y):
        time.sleep(1)
        if y == y[0]*len(y):
            def pureLeaf(test_x):
                return y
            return pureLeaf

        if len(features) == 0:
            allLabels = list(set(y))
            counts = [ self.__count(y,item) for item in allLabels]
            mostFrequent = allLabels[counts.index(max(counts))]
            def mixedLeaf(test_x):
                return mostFrequent 
            return mixedLeaf

        # compute total entropy
        currentEntropy = self.__computeEntropy(y)
        splitEntropies = [self.__getSplitEntropy(feature, x, y) for feature in features]
        informationGain = [ (currentEntropy - splitEntropies[i]) for  i in range(len(splitEntropies))]


        bestFeature = informationGain.index(max(informationGain))
        uniqueFeatureValues = list(set([row[bestFeature] for row in  x]))
        xDivides = [self.__genX(featureValue,bestFeature,x) for featureValue in uniqueFeatureValues]
        yDivides = [self.__genY(featureValue,bestFeature,x,y) for featureValue in uniqueFeatureValues]
        reducedFeatureSet = [item for item in features if item != bestFeature]

        subTrees = [self.__trainID3(reducedFeatureSet, xDivides[i], yDivides[i]) for i in range(len(uniqueFeatureValues))]


        def predictInternal(test_x):
            for i in range(len(uniqueFeatureValues)):
                if test_x[bestFeature] == uniqueFeatureValues[i]:
                    return subTrees[i](test_x)
            
        return predictInternal

        
                
    #--- helper functions
    def __genX(self,featureValue,feature,x):
        newX = []
        for i in range(len(x)):
            if x[i][feature]==featureValue:
                newX.append(x[i])
        return newX

    def __genY(self,featureValue, feature, x, y):
        newY = []
        for i in range(len(x)):
            if x[i][feature]==featureValue:
                newY.append(y[i])
        return newY

    def __count(self, listA, itemToCount):
        values = [True for item in listA if item == itemToCount]
        return len(values)
    
    def __computeEntropy(self,listA):
        allLabels = list(set(listA))
        listLength = len(listA)
        totalEntropy = 0
        for label in allLabels:
            classCount = self.__count(listA, label)
            classEntropy = -1*(float(classCount)/listLength)*math.log(float(classCount)/listLength,2)
            totalEntropy +=classEntropy
        return totalEntropy

    def __getSplitEntropy(self, feature, x, y):
        # first we must do the split and then compute the entropy for the
        # two sides
        
        def compare(x,y):
            return x == y
        
        def generateIndexArray(key, Row):
            return [compare(key, item ) for item in Row]

        def generateYSets(allY, indexBools):
            filteredSet = []
            for i in range(len(indexBools)):
                if indexBools[i] == True:
                    filteredSet.append(allY[i])
            return filteredSet

        featureRow = [row[feature] for row in x]
        uniques = list(set(featureRow))
        numUniques = len(uniques)

        uniquesIndexArray = [generateIndexArray(unique,featureRow) for unique in uniques]
        ySets = [generateYSets(y, indexBools) for indexBools in uniquesIndexArray ]
        entropies = [self.__computeEntropy(ySet) for ySet in ySets]
        
        splitEntropy = 0
        for i in range(len(entropies)):
            ent = entropies[i]
            size = len(ySets)
            splitEntropy += (float(size)/len(y))*ent 
        
        return splitEntropy

            
