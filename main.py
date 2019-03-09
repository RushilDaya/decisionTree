from DecisionTree import DecisionTree


x = [[1,0],[1,1],[1,0],[0,0],[1,0],[0,1]]
y = [2,2,1,2,1,1]

a = DecisionTree()
a.train(x,y)