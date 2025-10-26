import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


# Data set 1: # id:23-23--23-1 
# Data set 2: # id:23-46-23-1

# commented code splits csv file at the #, and writes them to their own unique csv files
# with open("week4.csv", "r") as f:
#     lines = f.readlines()

# splitIndex = [i for i, line in enumerate(lines) if line.startswith("#")][1]

# data1 = "".join(lines[1:splitIndex])
# data2 = "".join(lines[splitIndex + 1:])

# with open("set1.csv", "w") as f:
#     f.write(data1)

# with open("set2.csv", "w") as f:
#     f.write(data2)

# read in data from csv files
df1 = pd.read_csv("set1.csv")
df2 = pd.read_csv("set2.csv")

print(df1.head())
print(df2.head())

# separate out each column for df1
X1D1 = df1.iloc[:,0]
X2D1 = df1.iloc[:,1]
XD1 = np.column_stack((X1D1, X2D1))
YD1 = df1.iloc[:,2]

# separate out each column for df2
X1D2 = df2.iloc[:,0]
X2D2 = df2.iloc[:,1]
XD2 = np.column_stack((X1D2, X2D2))
YD2 = df2.iloc[:,2]

xConcD1 = pd.concat([X1D1, X2D1], axis = "columns") # concatenate X1 and X2 from dataset 1
xConcD2 = pd.concat([X1D2, X2D2], axis = "columns") # concatenate X1 and X2 from dataset 2

x1D1Pos = []
x1D1Neg = []
x2D1Pos = []
x2D1Neg = []

idx = 0
for i in range(len(YD1)):
    if(YD1.iloc[idx] == 1):
        x1D1Pos.insert(idx, X1D1.iloc[idx])
        x2D1Pos.insert(idx, X2D1.iloc[idx])
        idx += 1
    else:
        x1D1Neg.insert(idx, X1D1.iloc[idx])
        x2D1Neg.insert(idx, X2D1.iloc[idx])
        idx += 1

x1D2Pos = []
x1D2Neg = []
x2D2Pos = []
x2D2Neg = []

idx2 = 0
for i2 in range(len(YD2)):
    if(YD2.iloc[idx2] == 1):
        x1D2Pos.insert(idx2, X1D2.iloc[idx2])
        x2D2Pos.insert(idx2, X2D2.iloc[idx2])
        idx2 += 1
    else:
        x1D2Neg.insert(idx2, X1D2.iloc[idx2])
        x2D2Neg.insert(idx2, X2D2.iloc[idx2])
        idx2 += 1

# plot dataset 1 
plt.scatter(x1D1Pos, x2D1Pos, c = "red", marker = "+", label = "+1")
plt.scatter(x1D1Neg, x2D1Neg, c = "blue", marker = "_", label = "-1")
plt.xlabel("Feature: X1")
plt.ylabel("Feature: X2")
plt.title("Plot for Dataset 1")
plt.legend()
plt.show()

# plot dataset 2
plt.scatter(x1D2Pos, x2D2Pos, c = "yellow", marker = "+", label = "+1")
plt.scatter(x1D2Neg, x2D2Neg, c = "purple", marker = "_", label = "-1")
plt.xlabel("Feature: X1")
plt.ylabel("Feature: X2")
plt.title("Plot for Dataset 2")
plt.legend()
plt.show()

# i) a) augment data set w/ polynomial features, do cross validation & train Logistic regression
# coded with help from: https://www.mygreatlearning.com/blog/gridsearchcv/
polyOrders = [1, 2, 3, 4, 5]
cVals = [0.001, 0.1, 1, 10, 100]
meanErr = []
stdErr = []

def findBestCDeg(xConcat, YLabl):
    paramGrid = {
                "poly__degree": [1, 2, 3, 4, 5],
                "logReg__C": [0.001, 0.1, 1, 10, 100],
                "logReg__penalty": ["l2"]
            }
    pipeline = Pipeline([
            ("poly", PolynomialFeatures()),
            ("logReg", LogisticRegression())
        ])
    
    grid = GridSearchCV (
        estimator = pipeline,
        param_grid = paramGrid,
        scoring = "f1",
        cv = 5
    )
    
    grid.fit(xConcat, YLabl)
    print("Best Params: ", grid.best_params_)
    print("Best F1 Score: ", grid.best_score_, "\n")

    yPred = grid.predict(xConcat)
    print(classification_report(YLabl, yPred))

    meanF1 = grid.cv_results_["mean_test_score"]
    stdF1 = grid.cv_results_["std_test_score"]
    params = grid.cv_results_["params"]

    for deg in polyOrders:
        degMeans = [mean for mean, param in zip(meanF1, params) if param["poly__degree"] == deg]
        degSTDs = [std for std, param in zip(stdF1, params) if param["poly__degree"] == deg]
        plt.errorbar(cVals, degMeans, yerr = degSTDs, label = f"Degree: {deg}")
    
    bestParams = grid.best_params_
    bestC =  bestParams["logReg__C"]
    bestDeg = bestParams["poly__degree"]
    bestScore = grid.best_score_

    plt.scatter(bestC, bestScore, color = "red", s = 100, label = "Best Model")
    plt.xlabel("C Values")
    plt.ylabel("Mean F1 Scores +- std")
    plt.xscale("log")
    plt.title("Cross Validation F1 Score vs C for different Polynomial Degrees")
    plt.legend()
    plt.show()

    bestModel = grid.best_estimator_
    return bestModel, bestC, bestDeg

# i) b) train kNN Classifier on Data
# coded with help from: https://www.geeksforgeeks.org/machine-learning/understanding-decision-boundaries-in-k-nearest-neighbors-knn/
kNNVals = [1, 5, 10, 15, 20, 25]
kf = KFold(n_splits=5, shuffle= True, random_state=42)

def trainKNN(xConcat, YLabl):
    meanF1Scores = []
    stdF1Scores = []
    for k in kNNVals:
        f1Scores = []
        for train, test in kf.split(xConcat):
            xTrain, xTest = xConcat.iloc[train], xConcat.iloc[test]
            yTrain, yTest = YLabl.iloc[train], YLabl.iloc[test]

            kNNModel = KNeighborsClassifier(n_neighbors = k)
            kNNModel.fit(xTrain, yTrain)
            yPred = kNNModel.predict(xTest)

            f1Scores.append(f1_score(yTest, yPred))
        meanF1Scores.append(np.mean(f1Scores))
        stdF1Scores.append(np.std(f1Scores))

        xMinK, xMaxK = xConcat.iloc[:,0].min() - 0.1, xConcat.iloc[:,0].max()
        yMinK, yMaxK = xConcat.iloc[:,1].min() - 0.1, xConcat.iloc[:,1].max()

        # plot decision boundaries for each k val
        xxK, yyK = np.meshgrid(np.linspace(xMinK, xMaxK),
                             np.linspace(yMinK, yMaxK))
        
        ZK = kNNModel.predict(np.c_[xxK.ravel(), yyK.ravel()])
        ZK = ZK.reshape(xxK.shape)
        
        plt.contourf(xxK, yyK, ZK)
        plt.scatter(xConcat.iloc[:,0], xConcat.iloc[:,1], c = YLabl)
        plt.title(f"kNN Decision Boundary (k = {k})")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

    bestIdx = np.argmax(meanF1Scores)
    bestK = kNNVals[bestIdx]
    print("Best k: ", bestK, "\n")
    bestScore = meanF1Scores[bestIdx]


    bestkNNModel = KNeighborsClassifier(n_neighbors=bestK).fit(xConcat, YLabl)


    xMin, xMax = xConcat.iloc[:,0].min() - 0.1, xConcat.iloc[:,0].max()
    yMin, yMax = xConcat.iloc[:,1].min() - 0.1, xConcat.iloc[:,1].max()

    xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.01),
                            np.arange(yMin, yMax, 0.01))
    
    Z = bestkNNModel.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape) 

    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, Z, alpha = 0.3)
    plt.scatter(xConcat.iloc[:,0], xConcat.iloc[:,1], c = YLabl)
    plt.title(f"Best kNN Decision Boundary (k = {bestK})")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    return bestkNNModel, bestK

# i) c)
def calcConfMatrix(xConcat, YLabl, polyOrder, cVal, kVal):
    # confusion matrix for Logistic Regression
    xTrain, xTest, yTrain, yTest = train_test_split(xConcat, YLabl, test_size = 0.2, random_state=42)
    poly = PolynomialFeatures(polyOrder)
    xTrainPoly = poly.fit_transform(xTrain)
    xTestPoly = poly.transform(xTest)
    logRegModel = LogisticRegression(penalty="l2", C = cVal, random_state=42).fit(xTrainPoly, yTrain)
    yPredLR = logRegModel.predict(xTestPoly)

    print("Logistic Regression\n")
    print("Confusion Matrix", confusion_matrix(yTest, yPredLR))
    print("Classification Report", classification_report(yTest, yPredLR))

    dummyLR = DummyClassifier(strategy= "stratified").fit(xTrainPoly, yTrain)
    dumPredLR = dummyLR.predict(xTestPoly)

    print("Dummy Matrix: ", confusion_matrix(yTest, dumPredLR))
    print("Dummy Report: ", classification_report(yTest, dumPredLR))

    # confusion matrix for kNN
    kNNModel = KNeighborsClassifier(n_neighbors=kVal).fit(xTrain, yTrain)
    kPred = kNNModel.predict(xTest)

    print("\nkNN Classifier\n")
    print("Confusion Matrix: ", confusion_matrix(yTest, kPred))
    print("Classification Report: ", classification_report(yTest, kPred))

    dummyK = DummyClassifier(strategy= "stratified").fit(xTrain, yTrain)
    dumPredK = dummyK.predict(xTest)

    print("Dummy Matrix: ", confusion_matrix(yTest, dumPredK))
    print("Dummy Report: ", classification_report(yTest, dumPredK))

# i) d) Plot ROC curves for Logistic Regression & kNN
def plotROC(xConcat, YLabl, polyOrder, cVal, kVal):
    plt.rc("font", size = 18)
    plt.rcParams["figure.constrained_layout.use"] = True
    xTrain, xTest, yTrain, yTest = train_test_split(xConcat, YLabl, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(polyOrder)
    xTrainPoly = poly.fit_transform(xTrain)
    xTestPoly = poly.transform(xTest)
    logRegModel = LogisticRegression(penalty = "l2", C = cVal, random_state=42).fit(xTrainPoly, yTrain)

    kNNModel = KNeighborsClassifier(n_neighbors=kVal).fit(xTrain, yTrain)
    kPred = kNNModel.predict(xTest)


    fprl, tprl, _ = roc_curve(yTest, logRegModel.decision_function(xTestPoly))
    plt.plot(fprl, tprl)
    plt.title("ROC Curve for Logistic Regression")
    plt.xlabel("False Pos Rate")
    plt.ylabel("True Pos Rate")
    plt.plot([0, 1], [0, 1], color = "green")
    plt.show()

    fprk, tprk, _ = roc_curve(yTest, kNNModel.predict_proba(xTest)[:,1])
    plt.plot(fprk, tprk)
    plt.title("ROC Curve for kNN")
    plt.xlabel("False Pos Rate")
    plt.ylabel("True Pos Rate")
    plt.plot([0, 1], [0, 1], color = "green")
    plt.show()



d1BestLRModel, d1BestC, d1BestPoly = findBestCDeg(xConcD1, YD1)
# print(f"Best Model: {d1BestLRModel}\n, Best C: {d1BestC}\n, Best Polynomial: {d1BestPoly}") # debugging
d1BestkNN, bestK = trainKNN(xConcD1, YD1)
#print("Best kNN: ", d1BestkNN, "Best k ValueL", bestK) # debugging
d1ConfMatx = calcConfMatrix(xConcD1, YD1, d1BestPoly, d1BestC, bestK)
d1ROC = plotROC(xConcD1, YD1, d1BestPoly, d1BestC, bestK)

d2BestLRModel , d2BestC, d2BestPoly = findBestCDeg(xConcD2, YD2)
d2BestkNN, d2BestK = trainKNN(xConcD2, YD2)
d2ConfMatx = calcConfMatrix(xConcD2, YD2, d2BestPoly, d2BestC, d2BestK)
d2ROC = plotROC(xConcD2, YD2, d2BestPoly, d2BestC, d2BestK)