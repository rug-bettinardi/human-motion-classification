import os
import random

import scipy as sp
import numpy as np
import pandas as pd
import antropy as ant
import seaborn as sns
import matplotlib.pyplot as plt

from tsfresh import extract_relevant_features

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score


# define helper functions:
def convertToRealValue(x):
    return -14.709 + (x/63)*(2*14.709)

def loadOne(filePath):

    data = pd.read_csv(filePath, sep=" ", header=None)

    return data

def preProcessOne(dfRaw):

    dfRaw.columns = ["x", "y", "z"]
    dfRaw.index = dfRaw.index / fs  # convert to seconds

    # convert mapped data to accelerometer real-values
    dfReal = dfRaw.applymap(lambda x: convertToRealValue(x))

    # apply median filter to smooth possible artifacts:
    dfSmooth = dfReal.rolling(3, center=True).median()

    return dfSmooth.dropna()

def plotOne(df, title=None, ax=None):

    df.plot(ax=ax)
    plt.xlabel("time [s]")
    plt.ylabel("acceleration [$m/s^2$]")

    if title:
        plt.title(title)

def slopeAndIntercept(x, y):

    slope, intercept, _, _, _ = sp.stats.linregress(x, y)

    return slope, intercept

def featuresExtraction(df, fs):

    # compute features
    featuresDict = {
        "mean": df.mean(skipna=True).to_dict(),
        "median": df.median(skipna=True).to_dict(),
        "mode": df.mode().T.to_dict()[0],
        "std": df.std(skipna=True).to_dict(),
        "sem": df.sem(skipna=True).to_dict(),
        "sum": df.sum(skipna=True).to_dict(),
        "min": df.min(skipna=True).to_dict(),
        "max": df.max(skipna=True).to_dict(),
        "kurtosis": df.kurtosis(skipna=True).to_dict(),
        "skewness": df.skew(skipna=True).to_dict(),
        "mad": df.mad(skipna=True).to_dict(),
        "IQR": (df.quantile(q=0.75) - df.quantile(q=0.25)).to_dict(),
        "lineLength": df.diff().abs().sum().to_dict(),
        "energy": df.abs().applymap(lambda x: x**2).sum().to_dict(),
        "permEntropy": {k: ant.perm_entropy(df[k], normalize=True) for k in ['x', 'y', 'z']},
        "specEntropy": {k: ant.spectral_entropy(df[k], sf=fs, method='welch', normalize=True) for k in ['x', 'y', 'z']},
        "svdEntropy": {k: ant.svd_entropy(df[k], normalize=True) for k in ['x', 'y', 'z']},
        "sampleEntropy": {k: ant.sample_entropy(df[k]) for k in ['x', 'y', 'z']},
        "hjorthMobility": {k: ant.hjorth_params(df[k])[0] for k in ['x', 'y', 'z']},
        "hjorthComplexity": {k: ant.hjorth_params(df[k])[1] for k in ['x', 'y', 'z']},
        "petrosianFractDim": {k: ant.petrosian_fd(df[k]) for k in ['x', 'y', 'z']},
        "higuchiFractDim": {k: ant.higuchi_fd(df[k]) for k in ['x', 'y', 'z']},
        "dfa": {k: ant.detrended_fluctuation(df[k]) for k in ['x', 'y', 'z']},
        "cv": {k: sp.stats.variation(df[k], nan_policy='omit') for k in ['x', 'y', 'z']},
        "shapiro": {k: sp.stats.shapiro(df[k])[0] for k in ['x', 'y', 'z']},
        "slope": {k: slopeAndIntercept(x=df.index, y=df[k])[0] for k in ['x', 'y', 'z']},
        "intercept": {k: slopeAndIntercept(x=df.index, y=df[k])[1] for k in ['x', 'y', 'z']},
    }

    stretchedFeaturesDict = {f"{feat}_{ch}": featuresDict[feat][ch] for feat in featuresDict for ch in ['x', 'y', 'z']}

    return pd.Series(stretchedFeaturesDict)


    # # vectorize:
    # featuresVector = []
    # for feature in featuresDict:
    #     for chan in featuresDict[feature]:
    #
    #         if isinstance(featuresDict[feature][chan], list):
    #             featureVal = featuresDict[feature][chan]
    #         else:
    #             featureVal = [featuresDict[feature][chan]]
    #
    #         featuresVector = featuresVector + featureVal
    #
    # return featuresVector

def buildFeaturesDataMatrix(srcDir):

    activities = os.listdir(srcDir)

    allRecsVectors = []
    for activity in activities:
        for recName in os.listdir(os.path.join(srcDir, activity)):
            print(f"{activity}: {recName}")
            df = loadOne(filePath=os.path.join(srcDir, activity, recName))
            df = preProcessOne(dfRaw=df)
            ftsVector = featuresExtraction(df, fs)
            ftsVector = ftsVector.append(pd.Series({"activity": activity}))
            allRecsVectors.append(ftsVector)

    dfDataMatrix = pd.concat(allRecsVectors, axis=1, ignore_index=True).T

    print("== buildFeaturesDataMatrix: COMPLETE == ")

    return dfDataMatrix

def buildFeaturesDataMatrixUsingTSFresh(srcDir):

    """
    pay attention: tsfresh SORTS the column_id values!
    therefore the resulting DataFrame typically does not match with the order of true labels ...
    SOLUTION: convert ID values of each time series into integer, so that even sorting would not affect the results
    """

    activities = os.listdir(srcDir)

    dfRecordsLst, recNamesLst, recIdLst, recTrueActivityLst = [], [], [], []
    recID = 0
    for activity in activities:
        for recName in os.listdir(os.path.join(srcDir, activity)):
            print(f"{activity}: {recName}")
            recNamesLst.append(recName)
            recIdLst.append(recID)
            recTrueActivityLst.append(activity)
            df = loadOne(filePath=os.path.join(srcDir, activity, recName))
            df = preProcessOne(dfRaw=df)
            df['recordName'] = recName
            df['recordID'] = recID
            df['activity'] = activity
            df['time'] = df.index
            df.index = [x for x in range(len(df))]
            dfRecordsLst.append(df)
            recID += 1

    dfAllRecs = pd.concat(dfRecordsLst, axis=0, ignore_index=True)

    # prepare data for tsfresh, extract relevant features, append activity label:
    dfTimeseries = dfAllRecs.drop(labels=['recordName', 'activity'], axis=1)
    serTrueLabels = pd.Series(recTrueActivityLst, index=recIdLst)
    dfDataMatrix = extract_relevant_features(dfTimeseries, serTrueLabels, column_id="recordID", column_sort="time")
    dfDataMatrix['activity'] = recTrueActivityLst

    return dfDataMatrix

def plotTSNE(X, y):

    tSNE = TSNE(learning_rate=200)
    transformed = tSNE.fit_transform(X)
    dfTSNE = pd.DataFrame({'tSNE_1': transformed[:, 0], 'tSNE_2': transformed[:, 1], 'activity': y})
    sns.set_style("white")
    plt.figure()
    sns.scatterplot(data=dfTSNE, x='tSNE_1', y='tSNE_2', hue='activity')
    plt.xlabel('t-SNE 1-st component'), plt.ylabel('t-SNE 2-nd component')
    plt.title("t-distributed Stochastic Neighbors Embedding", fontweight='bold')

def plotPricipalComponents(samples):

    pca = PCA()
    pca.fit(samples)

    features = range(pca.n_components_)

    plt.figure()
    plt.bar(features, pca.explained_variance_, width=1, color='grey', edgecolor='w')
    plt.xlabel('principal components')
    plt.ylabel('explained variance')
    plt.title('Principal Components Analysis', fontweight='bold')

def evalClassifiers(X, y, classifiersLst, metric='f1_macro', cv=10):

    for clf in classifiersLst:

        cvScores = cross_val_score(clf, X, y, cv=cv, scoring=metric)
        print(f"{cv}-fold CV F1-Score: {round(np.mean(cvScores), 3)},  Classifier: {clf.__class__.__name__}")

def plotConfusionMatrix(y_test, y_pred):

    df = pd.DataFrame({"predicted_label": y_pred, "true_label": y_test})
    ct = pd.crosstab(df['predicted_label'], df['true_label'])

    kwsHeat = {
        'annot': True,
        'cbar': False,
        'square': True,
        'xticklabels': True,
        'yticklabels': True,
        'linewidths': .1,
        'fmt': 'd',
        'cmap': 'YlGnBu',  # 'cool',  #'Blues',
    }

    plt.figure()
    ax = sns.heatmap(ct, **kwsHeat)
    ax.tick_params(axis='both', which='both', labelsize=8)
    ax.set_xlabel('True Label', fontweight='bold')
    ax.set_ylabel('Predicted Label', fontweight='bold')

def f1ScoreClassifiers(X_train, X_test, y_train, y_test, classifiersLst, confusionMatrix=False):

    for clf in classifiersLst:

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1Score = f1_score(y_true=y_test, y_pred=y_pred, average="macro")

        print(f"F1-Score: {round(f1Score, 3)},  Classifier: {clf.__class__.__name__}")

        if confusionMatrix:

            plotConfusionMatrix(y_test, y_pred)
            plt.title(f"{clf.__class__.__name__}")

def f1ScoreEnsembleClassifier(X_train, X_test, y_train, y_test, classifiersTuplesLst):

    clf = StackingClassifier(
        estimators=classifiersTuplesLst,
        final_estimator=GradientBoostingClassifier()
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1Score = f1_score(y_true=y_test, y_pred=y_pred, average="macro")

    print(f"Ensemble Classifier F1-Score: {round(f1Score, 3)}")


if __name__ == "__main__":

    # define general variables:
    srcDir = r"C:\CodeRug\biometricData\accelerometer\humanmotionprimitives\data"
    fs = 32
    
    # get list of classes:
    activities = os.listdir(srcDir)

    # plot 4 records as example:
    sns.set_style("white")
    for i, activity in enumerate(['climb_stairs', 'comb_hair', 'drink_glass', 'walk']):
        fileName = random.choice(os.listdir(os.path.join(srcDir, activity)))
        df = loadOne(filePath=os.path.join(srcDir, activity, fileName))
        df = preProcessOne(dfRaw=df)
        ax = plt.subplot(2, 2, i + 1)
        plotOne(df, ax=ax)
        ax.set_title(activity, fontweight='bold')
        ax.set_xlabel('')

    # load all time series, extract features, store all in a matrix 
    dataMatrix = buildFeaturesDataMatrix(srcDir)

    # check support for each class:
    dataMatrix['activity'].groupby(dataMatrix['activity']).count()

    # prepare data:
    trueLabels = dataMatrix['activity']
    fts = dataMatrix.drop(labels='activity', axis=1).astype(float)
    print(fts.describe())
    scaler = StandardScaler()
    scaledFts = pd.DataFrame(data=scaler.fit_transform(fts), columns=fts.columns)
    scaledFts.describe()

    # t-SNE:
    plotTSNE(X=scaledFts, y=trueLabels.values)

    # split fts and labels into train and test set (stratifying by label as classes are unbalanced)
    X_train, X_test, y_train, y_test = train_test_split(scaledFts, trueLabels, test_size=0.3, random_state=42, stratify=trueLabels)

    # instantiate classifiers:
    classifiers = [
        KNeighborsClassifier(n_neighbors=len(activities)),
        SVC(kernel="rbf"),
        RandomForestClassifier(n_estimators=500, random_state=42),
        RidgeClassifier(max_iter=500),
        LogisticRegression(max_iter=500),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
    ]

    # run classification and return F1-score of each classifier:
    f1ScoreClassifiers(X_train, X_test, y_train, y_test, classifiers, confusionMatrix=True)

    # build an ensemble classifier:
    classifiersToStack = [
        ('SVC', SVC(kernel="rbf")),
        ('RFC', RandomForestClassifier(n_estimators=500, random_state=42)),
        ('RC', RidgeClassifier(max_iter=500)),
        ('LR', LogisticRegression(max_iter=500)),
        ('LDA', LinearDiscriminantAnalysis()),
    ]
    f1ScoreEnsembleClassifier(X_train, X_test, y_train, y_test, classifiersTuplesLst=classifiersToStack)


    ## now, repeat the analysis but using the time series features automatically extracted using tsfresh:

    dataMatrix = buildFeaturesDataMatrixUsingTSFresh(srcDir)

    # check support for each class:
    dataMatrix['activity'].groupby(dataMatrix['activity']).count()

    # prepare data:
    trueLabels = dataMatrix['activity']
    fts = dataMatrix.drop(labels='activity', axis=1).astype(float)
    print(fts.shape)
    print(fts.columns)
    scaler = StandardScaler()
    scaledFts = pd.DataFrame(data=scaler.fit_transform(fts), columns=fts.columns)

    # t-SNE:
    plotTSNE(X=scaledFts, y=trueLabels.values)

    # split fts and labels into train and test set (stratifying by label as classes are unbalanced)
    X_train, X_test, y_train, y_test = train_test_split(scaledFts, trueLabels, test_size=0.3, random_state=42, stratify=trueLabels)

    # instantiate classifiers:
    classifiers = [
        KNeighborsClassifier(n_neighbors=len(activities)),
        SVC(kernel="rbf"),
        RandomForestClassifier(n_estimators=500, random_state=42),
        RidgeClassifier(max_iter=500),
        LogisticRegression(max_iter=500),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
    ]

    # run classification and return F1-score of each classifier:
    f1ScoreClassifiers(X_train, X_test, y_train, y_test, classifiers, confusionMatrix=True)

    # build an ensemble classifier:
    classifiersToStack = [
        ('SVC', SVC(kernel="rbf")),
        ('RFC', RandomForestClassifier(n_estimators=500, random_state=42)),
        ('RC', RidgeClassifier(max_iter=500)),
        ('LR', LogisticRegression(max_iter=500)),
        ('LDA', LinearDiscriminantAnalysis()),
    ]
    f1ScoreEnsembleClassifier(X_train, X_test, y_train, y_test, classifiersTuplesLst=classifiersToStack)





