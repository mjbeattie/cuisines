"""
cuisines.py
This script creates clusters of cuisines from a database of recipes.  It starts
by reading all the recipes and then creates a set of relatively common ingredients.
These ingredients then become the dimensions of the vectors used to cluster cuisines.
The script then takes the recipes and groups them by tagged cuisine.  After this,
a centroid is defined that is the midpoint of the tagged cuisines.

Created on Wed Mar 25 19:29:00 2018

@author:  Matthew J. Beattie, OU, DSA5970
coding = utf-8
"""

import logging
import importlib
import sys
import io
import ujson
from typing import List
import numpy
from random import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
import joblib
from collections import defaultdict
import pprint
from operator import itemgetter
import argparse

importlib.reload(logging)  # Stops logging reloads in Python

# Setup logging for program
log = logging.getLogger("CUISINES")
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s %(message)s")
ch.setFormatter(formatter)
log.addHandler(ch)
log.info("Now logging")


def buildFeatureSets(recipes, listIngredients, n1=1000, n2=200):
    """
    buildFeatures() creates a set of vectorized features from the recipes data.
    It takes the ingredients as a set and uses those as the dimensions of binary
    vectors for the training and test datasets.
    :param recipes: a set of recipes read in from yummly.json
    :param n1: the number of training recipes
    :param n2: the number of test recipes
    :return: training features and test features.  The program also pickles the
    two feature sets.
    """

    if n1 > len(recipes) or n1 <= 0:
        print("Invalid number of training recipes, using default...")
        n1 = 1000
    if n2 > len(recipes) or n2 >= (len(recipes)-n1):
        print("Invalid value for number of test recipes, using default...")
        n2 = min(len(recipes-n1-1),200)

    print("n1=", n1, " n2=", n2)

    # Create a training set of data for the classifiers
    trainDict = recipes[0:n1]
    log.info("Extracting features and labels from training data...")
    trainFeatures = []
    trainLabels = []
    trainID = []
    counter = 0
    for recipe in trainDict:
        if counter % 100 == 0:
            log.info("Evaluating recipe number " + str(counter) + " of " + str(len(trainDict)))

        # Construct feature vector of 1s and 0s indicating presence of an ingredient
        ingrFeature = numpy.zeros(len(listIngredients))
        i = 0
        for ingredient in listIngredients:
            if ingredient in recipe.get('ingredients'):
                ingrFeature[i] = 1
            else:
                ingrFeature[i] = 0
            i += 1

        # Add feature vector to features list
        trainFeatures.append(ingrFeature)
        trainLabels.append(recipe.get('cuisine'))
        trainID.append(recipe.get('id'))
        counter += 1

        # Save training data to a pickle file
        joblib.dump([listIngredients, trainFeatures, trainLabels, trainID], 'mytraindata.pkl')

    log.info("Completed creation of training features and labels")

    # Create a test set of data for the classifiers
    testDict = recipes[n1:(n1+n2)]
    log.info("Extracting features and labels from test data...")
    testFeatures = []
    testLabels = []
    testID = []
    counter = 0
    for recipe in testDict:
        if counter % 100 == 0:
            log.info("Evaluating recipe number " + str(counter) + " of " + str(len(testDict)))

        # Construct feature vector of 1s and 0s indicating presence of an ingredient
        ingrFeature = numpy.zeros(len(setIngredients))
        i = 0
        for ingredient in listIngredients:
            if ingredient in recipe.get('ingredients'):
                ingrFeature[i] = 1
            else:
                ingrFeature[i] = 0
            i += 1

        # Add feature vector to features list
        testFeatures.append(ingrFeature)
        testLabels.append(recipe.get('cuisine'))
        testID.append(recipe.get('id'))
        counter += 1

        # Save training data to a pickle file
        joblib.dump([listIngredients, testFeatures, testLabels, testID], 'mytestdata.pkl')

    log.info("Completed creation of test features and labels")

    return ([[listIngredients, trainFeatures, trainLabels, trainID],
             [listIngredients, testFeatures, testLabels, testID]])


def buildClassifier(trainData, testData):
    """
    buildClassifier() construcs a Naive Bayes and Decision Tree classifier
    from a set of training data.  It outputs the accuracy of the two classifiers
    and returns the best of the two.
    :param trainData: a set of vectorized training data
    :param testData: a set of vectorized test data
    :return: a scikit classifier
    """
    # Run SciKit Naive Bayes classification and analyze results
    # Train Naive Bayes classifier
    log.info("Building Bayes classifier...")
    bayesclf = MultinomialNB().fit(tr[1], tr[2])
    log.info("Bayes classifier complete")

    # Predict results based upon Naive Bayes classifier and check accuracy
    bayespredict = bayesclf.predict(te[1])
    bayesscore = bayesclf.score(te[1], te[2])
    print("The accuracy of the Naive Bayes classifier is: " + str(bayesscore))

    # Run Decision Tree classification and analyze results
    # Train Decision Tree classifier
    log.info("Building Decision Tree classifier...")
    dtclf = tree.DecisionTreeClassifier().fit(tr[1], tr[2])
    log.info("Decision Tree classifier complete")

    # Predict results based upon Decision Tree classifier and check accuracy
    dtpredict = dtclf.predict(te[1])
    dtscore = dtclf.score(te[1], te[2])
    print("The accuracy of the Decision Tree classifier is: " + str(dtscore))

    if dtscore > bayesscore:
        print("The Decision Tree classifier was more accurate -- returning that one.")
        clf = dtclf
    else:
        print("The Bayes classifier was more accurate -- returning that one.")
        clf = bayesclf

    # Return the best classifier to the calling routine and store in a pickle file
    joblib.dump(clf, 'myclf.pkl')
    return clf


def findCuisine(refrig, listIngredients, clf):
    """
    findCuisine() takes a list of ingredients supplied by the user and returns
    the most likely cuisine that should be created.
    :param refrig: a list of ingredients
    :param listIngredients: the set of ingredients from the classifier
    :param clf: the classifier
    :return: a sorted list of cuisines and probability of matching refrigerator
    """
    # Vectorize the user's refrigerator by comparing to listIngredients
    ingrVector = numpy.zeros(len(listIngredients))
    for item in refrig:
        i = 0
        for ingredient in listIngredients:
            if item == ingredient:
                ingrVector[i] = 1
            i += 1

    # Use classifier to determine probability of cuisines
    r = clf.predict_proba([ingrVector])
    c = clf.classes_

    # Create list of tuples with cuisine and probabilities
    cuisprob = []
    probs = r[0]
    for i in range(0, len(probs) - 1):
        cuisprob.append([clf.classes_[i], round(probs[i], 4)])

    # Sort cuisines in descending order of probability
    def getKey(item):
        return (item[1])

    l = sorted(cuisprob, key=getKey, reverse=True)

    # Return sorted list of cuisines and probabilities
    return (l)


def findNeighbors(refrig, listIngredients, trainFeatures, trainID):
    """
    findNeighbors() takes a list of ingredients supplied by the user and returns
    the five closest recipes.
    :param refrig: a list of ingredients
    :param listIngredients: the set of ingredients from the training data
    :param trainFeatures: the training data features
    :param trainID:  the list of recipe IDs in the training data
    :return: a sorted list of recipes and distance from the refrigerator
    """
    # Vectorize the user's refrigerator by comparing to listIngredients
    ingrVector = numpy.zeros(len(listIngredients))
    for item in refrig:
        i = 0
        for ingredient in listIngredients:
            if item == ingredient:
                ingrVector[i] = 1
            i += 1

    # Use sklearn nearest neighbors routine to find five nearest recipes
    neigh = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine')
    neigh.fit(trainFeatures)
    n = neigh.kneighbors([ingrVector], 5)

    # Create list of tuples with recipes and distances
    closerecipes = []
    dists = n[0][0]
    ids = n[1][0]
    for i in range(0, len(dists)):
        closerecipes.append([ids[i], round(dists[i], 4)])

    # Sort cuisines in descending order of probability
    def getKey(item):
        return (item[1])

    l = sorted(closerecipes, key=getKey)

    # Return sorted list of recipes and distances
    return (l)


def printCuisinesClass(cuisinelist):
    """
    printCuisineClass() prints out the top five most likely cuisines
    :param cuisinelist: list of (cuisine,probability) tuples
    :return: nothing returned
    """
    l = cuisinelist[0:5]
    print("\nThe top five most likely cuisines are:")
    for tuple in l:
        print("Cuisine:  " + str(tuple[0]) + "     Probability:  " + str(tuple[1]))
    return


def printCuisinesClose(neighlist):
    """
    printCuisineClose() prints out the top five most likely recipes
    :param cuisinelist: list of (recipe,distance) tuples
    :return: nothing returned
    """
    print("\nThe top five most likely recipes are:")
    for tuple in neighlist:
        print("Recipe:  " + str(tuple[0]) + "     Distance:  " + str(tuple[1]))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # The --input flag defaults to the current directory unless another one is specified
    parser.add_argument("-f", "--features", help="Build training and test features",
                        action="store_true")
    parser.add_argument("-c", "--classifier", help="Build classifier", action="store_true")
    parser.add_argument("-p", "--pickles", help="Use pickle files for features or classifier",
                        action="store_true")
    parser.add_argument("-d", "--docs", type=str, help="Directory containing yummly.json",
                        default="./")
    parser.add_argument("--traincount", type=int, help="Number of training recipes, default=1000",
                        default=1000)
    parser.add_argument("--testcount", type=int, help="Number of test recipes, default=200",
                        default=200)
    parser.add_argument("-i", "--ingredient", help="An ingredient that you have", default=[],
                          action="append", dest="ingredient")
    args = parser.parse_args()

    # Read recipes into memory
    recipefile = args.docs + "yummly.json"

    print("Reading recipe files from ", recipefile)
    try:
        recipes = ujson.loads(open(recipefile, "r").read())
    except:
        print("File read error, exiting...")
        sys.exit(1)

    # Create statistics for number of recipes and number of ingredients.  Also
    # create a set of unique ingredients
    cuisines = defaultdict(int)
    ingredients = defaultdict(int)
    setIngredients = set()
    shuffle(recipes)
    for recipe in recipes:
        cuisines[recipe['cuisine']] += 1
        for ingr in recipe['ingredients']:
            ingredients[ingr] += 1
            setIngredients.add(ingr)

    # Place unique ingredients into a list to allow index referencing
    listIngredients = []
    for ingr in setIngredients:
        listIngredients.append(ingr)

    # Pretty print cuisines and ingredients and statistics
    #pprint.pprint(cuisines)
    #pprint.pprint(ingredients)
    print("There are {} cuisines from {} recipes.".format(len(cuisines), len(recipes)))
    print("There are {} unique ingredients.".format(len(ingredients)))

    # Check to see if ingredients are included in the yummly set
    refrig = []
    print("Your refrigerator has: ", args.ingredient, " Checking for matches with mine...")
    for item in args.ingredient:
        if item not in setIngredients:
            print("Couldn't find ", item, ". You may want to try reentering but we will skip for now.")
        else:
            refrig.append(item)
    print("Checking recipes with ", refrig)

    # Load features and training data from pickle files if applicable
    if args.pickles or (not args.features and not args.classifier):
        try:
            # Load data from pickle files
            tr = joblib.load('traindata.pkl')
            te = joblib.load('testdata.pkl')
            clf = joblib.load('clf.pkl')
        except:
            print("Error in loading pickled data, check for *.pkl files")
            sys.exit(1)

    # Create training and test features data
    if args.features:
        try:
            fsets = buildFeatureSets(recipes, listIngredients, args.traincount, args.testcount)
            tr = fsets[0]
            te = fsets[1]
        except:
            print("Error in building feature sets")
            sys.exit(1)

    listIngredients = tr[0]
    trainFeatures = tr[1]
    trainLabels = tr[2]
    trainID = tr[3]
    testFeatures = te[1]

    # Create classifier
    if args.classifier:
        try:
            clf = buildClassifier(tr, te)
        except:
            print("Error in building classifier")
            sys.exit(1)

    l2 = findCuisine(refrig, listIngredients, clf)
    printCuisinesClass(l2)

    l3 = findNeighbors(refrig, listIngredients, trainFeatures, trainID)
    printCuisinesClose(l3)
