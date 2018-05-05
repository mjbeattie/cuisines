Module:  redactor
Author:  Matthew J. Beattie
Author email:  mjbeattie@ou.edu
Version Date:  April 30, 2018
Class:  Univ of Oklahoma DSA5970 (Dr. Grant)

**OVERVIEW**
cuisines is a module that contains a set of functions and one program that can be used to
determine recipes and cuisines when given a set of ingredients by the user.  The main
program, cuisines.py is capable of several different processes, and it uses a data file,
yummly.json, that is included in this package.  The program can develop features sets,
build classifiers, determine most likely cuisine for a set of ingredients, and show the
five closest recipes in yummly.json when given a set of ingredients.


**INSTALLATION**
The files for the module are archived in a tar ball called beat_project2.tar.gz.  To
install the file, copy it to the /projects/ directory and unpack.  If you install to
a different directory, you will need to modify the pytest files, which expect a fixed
directory system.  The program itself should still run properly.  The module can then
be installed via pip with the following command:

	pip3 install --editable .

Depending on your permissions, you may need to sudo the install.  When complete, the
files installed will include:

	cuisines.py		The main program with functions
	clf.pkl			A pickle file containing a prebuilt classifier
	trainingdata.pkl	A pickle file containing prebuilt feature sets for 3000
				recipes
	testdata.pkl		A pickle file containing prebuild test sets for 300
				recipes
	/docs/			Directory for yummly.json
	/docs/yummly.json	The recipes data file from yummly
	/tests/			Directory for test files for cuiisines.py

*NOTE TO GITHUB USERS:  If you are copying my repository and using original code, you
will have to unpack the pickle files from the cuisines.pickle.files.tar.gz tar ball.*


**TESTING**
The cuisines module comes with a set of tests for the cuisines.py program and some of its
functions.  The tests DO NOT create new feature files due to the length of time it takes
to generate features.  The tests do take a few sample ingredient sets and return cuisines
and nearest recipes.  Note that since the classifier is probabilistic, it is possible to
fail a test even though the routine runs correctly.  I've tried to simplify the test sets
to make this less likely.  To run the tests, enter the following command from the
/projects/cuisines directory:

	python3 -m pytest -v


**RUNNING CUISINES.PY**
cuisines.py is the main program for the module and is the only interface for creating
features, classifiers, and analyzing ingredient sets.  The option flags with cuisines.py
are very important, but default to simple recipe analysis so that the user isn't stuck
defining feature sets.

Invocation:  Start the program with the command:  python3 cuisines.py
Options:	--pickles	This option directs the program to use the included
				pickle files.  If the user wants to generate a new
				classifier, this option uses traindata.pkl and
				testdata.pkl

		--features	Generates new feature sets from the yummly.json data.
				Saves features to mytraindata.pkl and mytestdata.pkl
				so that the included feature set, traindata.pkl and
				testdata.pkl, are not overwritten.

		--traincount	The number of recipes to include in the training data.
				Defaults to 1000.  Only used with --features

		--testcount	The number of recipes to include in the test data.
				Defaults to 200.  Only used with --features

		--classifier	Generates a new classifier, either a Naive Bayes or
				Decision Tree, whichever performs better.  If
				--features is not selected, it will use the pickle files
				for training and test data.
				Saves features to myclf.pkl so that the included classifier,
				clf.pkl, is not overwritten.

		--ingredient	An ingredient to include in a set to analyze.  The user
				can enter as many ingredients as he/she wants.  To
				enter multiple ingredients, simply enter this option
				multiple times:  python3 cuisines.py -i "onions" -i "cheese".
				Ingredients should be in lower case.  The program will
				see if the ingredient has been used in yummly.  If it
				hasn't, the ingredient is omitted from the analysis.


**HOW IT WORKS**
cuisines.py uses the SciKit package to perform most of its analyses.  The package in turn
requires that feature data be in the form of numeric vectors.  Therefore, cuisines.py
reads in the yummly.json data, and creates a set of ingredients seen in all the recipes.
To create feature sets, cuisines reads in each recipe and creates a binary vector that
represents the presence of each of the 6174 potential ingredients.  Feature creation is
by far the most time intense portion of the program.  

Once these features are created, cuisines calls the Naive Bayes and Decision Tree classifier
construction methods in SciKit.  It creates the classifiers and evaluates their performance
on test data.  cuisines then stores and returns the classifier that performs best.

With a classifier defined, cuisines can then analyze ingredient sets supplied by the user.
The user enters ingredients as defined above.  cuisines then uses the classifier.predict_proba
routine from SciKit to identify the most likely cuisines and the probability that each of
those cuisines is correct.

To find the recipes that are best for the ingredient set, cuisines uses KNN clustering, again
with SciKit, to determine the closest recipes.  I've restricted the recipe set to the training
data in order to speed up the program.


**PERFORMANCE:**
Feature creation in cuisines.py uses a great deal of memory and is very slow.  Each recipe
will have binary vector of 6174, so the larger the training and test sets, the greater the
amount of memory required.  These are sparse vectors, so it may be possible to improve
performance with special packages.  Nevertheless, with N training recipes, we are performing
Nx6174 comparisons to create the binary vectors.

Another way to improve feature creation performance would be to rewrite cuisines.py to take
advantage of multiple CPUs.  I actually tried this but had some trouble getting the Pool()
routine to work properly.  While I did find an alternative, I was running out of time, so I
instead have used a maximum training set of 3000 to create the pickled training data.

Even with 3000 recipes, the classifier is pretty accurate.  It properly predicted the cuisine
of 60% of the test set.  In practice, 'italian' comes up pretty often in use, but this is
because there are many 'italian' recipes in yummly.json.  The best classifier tends to be
the Naive Bayes classifier.

One other potential area of improvement would be to lemmatize the ingredient set.  This would
reduce the size of the binary vector and would ensure better matches with ingredients
entered by the user.  Both lemmatization and parallel processing would be easy to implement,
but I didn't have time to do so.

