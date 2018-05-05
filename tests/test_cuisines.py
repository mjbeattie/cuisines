import cuisines
import joblib

clf = joblib.load('cuisines/clf.pkl')
tr = joblib.load('cuisines/traindata.pkl')
te = joblib.load('cuisines/testdata.pkl')

def test_findCuisine0():
    cuisinelist = cuisines.findCuisine(["feta cheese", "pita bread"], tr[0], clf)
    ing = cuisinelist[0]
    assert ing[0] == "greek"

def test_findCuisine1():
    cuisinelist = cuisines.findCuisine(["tortillas", "habanero pepper", "chiles"],
                                       tr[0], clf)
    ing = cuisinelist[0]
    assert ing[0] == "mexican"

def test_findCuisine2():
    cuisinelist = cuisines.findCuisine(["oregano", "pasta", "tomatoes"],
                                       tr[0], clf)
    ing = cuisinelist[0]
    assert ing[0] == "italian"

def test_findNeighbors0():
    recipelist = cuisines.findNeighbors(["feta cheese", "pita bread"],
                                        tr[0], tr[1], tr[3])
    rec = recipelist[0]
    assert rec[0] == 584

def test_findNeighbors1():
    recipelist = cuisines.findNeighbors(["tortillas", "habanero pepper", "chiles"],
                                        tr[0], tr[1], tr[3])
    rec = recipelist[0]
    assert rec[0] == 2239

def test_findNeighbors2():
    recipelist = cuisines.findNeighbors(["oregano", "pasta", "tomatoes"],
                                        tr[0], tr[1], tr[3])
    rec = recipelist[0]
    assert rec[0] == 1538





