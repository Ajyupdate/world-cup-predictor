import random

from xgboost_predict import predict_probabilities

def simulate_match(feature_row): 
    # print(feature_row)
    probs = predict_probabilities(feature_row)
   
    result = random.choices(
        population=[
            "HOME_WIN",
            "DRAW",
            "AWAY_WIN"
        ],
        weights=[
            probs["HOME_WIN"],
            probs["DRAW"],
            probs["AWAY_WIN"]
        ],
        k=1
    )[0]
    
    return result