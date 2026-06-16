from itertools import combinations

from simulate_match import simulate_match

def generate_group_features(teams):

    return list(combinations(teams, 2))

def simulate_group(teams, feature_builder):
    standings = {team: 0 for team in teams}

    fixtures = generate_group_features(teams)

    for home, away in fixtures:

        features = feature_builder(home, away)
        result = simulate_match(features)

        if result == "HOME_WIN":
            standings[home] += 3
        elif result == "AWAY_WIN":
            standings[away] += 3
        else:
            standings[home] += 1
            standings[away] += 1
    sorted_table = sorted(standings.items(), key=lambda x: x[1], reverse=True)
    return sorted_table
