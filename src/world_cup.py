from collections import defaultdict
from live_feature_builder import build_match_features
from groups import simulate_group
from initialize_state import initialize_state
import time

start = time.time()
SIMULATIONS = 1000

GROUP_A = ["Mexico", "South Africa", "South Korea", "Czech Republic"]
GROUP_B = ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"]
GROUP_C = ["Brazil", "Morocco", "Haiti", "Scotland"]
GROUP_D = ["United States", "Paraguay", "Australia", "Turkey"]
GROUP_E = ["Germany", "Curaçao", "Ivory Coast", "Ecuador"]
GROUP_F = ["Netherlands", "Japan", "Sweden", "Tunisia"]
GROUP_G = ["Belgium", "Egypt", "Iran", "New Zealand"]
GROUP_H = ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"]
GROUP_I = ["France", "Senegal", "Iraq", "Norway"]
GROUP_J = ["Argentina", "Algeria", "Austria", "Jordan"]
GROUP_K = ["Portugal", "DR Congo", "Uzbekistan", "Colombia"]
GROUP_L = ["England", "Croatia", "Ghana", "Panama"]

ALL_GROUPS = {
    "A": GROUP_A,
    "B": GROUP_B,
    "C": GROUP_C,
    "D": GROUP_D,
    "E": GROUP_E,
    "F": GROUP_F,
    "G": GROUP_G,
    "H": GROUP_H,
    "I": GROUP_I,
    "J": GROUP_J,
    "K": GROUP_K,
    "L": GROUP_L
}


elo_system, histories = initialize_state()

qualification_counts = defaultdict(int)

print(f"Running {SIMULATIONS} simulations across 12 groups...")

for sim_num in range(SIMULATIONS):
    if (sim_num + 1) % max(1, SIMULATIONS // 10) == 0:
        print(f"  Progress: {sim_num + 1}/{SIMULATIONS} simulations completed")
    
    for group_name, teams in (
        ALL_GROUPS.items()
    ):
        # Simulate this group
        table = simulate_group(
            teams,
            feature_builder=lambda home, away:
                build_match_features(
                    home,
                    away,
                    elo_system,
                    histories
                )
        )
        
        # Count first and second place qualifiers
        first_place = table[0][0]
        second_place = table[1][0]
        
        qualification_counts[first_place] += 1
        qualification_counts[second_place] += 1

        
print("\nQualification Probabilities \n")        

for team, count in sorted(
    qualification_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    probability = (count / SIMULATIONS) * 100
    print(f"{team}: ({probability:.2f}%)")

print("Elapsed:", time.time() - start)