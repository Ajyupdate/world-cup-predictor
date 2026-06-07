31/05/2026
Version 1 focuses on building the foundation of a football match prediction system. Historical match data is processed through an ELO rating engine to quantify team strength over time. These ELO ratings are transformed into machine-learning features and used to train a Logistic Regression model. The current system can learn from past matches and generate basic win, draw, or loss predictions based on team ELO differences.


06/05/2026
In version 2, the pipeline was made time-aware: feature_builder.py now ignores future/unplayed matches so ELO and form history only reflect completed games, train.py sorts data by date and uses a chronological 80/20 split instead of random shuffling, and the save/load flow was fixed so the trained model is stored under data/models and predict.py loads from that same path with a clear error if the files are missing.


07/06/2026
In version 3, the features were improved by adding home advantage and tournament importance
