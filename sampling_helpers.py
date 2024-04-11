import pandas as pd
import numpy as np
from random import randint


# Calculate distance for each embedding from its cluster's center
def get_distance_from_center(df: pd.DataFrame, col: str, kmeans_model):
    # Store cluster centers in df
    df['cluster_center'] = df.apply(lambda x: kmeans_model.cluster_centers_[x['kmeans_cluster']], axis=1)

    # Calculate square distance
    df['sq_distance'] = np.subtract(df[col], df['cluster_center'])**2
    df['sq_distance'] = df['sq_distance'].apply(lambda x: sum(x))

    # Drop cluster center column
    df = df.drop(columns=['cluster_center'])

    return df


def get_train_test_questions(df: pd.DataFrame, num_clusters: int, train_size: int, test_size: int):
    train_questions = []
    test_questions = []

    # Use each cluster to populate train and test questions
    for c in range(0, num_clusters):
        # Filter df down to cluster
        temp_df = df[df['kmeans_cluster'] == c]

        # Store observation closest to center and continue to next cluster if cluster only contains 2 observations
        if len(temp_df) == 2:
            print(f"Cluster {c} only contains 2 observations. Storing the observation closest to center in train set.")
            train_questions.append(temp_df[temp_df['sq_distance'] == temp_df['sq_distance'].min()]['questions'].iloc[0])

            # Store other observation in test_questions
            print(f"Storing other observation in test set.")
            print("")
            test_questions.append(temp_df[temp_df['sq_distance'] == temp_df['sq_distance'].max()]['questions'].iloc[0])
            continue

        # Continue without storing any questions if cluster only contains 1 observation
        if len(temp_df) == 1:
            print(f"Cluster {c} only contains 1 observation. No observation will be stored in train set for this cluster.")
            # Store observation in test_questions
            print(f"Storing only observation in test set.")
            print("")
            test_questions.append(temp_df['questions'].iloc[0])
            continue

        # Store highest and lowest distances from cluster center
        train_questions.append(temp_df[temp_df['sq_distance'] == temp_df['sq_distance'].max()]['questions'].iloc[0])
        train_questions.append(temp_df[temp_df['sq_distance'] == temp_df['sq_distance'].min()]['questions'].iloc[0])

        # Check that temp_df is big enough to supply more train questions
        if len(temp_df) > train_size > 2:
            # Filter these out of the temp_df before pulling more questions and reset index
            temp_df = temp_df[temp_df['sq_distance'] != temp_df['sq_distance'].max()]
            temp_df = temp_df[temp_df['sq_distance'] != temp_df['sq_distance'].min()]
            temp_df = temp_df.reset_index(drop=True)

            # Calculate remaining number of observations needed
            remaining_num = train_size - 2

            # Create list of indices and calculate length (to be used in randint function)
            temp_list = list(temp_df.index)
            length = len(temp_list) - 1 # randint function is inclusive of last number

            # Track which question indices have been used and randomly generate new ones until remaining number of questions needed is 0
            used_indices = []
            while remaining_num > 0:
                # Randomly select remaining questions
                idx = randint(0, length)

                # Keep generating random number till a new one is found
                while idx in used_indices:
                    idx = randint(0, length)

                # Store item in list
                train_questions.append(temp_df.loc[idx: idx]['questions'].iloc[0])

                # Add index to used_indices list
                used_indices.append(idx)

                # Subtract 1 from remaining_num
                remaining_num -= 1

        # Extract test questions from non-training questions
        temp_df['test_candidates'] = temp_df['questions'].apply(lambda x: 1 if x not in train_questions else 0)
        temp_df = temp_df[temp_df['test_candidates'] == 1]
        temp_df = temp_df.reset_index(drop=True)

        # Check if enough test questions to meet test_size
        if len(temp_df) < test_size:
            print(f"Cluster {c} does not have enough observations for test size of {test_size}. Storing all non-training observations ({len(temp_df)}) into test set.")
            print("")

        # If test questions are equal to or less than test size, store all remaining observations in test set.
        if len(temp_df) <= test_size:
            [test_questions.append(q) for q in temp_df['questions'].to_list()]
            continue

        # Create list of indices and calculate length (to be used in randint function)
        temp_list = list(temp_df.index)
        length = len(temp_list) - 1 # randint function is inclusive of last number

        # Reset used_indices to track which indices were used for test
        used_indices = []

        # Randomly pull test questions
        while test_size > 0:
            # Randomly select remaining questions
            idx = randint(0, length)

            # Keep generating random number till a new one is found
            while idx in used_indices:
                idx = randint(0, length)

            # Store item in list
            test_questions.append(temp_df.loc[idx: idx]['questions'].iloc[0])

            # Add index to used_indices list
            used_indices.append(idx)

            # Subtract 1 from remaining_num
            test_size -= 1

    return train_questions, test_questions
