import pandas as pd
import subprocess

if __name__ == '__main__':

    wget_command = ["wget",
                    "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
                    "-O", "ml-1m.zip"]

    try:
        subprocess.run(wget_command, check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    # Use subprocess to run the wget command
    wget_command = ["unzip", "-o", "ml-1m.zip"]
    try:
        subprocess.run(wget_command, check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    data_path = ""

    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    user = pd.read_csv(data_path + 'ml-1m/users.dat', sep='::', header=None, names=unames)

    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(data_path + 'ml-1m/ratings.dat', sep='::', header=None, names=rnames)

    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(data_path + 'ml-1m/movies.dat', sep='::', header=None, names=mnames, encoding="unicode_escape")
    movies['genres'] = list(map(lambda x: x.split('|')[0], movies['genres'].values))

    data = pd.merge(pd.merge(ratings, movies), user)
    data.to_csv("movie_lens_1m.csv", index=False)
