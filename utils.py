import os
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup
import matplotlib.pyplot as plt
from keras.models import Model

def dataprep(args):
    if not os.path.exists("ml-1m"):
        urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip")
        ZipFile("movielens.zip", "r").extractall()

    ratings_data = pd.read_csv(
        "ml-1m/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "unix_timestamp"], engine="python"
    )

    ratings_data["movie_id"] = ratings_data["movie_id"].apply(lambda x: f"movie_{x}")
    ratings_data["user_id"] = ratings_data["user_id"].apply(lambda x: f"user_{x}")
    ratings_data["rating"] = ratings_data["rating"].apply(lambda x: float(x))
    del ratings_data["unix_timestamp"]

    print(f"Number of users: {len(ratings_data.user_id.unique())}")
    print(f"Number of movies: {len(ratings_data.movie_id.unique())}")
    print(f"Number of ratings: {len(ratings_data.index)}")
    csv_header = list(ratings_data.columns)
    user_vocabulary = list(ratings_data.user_id.unique())
    movie_vocabulary = list(ratings_data.movie_id.unique())
    target_feature_name = "rating"
    if args.use_deterministic:
        np.random.seed(args.seed)
    random_selection = np.random.rand(len(ratings_data.index)) <= args.train_test_split
    train_data = ratings_data[random_selection]
    eval_data = ratings_data[~random_selection]

    train_data.to_csv("ml-1m/train_data.csv", index=False, sep="|", header=False)
    eval_data.to_csv("ml-1m/eval_data.csv", index=False, sep="|", header=False)
    print(f"Train data split: {len(train_data.index)}")
    print(f"Eval data split: {len(eval_data.index)}")
    print("Train and eval data files are saved.")

    def get_dataset_from_csv(csv_file_path, batch_size=128, shuffle=True):
        return tf.data.experimental.make_csv_dataset(
            csv_file_path,
            batch_size=batch_size,
            column_names=csv_header,
            label_name=target_feature_name,
            num_epochs=1,
            header=False,
            field_delim="|",
            shuffle=shuffle,
        )

    # Read the training data.
    train_dataset = get_dataset_from_csv("ml-1m/train_data.csv", args.batch_size)
    # Read the test data.
    eval_dataset = get_dataset_from_csv("ml-1m/eval_data.csv", args.batch_size, shuffle=False)

    return train_dataset, eval_dataset, user_vocabulary, movie_vocabulary, ratings_data


def embedding_encoder(vocabulary, embedding_dim, num_oov_indices=0, name=None):
    return keras.Sequential(
        [
            StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=num_oov_indices
            ),
            layers.Embedding(
                input_dim=len(vocabulary) + num_oov_indices, output_dim=embedding_dim
            ),
        ],
        name=f"{name}_embedding" if name else None,
    )


def NN(user_vocabulary, movie_vocabulary, args):
    dropout = args.dropout
    embedding_dim = args.embedding_dim
    user_input = layers.Input(name="user_id", shape=(), dtype=tf.string)
    # Get user embedding.
    user_embedding = embedding_encoder(vocabulary=user_vocabulary, embedding_dim=embedding_dim, name="user")(user_input)
    # Receive the movie as an input.
    movie_input = layers.Input(name="movie_id", shape=(), dtype=tf.string)
    # Get movie embedding.
    movie_embedding = embedding_encoder(vocabulary=movie_vocabulary, embedding_dim=embedding_dim, name="movie")(movie_input)
    # Setup  linear layers with relu activaiton function
    d = layers.Concatenate(axis=1)([user_embedding, movie_embedding])
    d = layers.Dense(256, activation="relu")(d)
    d = layers.Dropout(dropout)(d)
    d = layers.Dense(64, activation="relu")(d)
    d = layers.Dropout(dropout)(d)
    d = layers.Dense(16, activation="relu")(d)
    d = layers.Dropout(dropout)(d)
    d = tf.keras.layers.Dense(1, activation="sigmoid")(d)
    # Convert to rating scale.
    output = d * 4 + 1
    # Create the model.
    model = Model(inputs=[user_input, movie_input], outputs=output, name="nn_model")

    return model


def CF(user_vocabulary, movie_vocabulary, args):
    embedding_dim = args.embedding_dim
    # Receive the user as an input.
    user_input = layers.Input(name="user_id", shape=(), dtype=tf.string)
    # Get user embedding.
    user_embedding = embedding_encoder(vocabulary=user_vocabulary, embedding_dim=embedding_dim, name="user" )(user_input)
    # Receive the movie as an input.
    movie_input = layers.Input(name="movie_id", shape=(), dtype=tf.string)
    # Get movie embedding.
    movie_embedding = embedding_encoder(vocabulary=movie_vocabulary, embedding_dim=embedding_dim, name="movie")(movie_input)
    # Compute dot product similarity between user and movie embeddings.
    logits = layers.Dot(axes=1, name="dot_similarity")([user_embedding, movie_embedding])
    
    # Convert to rating scale.
    prediction = keras.activations.sigmoid(logits) * 4 + 1
    # Create the model.
    model = keras.Model(inputs=[user_input, movie_input], outputs=prediction, name="CF")
    return model


def optimizer_selection(args):
    if args.opt == "Adam":
        optimizer = keras.optimizers.Adam(args.lr)
    elif args.opt == "SGD":
        optimizer = keras.optimizers.SGD(args.lr)
    elif args.opt == "RMSprop":
        optimizer = keras.optimizers.RMSprop(args.lr)
    elif args.opt == "Adagrad":
        optimizer = keras.optimizers.Adagrad(args.lr)
    return optimizer

def model_selection(user_vocabulary, movie_vocabulary, args):
    if args.model == "NN":
        model = NN(user_vocabulary, movie_vocabulary, args)
    elif args.model == "CF":
        model = CF(user_vocabulary, movie_vocabulary, args)
    return model

def plots(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "eval"], loc="upper left")
    plt.show()
    plt.plot(history.history["rmse"])
    plt.plot(history.history["val_rmse"])
    plt.title("Model RMSE")
    plt.ylabel("RMSE")
    plt.xlabel("Epoch")
    plt.legend(["train", "eval"], loc="upper left")
    plt.show()

def recommended_movies(ratings_data,model, args):
    movie_df = pd.read_csv("ml-1m/movies.dat", sep="::", names=['movieId', 'title', 'genres'], encoding='latin-1', engine="python")
    movie_df["movieId"] = movie_df["movieId"].apply(lambda x: f"movie_{x}")
    
    # Let us get a user and see the top recommendations.
    df = ratings_data
    if args.specific_user:
        userid="user_"+str(args.user_id)
    else:
        userid = df.user_id.sample(1).iloc[0]
    movies_watched_by_user = df[df.user_id == userid]
    movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user.movie_id.values)]["movieId"]
    # movies_not_watched = movie_df["movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(set(df["movie_id"])))
    # movies_not_watched = list(set(movies_watched_by_user["movie_id"]).intersection(set(df["movie_id"])))
    movies_not_watched = [[x] for x in movies_not_watched]
   
    user_movie_array = np.hstack(([[userid]] * len(movies_not_watched), movies_not_watched))
    ratings = model.predict([user_movie_array[:, 0], user_movie_array[:, 1]]).flatten()
    ratings = [[x] for x in ratings]
    ratings_indices = np.hstack((movies_not_watched, ratings))
    top_ratings_indices = ratings_indices[ratings_indices[:, 1].argsort()[-5:][::-1]]
    recommended_movie_ids = [x[0] for x in top_ratings_indices]
    
    # print("Showing recommendations for user: {}".format(userid))
    # print("====" * 9)
    # print("Movies with high ratings from user")
    # print("----" * 8)
    # top_movies_user = (
    #     movies_watched_by_user.sort_values(by="rating", ascending=False)
    #         .head(10)
    #         .movie_id.values
    # )
    # movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
    # for row in movie_df_rows.itertuples():
    #     print(row.title, ":", row.genres)
    print("----" * 8)
    print("Top 5 movie recommendations for {}".format(userid))
    print("----" * 8)
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    for row in recommended_movies.itertuples():
        print(row.title, ":", row.genres)
        
    #     #burası silinicek sonrasında denemelik
    # array1=movies_watched_by_user.sort_values(by="movie_id", ascending=True).rating.values
    # array2=ratings_indices[ratings_indices[:, 0].argsort()][:,1]
    # print(array1) 
    # print(array2) 