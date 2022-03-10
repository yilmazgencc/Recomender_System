from utils import dataprep, plots, optimizer_selection, recommended_movies, model_selection
import tensorflow as tf
import argparse

def train(args):
    #Download and prepare MovieLens 1M dataset automatically for train process
    train_dataset, eval_dataset, user_vocabulary, movie_vocabulary, ratings_data = dataprep(args)
    #Prepare model according to selected model and embedding size / default=Neural Newtork(NN) , embedding size=64
    model = model_selection(user_vocabulary, movie_vocabulary, args)
    #print(model.summary())
    #Selected optimizer, loss, and metric are defined to model
    model.compile(optimizer=optimizer_selection(args),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[ tf.keras.metrics.RootMeanSquaredError(name="rmse")] )
    print("Training is starting")
    history = model.fit(train_dataset, epochs=args.epochs, validation_data=eval_dataset )
    model.save('best_model_NN.hdf5')
    #Plots loss and rsme graph accorging to ascending epochs
    plots(history)
    #Print 5 recommended movies for specific user    
    movie_recommendations=recommended_movies(ratings_data,model, args)
    return model, movie_recommendations, history
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MovieLens 1M Recommender system")
    parser.add_argument("--model", choices=["CF", "NN"],default="NN" )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--opt", choices=["Adam", "SGD", "RMSprop", "Adagrad"], default="Adam")
    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--use_deterministic", action="store_true")
    parser.add_argument("--specific_user", action="store_true")
    parser.add_argument("--user_id", default=1, type=int)
    parser.add_argument("--train_test_split", type=float, default=0.8)
    args = parser.parse_args()
    
    print("Starting...")
    model, movie_recommendations, history=train(args)
