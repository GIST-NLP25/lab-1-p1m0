from data_reader import read_train_dataset, read_test_dataset
from one_hot import create_train_test_df_oh, train_oh, predict_test_oh
from embedding import create_train_test_df_embedding, train_embedding, predict_test_embedding

#####################
# YOU MUST WRITE YOUR STUDENT ID IN THE VARIABLE STUDENT_ID
# EXAMPLE: STUDENT_ID = "12345678"
#####################
STUDENT_ID = "20258009"

def load_data():
    # EXAMPLE
    # Load the data from a csv file
    # You can change function
    train_df = read_train_dataset('dataset/simple_seq.train.csv')
    test_df = read_test_dataset('dataset/simple_seq.test.csv')

    return train_df, test_df

def save_data(df1, df2):
    # EXAMPLE
    # Save the data to a csv file
    # You can change function
    # BUT you should keep the file name as "{STUDENT_ID}_simple_seq.p#.answer.csv"
    df1.to_csv(f'{STUDENT_ID}_simple_seq.p1.answer.csv', index=False)
    df2.to_csv(f'{STUDENT_ID}_simple_seq.p2.answer.csv', index=False)

def main():
    # WRITE YOUR CODE HERE
    verbose = False
    train_df, test_df = load_data()
    # ================ OH ================ #
    if verbose:
        print("================= One Hot Encoding =================")
    train_df_oh, test_df_oh, y_dict_oh = create_train_test_df_oh(train_df, test_df)
    best_model_oh = train_oh(train_df_oh, num_epochs=40, verbose=verbose)
    results_df_oh = predict_test_oh(best_model_oh, test_df_oh, y_dict_oh)
    # ================ Embedding ================ #
    if verbose:
        print("================= Embedding =================")
    train_df_embedding, test_df_embedding, x_dict_embedding, y_dict_embedding = create_train_test_df_embedding(train_df, test_df)
    vocab_size = len(x_dict_embedding) + 1
    best_model_embedding = train_embedding(train_df_embedding, vocab_size, num_epochs=40, verbose=verbose)
    results_df_embedding = predict_test_embedding(best_model_embedding, test_df_embedding, y_dict_embedding)
    
    save_data(results_df_oh, results_df_embedding)
    print("Successfully trained models, predicted new data and saved the results to CSV files.")

if __name__ == "__main__":
    main()
