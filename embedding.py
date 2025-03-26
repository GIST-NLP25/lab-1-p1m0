import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_torch_modules import SequenceDatasetEmbedding, CustomNNEmbedding


def _create_dict(df):
    x_dict = {}
    unique_x_values = set()

    for col in df.columns[:-1]:
        unique_x_values.update(df[col].dropna().unique())
    unique_x_values.add('unseen symbol')

    for idx, value in enumerate(sorted(unique_x_values)):
        x_dict[value] = idx + 1

    y_dict = {}
    unique_y_values = set(df['label'].dropna().unique())

    for idx, value in enumerate(sorted(unique_y_values)):
        y_dict[value] = idx

    return x_dict, y_dict


def _create_train_df_from_dict(df, x_dict, y_dict):
    transformed_df = pd.DataFrame()
    
    for col in df.columns[:-1]:
        transformed_df[col] = df[col].map(lambda x: x_dict.get(x, x_dict.get('unseen value')) if pd.notna(x) else 0)
    
    transformed_df['label'] = df['label'].map(lambda x: y_dict.get(x))

    return transformed_df


def _create_test_df_from_dict(df, x_dict):
    test_df_emb = pd.DataFrame()

    for col in df.columns:
        test_df_emb[col] = df[col].map(lambda x: x_dict.get(x, x_dict.get('unseen symbol')) if pd.notna(x) else 0)
    
    return test_df_emb


def _create_dataloarders(training_data, batch_size, verbose=True):
    training_data_t = SequenceDatasetEmbedding(training_data.values)
    train_dataloader = DataLoader(training_data_t, batch_size=batch_size, shuffle=True)

    if verbose:
        for X, y in train_dataloader:
            print(f"Shape of X: {X.shape}, {X.dtype}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

    return train_dataloader


def _calculate_accuracy(predictions, actuals):
    n_correct = 0
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == actuals[i]:
            n_correct += 1
    return n_correct / len(predictions)


def _train_epoch(dataloader, model, loss_fn, optimizer, verbose=True):
    model.train()
    num_batches = len(dataloader)
    train_loss = 0
    predictions = []
    actuals = []

    if verbose:
        dataloader = tqdm(dataloader, total=len(dataloader))
    
    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        predictions.extend(pred.cpu().detach().numpy())
        actuals.extend(y.detach().cpu().numpy())
            
    train_loss /= num_batches
    accuracy = _calculate_accuracy(predictions, actuals)
    
    train_loss /= num_batches
    if verbose:
        print(f"Training avg loss: {train_loss:>7f}")
        print(f"Training Accuracy     : {100 * accuracy:.4f}%")

    return train_loss, accuracy


def create_train_test_df_embedding(train_df, test_df):
    x_dict, y_dict = _create_dict(train_df)
    
    train_df_oh = _create_train_df_from_dict(train_df, x_dict, y_dict)
    test_df_oh = _create_test_df_from_dict(test_df, x_dict)

    return train_df_oh, test_df_oh, x_dict, y_dict


def train_embedding(train_df, vocab_size, num_epochs=50, verbose=True):
    model = CustomNNEmbedding(vocab_size)
    loss_fn = nn.CrossEntropyLoss()
    training_dataloader = _create_dataloarders(train_df, batch_size=32, verbose=verbose)

    best_model = None
    best_accuracy = 0
    best_loss = float('inf')

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(num_epochs):
        if verbose:
            print(f"-------------------------------\nEpoch {epoch+1}")
        tr_loss, tr_acc = _train_epoch(training_dataloader, model, loss_fn, optimizer, verbose=verbose)
        if tr_loss < best_loss:
            best_loss = tr_loss
            best_accuracy = tr_acc
            best_model = model

    if verbose:
        print("\n+-------------------------------+")
        print(f"Best model loss: {best_loss:.4f}")
        print(f"Best model accuracy: {100 * best_accuracy:.4f}%\n\n")

    return best_model


def predict_test_embedding(model, test_df, y_dict):
    results_df = pd.DataFrame(columns=['id', 'pred'])

    model.eval()
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            features = torch.tensor(row.values, dtype=torch.long)
            output = model(features.unsqueeze(0))
            
            predicted_class_idx = torch.argmax(output).item()
            labels = list(y_dict.keys())
            predicted_label = labels[predicted_class_idx]
            
            if idx < 9:
                results_df.loc[idx] = [f'S00{idx + 1}', predicted_label]
            elif idx < 99:
                results_df.loc[idx] = [f'S0{idx + 1}', predicted_label]
            else:
                results_df.loc[idx] = [f'S{idx + 1}', predicted_label]

    return results_df
