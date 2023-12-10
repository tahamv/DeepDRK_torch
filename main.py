from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from data_loader import RawDataLoader
from utils import *
import random
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim, no_grad
from evaluation import Evaluation

# Step 1: Define the batch size for training
batch_size = 64
num_epochs = 10


class DeepDRK(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepDRK, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(200, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)


def train(model, train_loader, val_loader, num_epochs, class_weights):
    """
    Trains the DeepDRA (Deep Drug Response Anticipation) model.

    Parameters:
    - model (DeepDRA): The DeepDRA model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - num_epochs (int): Number of training epochs.
    """
    mlp_loss_fn = nn.BCELoss()

    train_accuracies = []
    val_accuracies = []

    train_loss = []
    val_loss = []

    mlp_optimizer = optim.SGD(model.parameters(), lr=0.05,nesterov=True,momentum=0.8 )

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total_samples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            mlp_optimizer.zero_grad()

            # Forward pass
            mlp_output = model(data)

            # Compute class weights for the current batch
            # batch_class_weights = class_weights[target.long()]
            # mlp_loss_fn = nn.BCEWithLogitsLoss(weight=batch_class_weights)

            # Compute losses
            total_loss = mlp_loss_fn(mlp_output, target)

            # Backward pass and optimization
            total_loss.backward()
            mlp_optimizer.step()
            total_train_loss += total_loss.item()

            # Calculate accuracy
            train_predictions = torch.round(mlp_output)
            train_correct += (train_predictions == target).sum().item()
            train_total_samples += target.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for val_batch_idx, (data_val, val_target) in enumerate(val_loader):
                mlp_output_val = model(data_val)

                # Compute losses
                total_val_loss = mlp_loss_fn(mlp_output_val, val_target)

                # Calculate accuracy
                val_predictions = torch.round(mlp_output_val)
                correct += (val_predictions == val_target).sum().item()
                total_samples += val_target.size(0)

            avg_val_loss = total_val_loss / len(val_loader)
            val_loss.append(avg_val_loss)

        train_accuracy = train_correct / train_total_samples
        train_accuracies.append(train_accuracy)
        val_accuracy = correct / total_samples
        val_accuracies.append(val_accuracy)

        print(
            'Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Train Accuracy: {:.4f}, Val Accuracy: {:.4f}'.format(
                epoch + 1, num_epochs, avg_train_loss, avg_val_loss, train_accuracy,
                val_accuracy))

    # Save the trained model
    torch.save(model.state_dict(), 'DeepDRK.pth')


def test(model, test_loader, reverse=False):
    """
    Tests the given model on the test dataset using evaluation metrics.

    Parameters:
    - model: The trained model to be evaluated.
    - test_loader: DataLoader for the test dataset.
    - reverse (bool): If True, reverse the predictions for evaluation.

    Returns:
    - result: The evaluation result based on the chosen metrics.
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize lists to store predictions and ground truth labels
    all_predictions = []
    all_labels = []

    # Iterate over the test dataset
    for i, (data_loader, labels) in enumerate(test_loader):
        # Forward pass through the model
        with torch.no_grad():
            mlp_output = model(data_loader)

        # Apply reverse if specified
        predictions = 1 - mlp_output if reverse else mlp_output

    # Evaluate the predictions using the specified metrics
    result = Evaluation.evaluate(labels, predictions)

    return result


def train_DeepDRK(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes, drug_sizes, device):
    """

    Train and evaluate the DeepDRA model.

    Parameters:
    - X_cell_train (pd.DataFrame): Training data for the cell modality.
    - X_cell_test (pd.DataFrame): Test data for the cell modality.
    - X_drug_train (pd.DataFrame): Training data for the drug modality.
    - X_drug_test (pd.DataFrame): Test data for the drug modality.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Test labels.
    - cell_sizes (list): Sizes of the cell modality features.
    - drug_sizes (list): Sizes of the drug modality features.

    Returns:
    - result: Evaluation result on the test set.
    """

    model = DeepDRK(sum(cell_sizes) + sum(drug_sizes), 1)
    model.to(device)

    train_data = pd.concat([x_cell_train, x_drug_train], axis=1)

    # Step 3: Convert your training data to PyTorch tensors
    x_train_tensor = torch.Tensor(train_data.values)
    # x_train_tensor = torch.nn.functional.normalize(x_train_tensor, dim=0)
    y_train_tensor = torch.Tensor(y_train)
    y_train_tensor = y_train_tensor.unsqueeze(1)

    x_train_tensor.to(device)
    y_train_tensor.to(device)
    # Compute class weights
    classes = [0, 1]  # Assuming binary classification
    class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=y_train),
                                 dtype=torch.float32)

    x_train_tensor, x_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
        x_train_tensor, y_train_tensor, test_size=0.1,
        random_state=RANDOM_SEED,
        shuffle=True)

    # Step 4: Create a TensorDataset with the input features and target labels
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    # Step 5: Create the train_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Step 6: Train the model
    train(model, train_loader, val_loader, num_epochs, class_weights)

    # Step 7: Save the trained model
    torch.save(model, 'DeepDRA.pth')

    # Step 8: Load the saved model
    model = torch.load('DeepDRA.pth')


    # Step 9: Convert your test data to PyTorch tensors
    test_data = pd.concat([x_cell_test, x_drug_test], axis=1)
    x_test_tensor = torch.Tensor(test_data.values)
    y_test_tensor = torch.Tensor(y_test)

    # normalize data
    # x_test_tensor = torch.nn.functional.normalize(x_test_tensor, dim=0)

    # Step 10: Create a TensorDataset with the input features and target labels for testing
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=len(x_cell_test))

    # Step 11: Test the model
    return test(model, test_loader)


def cv_train(x_cell_train, x_drug_train, y_train, cell_sizes,
             drug_sizes, device, k=5, ):
    splits = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(x_cell_train)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        model = DeepDRK(sum(cell_sizes) + sum(drug_sizes), 1)

        train_data = pd.concat([x_cell_train, x_drug_train],axis=1 )

        # Convert your training data to PyTorch tensors
        x_train_tensor = torch.Tensor(train_data.values)
        y_train_tensor = torch.Tensor(y_train)
        y_train_tensor = y_train_tensor.unsqueeze(1)

        # Compute class weights
        classes = [0, 1]  # Assuming binary classification
        class_weights = torch.tensor(compute_class_weight(class_weight='balanced', classes=classes, y=y_train),
                                     dtype=torch.float32)

        # Create a TensorDataset with the input features and target labels
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

        # Create the train_loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        # Train the model
        train(model, train_loader, train_loader, num_epochs, class_weights)

        # Create a TensorDataset with the input features and target labels
        test_loader = DataLoader(train_dataset, batch_size=len(x_cell_train), sampler=test_sampler)

        # Test the model
        results = test(model, test_loader)

        # Step 10: Add results to the history dictionary
        Evaluation.add_results(history, results)

    return Evaluation.show_final_results(history)


def build_similarity_matrices(raw_dict, sim_data):
    if not BUILD_SIM_MATRICES: return
    for entity in tqdm(DATA_MODALITIES, 'Building Similarity Matrices...'):
        method, _lambda = SIM_KERNEL[entity]
        temp_df = pd.DataFrame(
            np.round(np.exp(-_lambda * squareform(pdist(raw_dict[entity].fillna(0), metric=method))), 10)
            , columns=raw_dict[entity].index)
        temp_df.index = temp_df.columns
        temp_df.to_csv(os.path.join(sim_data, f'{entity}_sim.tsv'), sep='\t')


def combine_test_train(train_data, test_data):
    data = {}
    for i in train_data:
        data[i] = pd.concat([train_data[i], test_data[i].T.add_suffix('_tst').T])

    return data


def load_sim_files(data_modalities, sim_directory):
    _similarity_dict = {}
    for file in tqdm(os.listdir(sim_directory), 'Reading Similarity Matrices...'):
        if any([file.startswith(x) for x in data_modalities]):
            if file.endswith('_sim.tsv'):
                _similarity_dict[file[:file.find('_sim.tsv')]] = pd.read_csv(
                    os.path.join(sim_directory, file), sep='\t', index_col=0)
    if len(_similarity_dict) < len(data_modalities):
        raise Exception(
            f"{', '.join(set(data_modalities) - set(_similarity_dict.keys()))} data is missing!")
    return _similarity_dict


def separate_sim(data):
    train_sim = {}
    test_sim = {}
    for i in data:
        test_cols = [col for col in data[i].columns if '_tst' in col]
        test_sim[i] = data[i].loc[test_cols]
        test_sim[i] = test_sim[i].drop(test_cols, axis=1)
        test_sim[i].index = test_sim[i].index.str.rstrip('_tst')
        train_sim[i] = data[i]
        train_sim[i] = train_sim[i].drop(test_cols, axis=1).drop(test_cols, axis=0)

    return train_sim, test_sim


def run(k, is_test=False):
    """
    Run the training and evaluation process k times.

    Parameters:
    - k (int): Number of times to run the process.
    - is_test (bool): If True, run on test data; otherwise, perform train-validation split.

    Returns:
    - history (dict): Dictionary containing evaluation metrics for each run.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Step 1: Initialize a dictionary to store evaluation metrics
    history = {'AUC': [], 'AUPRC': [], "Accuracy": [], "Precision": [], "Recall": [], "F1 score": []}

    # Step 2: Load training data
    train_data, train_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                            raw_file_directory= RAW_BOTH_DATA_FOLDER,
                                                            screen_file_directory= BOTH_SCREENING_DATA_FOLDER,
                                                            drug_directory= DRUG_DATA_FOLDER,
                                                            sep="\t")




    # Step 3: Load test data if applicable
    if is_test:
        test_data, test_drug_screen = RawDataLoader.load_data(data_modalities=DATA_MODALITIES,
                                                              raw_file_directory=CCLE_RAW_DATA_FOLDER,
                                                              screen_file_directory=CCLE_SCREENING_DATA_FOLDER,
                                                              sep="\t",
                                                              drug_directory=DRUG_DATA_FOLDER
                                                              )
        train_data, test_data = RawDataLoader.data_features_intersect(train_data, test_data)

        data = combine_test_train(train_data, test_data)
        build_similarity_matrices(data, SIM_DATA_FOLDER)
        similarity_dict = load_sim_files(DATA_MODALITIES, SIM_DATA_FOLDER)
        train_sim, test_sim = separate_sim(similarity_dict)

        x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(train_sim,
                                                                                                       train_drug_screen)
        x_cell_test, x_drug_test, y_test, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(test_sim,
                                                                                                    test_drug_screen)
    else:
        build_similarity_matrices(train_data, SIM_DATA_FOLDER)
        similarity_dict = load_sim_files(DATA_MODALITIES, SIM_DATA_FOLDER)
        x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes = RawDataLoader.prepare_input_data(similarity_dict,
                                                                                                       train_drug_screen)
    rus = RandomUnderSampler(sampling_strategy="majority", random_state=RANDOM_SEED)
    dataset = pd.concat([x_cell_train, x_drug_train], axis=1)
    dataset.index = x_cell_train.index
    dataset, y_train = rus.fit_resample(dataset, y_train)
    x_cell_train = dataset.iloc[:, :sum(cell_sizes)]
    x_drug_train = dataset.iloc[:, sum(cell_sizes):]


    # Step 5: Loop over k runs
    for i in range(k):
        print('Run {}'.format(i))

        # Step 6: If is_test is True, perform random under-sampling on the training data
        if is_test:

            # Step 7: Train and evaluate the DeepDRA model on test data
            results = train_DeepDRK(x_cell_train, x_cell_test, x_drug_train, x_drug_test, y_train, y_test, cell_sizes,
                                    drug_sizes, device)

        else:
            # # Step 8: Split the data into training and validation sets
            # X_cell_train, X_cell_test, X_drug_train, X_drug_test, y_train, y_test = train_test_split(X_cell_train,
            #                                                                                          X_drug_train, y_train,
            #                                                                                          test_size=0.2,
            #                                                                                          random_state=RANDOM_SEED,
            #                                                                                          shuffle=True)
            # # Step 9: Train and evaluate the DeepDRA model on the split data
            # results = train_DeepDRA(X_cell_train, X_cell_test, X_drug_train, X_drug_test, y_train, y_test, cell_sizes,
            #                         drug_sizes, device)

            results = cv_train(x_cell_train, x_drug_train, y_train, cell_sizes, drug_sizes, device, k=5)

        # Step 10: Add results to the history dictionary
        Evaluation.add_results(history, results)

    # Step 11: Display final results
    Evaluation.show_final_results(history)
    return history


if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    run(10, is_test=True)
