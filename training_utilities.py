# Some imports
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib_inline.backend_inline
import matplotlib
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from datasets_utilities import class_mapping

# Some style settings
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
matplotlib.rcParams.update({'font.size': 5})

# Function that computes accuracy
def accuracy(labels, outputs):
    correct = 0
    total = 0
    predicted = torch.max(outputs.data, 1)[1]
    for i in range(len(labels)):
        if labels[i] == predicted[i]:
            correct += 1
        total += 1
    return correct / total

# Function that trains a CNN for one epoch
def train_one_epoch(model, loss_function, optimizer, loader):
    running_loss = 0.
    running_accuracy = 0.
    for i, data in enumerate(loader):
        inputs, labels = data

        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        running_loss += loss.item()

        running_accuracy += accuracy(labels, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = running_loss/len(loader)
        acc = running_accuracy/len(loader)
    return loss, acc

# Function to plot loss and accuracy over epochs
def plot_loss_acc(best_epoch, train_loss, validation_loss, train_accuracy, validation_accuracy):

    print("\n")

    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # Loss subplot
    ax1.plot(range(len(train_loss)), train_loss, label='Train Loss', marker='.')
    ax1.plot(range(len(validation_loss)), validation_loss, label='Validation Loss', marker='.')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs', fontsize=8, fontweight='bold')
    ax1.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Set ticks on x axis as integers
    ax1.legend()
    ax1.grid(True)

    # Accuracy subplot
    ax2.plot(range(len(train_accuracy)), train_accuracy, label='Train Accuracy', marker='.')
    ax2.plot(range(len(validation_accuracy)), validation_accuracy, label='Validation Accuracy', marker='.')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs', fontsize=8, fontweight='bold')
    ax2.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch')
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Set ticks on x axis as integers
    ax2.legend()
    ax2.grid(True)

    # Tight layout to avoid overlapping
    plt.tight_layout()
    # Show subplots
    plt.show()


# Function to check performance on test-set, return accuracy and, optionally, plot the confusion matrix
def testSet_satatistics(model, show_confusion_matrix, test_loader):
    print("____________________________________________________________________________________________________________")
    print("TEST-SET STATISTICS:")

    y_pred = []
    y_true = []

    correct = 0
    total = 0
    current_test_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            current_test_accuracy += accuracy(labels, outputs)

            y_pred.extend(predicted) # Save Prediction
            y_true.extend(labels) # Save Truth

    test_accuracy = current_test_accuracy / len(test_loader)
    print('Accuracy of the network on the test set: %d %%' % (100 * test_accuracy))

    # Build confusion matrix
    if show_confusion_matrix:
        print("\n")
        classes=list(class_mapping.keys())
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])
        plt.figure(figsize = (9,6))
        plt.title("Test-set Confusion Matrix", fontsize=8, fontweight='bold')
        sn.heatmap(df_cm, annot=True)
        #plt.savefig('output.png')

    return test_accuracy

# Function to train a model of CNN and, optionally, print some statistics
def train_CNN(model, MAX_epochs, MAX_patience, learning_rate, train_loader,
                                  validation_loader, test_loader, momentum = 0,
                                  show_update_loss_acc = True, show_plots_loss_acc = True,
                                  show_confusion_matrix = True, test_testSet = True, regularization_rate=0,
                                  optimizer_function= optim.SGD):

    # Set loss function and optimizer
    loss_function = nn.CrossEntropyLoss()

    # Set the specified optimizer
    if (optimizer_function== optim.SGD):
        optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum = momentum, weight_decay= regularization_rate)
    elif(optimizer_function== optim.Adam):
        optimizer = optim.Adam(model.parameters(), lr= learning_rate, weight_decay= regularization_rate)
    else:
        raise ValueError("Only optim.SGD or optim.Adam are supported")


    # Inizialize some variables
    best_validation_loss = np.inf   # To save best validation loss obtained
    best_epoch = 0
    patience_counter = 0    # To use an early stopping algorithm
    validation_loss = []    # To save the loss on the validation set for every epoch during training
    train_loss = []    # To save the loss on the training batch for every epoch during training
    validation_accuracy = []    # To save the accuracy on the validation set for every epoch during training
    train_accuracy = []    # To save the accuracy on the training batch for every epoch during training

    # Current time stamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Training cycle
    for epoch in range(MAX_epochs):
        print('Staring epoch {:>3d} ...'.format(epoch))

        # Make sure gradient tracking is on, and perform one epoch
        model.train(True)
        t_loss, t_accuracy = train_one_epoch(model, loss_function, optimizer, train_loader)

        # Append current loss and accuracy on the training set
        train_loss.append(t_loss)
        train_accuracy.append(t_accuracy)

        # If using dropout and/or batch normalization we need the following to set the model to evaluation mode, disabling dropout and using population
        #   statistics for batch normalization.
        model.eval()

        # Evaluate the loss and the accuracy on the validation set
        running_validation_loss = 0.0
        running_validation_accuracy = 0.0
        with torch.no_grad():      # Disable gradient computation and reduce memory consumption.
            for i, v_data in enumerate(validation_loader):
                v_inputs, v_labels = v_data
                v_outputs = model(v_inputs)
                # Loss
                v_loss = loss_function(v_outputs, v_labels)
                running_validation_loss += v_loss
                # Accuracy
                v_accuracy = accuracy(v_labels, v_outputs)
                running_validation_accuracy += v_accuracy

        current_validation_loss = running_validation_loss / len(validation_loader)
        current_validation_accuracy = running_validation_accuracy / len(validation_loader)

        # Append average validation loss and accuracy for the running epoch
        validation_loss.append(current_validation_loss)
        validation_accuracy.append(current_validation_accuracy)


        # Display progresses
        if show_update_loss_acc:
            print('   ACCURACY: train {:>4.1f} % | validation {:4.1f} %'.format(t_accuracy*100, current_validation_accuracy*100))
            print('       LOSS: train {:>6.3f} | validation {:>6.3f}'.format(t_loss, current_validation_loss))


        # Patience for early stopping
        patience_counter = patience_counter + 1

        # Track best performance (based on validation set loss), and save the model
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            model_path = '/content/model_{}'.format(timestamp) # Only the best model will be available at the end of the training
            torch.save(model.state_dict(), model_path)
            best_epoch = epoch
            patience_counter = 0

        # Check if stopping criterion is met
        if patience_counter > MAX_patience:
            break

    # Display some statistics
    print("\n")
    print("TRAINING ENDED!")
    print("\n")

    print("============================================================================================================")
    print("TRAINING- and VALIDATION-SET STATISTICS")
    print(f"Best model obtained in epoch: {best_epoch}")

    # Final loss and accuracy plots
    if show_plots_loss_acc:
        plot_loss_acc(best_epoch, train_loss, validation_loss, train_accuracy, validation_accuracy)
    print("\n")

    # Load the best model and evaluate the performance on the test set
    best_model = model
    best_model.load_state_dict(torch.load(model_path))
    if test_testSet:
        test_accuracy = testSet_satatistics(best_model, show_confusion_matrix, test_loader)

    return best_model, test_accuracy, best_epoch