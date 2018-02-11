import matplotlib.pyplot as plt


def plot_loss(train_loss, test_loss):
    fig, ax = plt.subplots()
    plt.title('Training Loss')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss (Cost Function)')
    plt.plot(train_loss, 'b-', label='training data loss')
    plt.plot(test_loss, 'r--', label='test data loss')
    ax.legend(loc='upper right', shadow=True)
    plt.grid()
    plt.show()


def print_progress(current_epoch, epochs, train_loss, test_loss):
    print('Epoch #' + str(current_epoch + 1) + ' of ' + str(epochs))
    print('Training data loss: ', train_loss)
    print('Testing data loss: ', test_loss)