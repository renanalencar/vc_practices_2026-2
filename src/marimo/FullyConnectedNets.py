import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell
def _(subprocess):
    # This mounts your Google Drive to the Colab VM.
    from google.colab import drive
    drive.mount('/content/drive')
    FOLDERNAME = None
    # TODO: Enter the foldername in your Drive where you have saved the unzipped
    # assignment folder, e.g. 'cs231n/assignments/assignment2/'
    assert FOLDERNAME is not None, '[!] Enter the foldername.'
    import sys
    sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))
    # Now that we've mounted your Drive, this ensures that
    # the Python interpreter of the Colab VM can load
    # python files from within it.
    import os
    os.chdir('/content/drive/My\\ Drive/$FOLDERNAME/cs231n/datasets/')
    subprocess.call(['bash', 'get_datasets.sh'])
    # This downloads the CIFAR-10 dataset to your Drive
    # if it doesn't already exist.
    #! bash get_datasets.sh
    os.chdir('/content/drive/My\\ Drive/$FOLDERNAME')
    return os, sys


@app.cell
def _(os, subprocess, sys):
    # Cell tags: pdf-ignore
    # This downloads the CIFAR-10 dataset to your local folder
    # if it doesn't already exist.
    from pathlib import Path
    orig_dir = os.getcwd()
    datasets_path = Path('cs231n/datasets')
    if not datasets_path.exists():
    # Salva o diretório original do notebook para retorno seguro
        datasets_path = Path('src/cs231n/datasets')
    datasets_dir = str(datasets_path.resolve())
    # Localiza a pasta cs231n/datasets de forma resiliente
    os.chdir('$datasets_dir')
    if sys.platform.startswith('win'):
        subprocess.call(['C:\\Program Files\\Git\\bin\\bash.exe', 'get_datasets.sh'])
    else:
        subprocess.call(['bash', 'get_datasets.sh'])
    os.chdir('-q $orig_dir')  #! "C:\Program Files\Git\bin\bash.exe" get_datasets.sh  #! bash get_datasets.sh
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-Layer Fully Connected Network
    In this exercise, you will implement a fully connected network with an arbitrary number of hidden layers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read through the `FullyConnectedNet` class in the file `cs231n/classifiers/fc_net.py`.

    Implement the network initialization, forward pass, and backward pass. Throughout this assignment, you will be implementing layers in `cs231n/layers.py`. You can re-use your implementations for `affine_forward`, `affine_backward`, `relu_forward`, `relu_backward`, and `softmax_loss` from Assignment 1. For right now, don't worry about implementing dropout or batch/layer normalization yet, as you will add those features later.
    """)
    return


app._unparsable_cell(
    """
    # Cell tags: pdf-ignore
    # Setup cell.
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from cs231n.classifiers.fc_net import *
    from cs231n.data_utils import get_CIFAR10_data
    from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
    from cs231n.solver import Solver

    # '%matplotlib inline' command supported automatically in marimo
    plt.rcParams[\"figure.figsize\"] = (10.0, 8.0)  # Set default size of plots.
    plt.rcParams[\"image.interpolation\"] = \"nearest\"
    plt.rcParams[\"image.cmap\"] = \"gray\"

    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo

    def rel_error(x, y):
        \"\"\"Returns relative error.\"\"\"
        return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
    """,
    name="_"
)


@app.cell
def _(get_CIFAR10_data):
    # Load the (preprocessed) CIFAR-10 data.
    data = get_CIFAR10_data()
    for k, v in list(data.items()):
        print(f"{k}: {v.shape}")
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial Loss and Gradient Check

    As a sanity check, run the following to check the initial loss and to gradient check the network both with and without regularization. This is a good way to see if the initial losses seem reasonable.

    For gradient checking, you should expect to see errors around 1e-7 or less.
    """)
    return


@app.cell
def _(FullyConnectedNet, eval_numerical_gradient, np, rel_error):
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print("Running check with reg = ", reg)
        model = FullyConnectedNet(
            [H1, H2],
            input_dim=D,
            num_classes=C,
            reg=reg,
            weight_scale=5e-2,
            dtype=np.float64
        )

        loss, grads = model.loss(X, y)
        print("Initial loss: ", loss)

        # Most of the errors should be on the order of e-7 or smaller.   
        # NOTE: It is fine however to see an error for W2 on the order of e-5
        # for the check when reg = 0.0
        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print(f"{name} relative error: {rel_error(grad_num, grads[name])}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As another sanity check, make sure your network can overfit on a small dataset of 50 images. First, we will try a three-layer network with 100 units in each hidden layer. In the following cell, tweak the **learning rate** and **weight initialization scale** to overfit and achieve 100% training accuracy within 20 epochs.
    """)
    return


@app.cell
def _(FullyConnectedNet, Solver, data, np, plt):
    # TODO: Use a three-layer Net to overfit 50 training examples by 
    # tweaking just the learning rate and initialization scale.
    num_train = 50
    small_data = {'X_train': data['X_train'][:num_train], 'y_train': data['y_train'][:num_train], 'X_val': data['X_val'], 'y_val': data['y_val']}
    weight_scale = 0.1
    learning_rate = 0.01
    model_1 = FullyConnectedNet([100, 100], weight_scale=weight_scale, dtype=np.float64)
    solver = Solver(model_1, small_data, print_every=10, num_epochs=20, batch_size=25, update_rule='sgd', optim_config={'learning_rate': learning_rate})
    solver.train()
    plt.plot(solver.loss_history)
    plt.title('Training loss history')
    # weight_scale = 1e-2   # 1a tentativa
    # learning_rate = 1e-4  # 1a tentativa
    plt.xlabel('Iteration')
    # weight_scale = 1e-2   # 2a tentativa
    # learning_rate = 1e-2  # 2a tentativa
    plt.ylabel('Training loss')
    plt.grid(linestyle='--', linewidth=0.5)  # 3a tentativa
    plt.show()  # 3a tentativa
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, try to use a five-layer network with 100 units on each layer to overfit on 50 training examples. Again, you will have to adjust the learning rate and weight initialization scale, but you should be able to achieve 100% training accuracy within 20 epochs.
    """)
    return


@app.cell
def _(FullyConnectedNet, Solver, data, np, plt):
    # TODO: Use a five-layer Net to overfit 50 training examples by 
    # tweaking just the learning rate and initialization scale.
    num_train_1 = 50
    small_data_1 = {'X_train': data['X_train'][:num_train_1], 'y_train': data['y_train'][:num_train_1], 'X_val': data['X_val'], 'y_val': data['y_val']}
    weight_scale_1 = 0.1
    learning_rate_1 = 0.002
    model_2 = FullyConnectedNet([100, 100, 100, 100], weight_scale=weight_scale_1, dtype=np.float64)
    solver_1 = Solver(model_2, small_data_1, print_every=10, num_epochs=20, batch_size=25, update_rule='sgd', optim_config={'learning_rate': learning_rate_1})
    solver_1.train()
    plt.plot(solver_1.loss_history)
    plt.title('Training loss history')
    # learning_rate = 2e-3  # 1a tentativa
    # weight_scale = 1e-5   # 1a tentativa
    plt.xlabel('Iteration')
    # weight_scale = 1e-1   # 2a tentativa
    # learning_rate = 1e-2  # 2a tentativa
    plt.ylabel('Training loss')
    plt.grid(linestyle='--', linewidth=0.5)  # 3a tentativa
    plt.show()  # 3a tentativa
    return


@app.cell(hide_code=True)
def _(mo):
    # Cell tags: pdf-inline
    mo.md(r"""
    ## Inline Question 1:
    Did you notice anything about the comparative difficulty of training the three-layer network vs. training the five-layer network? In particular, based on your experience, which network seemed more sensitive to the initialization scale? Why do you think that is the case?

    ## Answer:
    [FILL THIS IN]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Update rules
    So far we have used vanilla stochastic gradient descent (SGD) as our update rule. More sophisticated update rules can make it easier to train deep networks. We will implement a few of the most commonly used update rules and compare them to vanilla SGD.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SGD+Momentum
    Stochastic gradient descent with momentum is a widely used update rule that tends to make deep networks converge faster than vanilla stochastic gradient descent. See the Momentum Update section at http://cs231n.github.io/neural-networks-3/#sgd for more information.

    Open the file `cs231n/optim.py` and read the documentation at the top of the file to make sure you understand the API. Implement the SGD+momentum update rule in the function `sgd_momentum` and run the following to check your implementation. You should see errors less than e-8.
    """)
    return


@app.cell
def _(np, rel_error):
    from cs231n.optim import sgd_momentum
    N_1, D_1 = (4, 5)
    w = np.linspace(-0.4, 0.6, num=N_1 * D_1).reshape(N_1, D_1)
    dw = np.linspace(-0.6, 0.4, num=N_1 * D_1).reshape(N_1, D_1)
    v_1 = np.linspace(0.6, 0.9, num=N_1 * D_1).reshape(N_1, D_1)
    config = {'learning_rate': 0.001, 'velocity': v_1}
    next_w, _ = sgd_momentum(w, dw, config=config)
    expected_next_w = np.asarray([[0.1406, 0.20738947, 0.27417895, 0.34096842, 0.40775789], [0.47454737, 0.54133684, 0.60812632, 0.67491579, 0.74170526], [0.80849474, 0.87528421, 0.94207368, 1.00886316, 1.07565263], [1.14244211, 1.20923158, 1.27602105, 1.34281053, 1.4096]])
    expected_velocity = np.asarray([[0.5406, 0.55475789, 0.56891579, 0.58307368, 0.59723158], [0.61138947, 0.62554737, 0.63970526, 0.65386316, 0.66802105], [0.68217895, 0.69633684, 0.71049474, 0.72465263, 0.73881053], [0.75296842, 0.76712632, 0.78128421, 0.79544211, 0.8096]])
    print('next_w error: ', rel_error(next_w, expected_next_w))
    # Should see relative errors around e-8 or less
    print('velocity error: ', rel_error(expected_velocity, config['velocity']))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once you have done so, run the following to train a six-layer network with both SGD and SGD+momentum. You should see the SGD+momentum update rule converge faster.
    """)
    return


@app.cell
def _(FullyConnectedNet, Solver, data, plt):
    num_train_2 = 4000
    small_data_2 = {'X_train': data['X_train'][:num_train_2], 'y_train': data['y_train'][:num_train_2], 'X_val': data['X_val'], 'y_val': data['y_val']}
    solvers = {}
    for update_rule in ['sgd', 'sgd_momentum']:
        print('Running with ', update_rule)
        model_3 = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=0.05)
        solver_2 = Solver(model_3, small_data_2, num_epochs=5, batch_size=100, update_rule=update_rule, optim_config={'learning_rate': 0.005}, verbose=True)
        solvers[update_rule] = solver_2
        solver_2.train()
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    axes[0].set_title('Training loss')
    axes[0].set_xlabel('Iteration')
    axes[1].set_title('Training accuracy')
    axes[1].set_xlabel('Epoch')
    axes[2].set_title('Validation accuracy')
    axes[2].set_xlabel('Epoch')
    for update_rule, solver_2 in solvers.items():
        axes[0].plot(solver_2.loss_history, label=f'loss_{update_rule}')
        axes[1].plot(solver_2.train_acc_history, label=f'train_acc_{update_rule}')
        axes[2].plot(solver_2.val_acc_history, label=f'val_acc_{update_rule}')
    for ax in axes:
        ax.legend(loc='best', ncol=4)
        ax.grid(linestyle='--', linewidth=0.5)
    plt.show()
    return small_data_2, solvers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RMSProp and Adam
    RMSProp [1] and Adam [2] are update rules that set per-parameter learning rates by using a running average of the second moments of gradients.

    In the file `cs231n/optim.py`, implement the RMSProp update rule in the `rmsprop` function and implement the Adam update rule in the `adam` function, and check your implementations using the tests below.

    **NOTE:** Please implement the _complete_ Adam update rule (with the bias correction mechanism), not the first simplified version mentioned in the course notes.

    [1] Tijmen Tieleman and Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." COURSERA: Neural Networks for Machine Learning 4 (2012).

    [2] Diederik Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
    """)
    return


@app.cell
def _(np, rel_error):
    # Test RMSProp implementation
    from cs231n.optim import rmsprop
    N_2, D_2 = (4, 5)
    w_1 = np.linspace(-0.4, 0.6, num=N_2 * D_2).reshape(N_2, D_2)
    dw_1 = np.linspace(-0.6, 0.4, num=N_2 * D_2).reshape(N_2, D_2)
    cache = np.linspace(0.6, 0.9, num=N_2 * D_2).reshape(N_2, D_2)
    config_1 = {'learning_rate': 0.01, 'cache': cache}
    next_w_1, _ = rmsprop(w_1, dw_1, config=config_1)
    expected_next_w_1 = np.asarray([[-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247], [-0.132737, -0.08078555, -0.02881884, 0.02316247, 0.07515774], [0.12716641, 0.17918792, 0.23122175, 0.28326742, 0.33532447], [0.38739248, 0.43947102, 0.49155973, 0.54365823, 0.59576619]])
    expected_cache = np.asarray([[0.5976, 0.6126277, 0.6277108, 0.64284931, 0.65804321], [0.67329252, 0.68859723, 0.70395734, 0.71937285, 0.73484377], [0.75037008, 0.7659518, 0.78158892, 0.79728144, 0.81302936], [0.82883269, 0.84469141, 0.86060554, 0.87657507, 0.8926]])
    print('next_w error: ', rel_error(expected_next_w_1, next_w_1))
    # You should see relative errors around e-7 or less
    print('cache error: ', rel_error(expected_cache, config_1['cache']))
    return


@app.cell
def _(np, rel_error):
    # Test Adam implementation
    from cs231n.optim import adam
    N_3, D_3 = (4, 5)
    w_2 = np.linspace(-0.4, 0.6, num=N_3 * D_3).reshape(N_3, D_3)
    dw_2 = np.linspace(-0.6, 0.4, num=N_3 * D_3).reshape(N_3, D_3)
    m = np.linspace(0.6, 0.9, num=N_3 * D_3).reshape(N_3, D_3)
    v_2 = np.linspace(0.7, 0.5, num=N_3 * D_3).reshape(N_3, D_3)
    config_2 = {'learning_rate': 0.01, 'm': m, 'v': v_2, 't': 5}
    next_w_2, _ = adam(w_2, dw_2, config=config_2)
    expected_next_w_2 = np.asarray([[-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977], [-0.1380274, -0.08544591, -0.03286534, 0.01971428, 0.0722929], [0.1248705, 0.17744702, 0.23002243, 0.28259667, 0.33516969], [0.38774145, 0.44031188, 0.49288093, 0.54544852, 0.59801459]])
    expected_v = np.asarray([[0.69966, 0.68908382, 0.67851319, 0.66794809, 0.65738853], [0.64683452, 0.63628604, 0.6257431, 0.61520571, 0.60467385], [0.59414753, 0.58362676, 0.57311152, 0.56260183, 0.55209767], [0.54159906, 0.53110598, 0.52061845, 0.51013645, 0.49966]])
    expected_m = np.asarray([[0.48, 0.49947368, 0.51894737, 0.53842105, 0.55789474], [0.57736842, 0.59684211, 0.61631579, 0.63578947, 0.65526316], [0.67473684, 0.69421053, 0.71368421, 0.73315789, 0.75263158], [0.77210526, 0.79157895, 0.81105263, 0.83052632, 0.85]])
    print('next_w error: ', rel_error(expected_next_w_2, next_w_2))
    print('v error: ', rel_error(expected_v, config_2['v']))
    # You should see relative errors around e-7 or less
    print('m error: ', rel_error(expected_m, config_2['m']))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once you have debugged your RMSProp and Adam implementations, run the following to train a pair of deep networks using these new update rules:
    """)
    return


@app.cell
def _(FullyConnectedNet, Solver, plt, small_data_2, solvers):
    learning_rates = {'rmsprop': 0.0001, 'adam': 0.001}
    for update_rule_1 in ['adam', 'rmsprop']:
        print('Running with ', update_rule_1)
        model_4 = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=0.05)
        solver_3 = Solver(model_4, small_data_2, num_epochs=5, batch_size=100, update_rule=update_rule_1, optim_config={'learning_rate': learning_rates[update_rule_1]}, verbose=True)
        solvers[update_rule_1] = solver_3
        solver_3.train()
        print()
    fig_1, axes_1 = plt.subplots(3, 1, figsize=(15, 15))
    axes_1[0].set_title('Training loss')
    axes_1[0].set_xlabel('Iteration')
    axes_1[1].set_title('Training accuracy')
    axes_1[1].set_xlabel('Epoch')
    axes_1[2].set_title('Validation accuracy')
    axes_1[2].set_xlabel('Epoch')
    for update_rule_1, solver_3 in solvers.items():
        axes_1[0].plot(solver_3.loss_history, label=f'{update_rule_1}')
        axes_1[1].plot(solver_3.train_acc_history, label=f'{update_rule_1}')
        axes_1[2].plot(solver_3.val_acc_history, label=f'{update_rule_1}')
    for ax_1 in axes_1:
        ax_1.legend(loc='best', ncol=4)
        ax_1.grid(linestyle='--', linewidth=0.5)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    # Cell tags: pdf-inline
    mo.md(r"""
    ## Inline Question 2:

    AdaGrad, like Adam, is a per-parameter optimization method that uses the following update rule:

    ```
    cache += dw**2
    w += - learning_rate * dw / (np.sqrt(cache) + eps)
    ```

    John notices that when he was training a network with AdaGrad that the updates became very small, and that his network was learning slowly. Using your knowledge of the AdaGrad update rule, why do you think the updates would become very small? Would Adam have the same issue?

    ## Answer:
    [FILL THIS IN]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train a Good Model!
    Train the best fully connected model that you can on CIFAR-10, storing your best model in the `best_model` variable. We require you to get at least 50% accuracy on the validation set using a fully connected network.

    If you are careful it should be possible to get accuracies above 55%, but we don't require it for this part and won't assign extra credit for doing so. Later in the assignment we will ask you to train the best convolutional network that you can on CIFAR-10, and we would prefer that you spend your effort working on convolutional networks rather than fully connected networks.

    **Note:** You might find it useful to complete the `BatchNormalization.ipynb` and `Dropout.ipynb` notebooks before completing this part, since those techniques can help you train powerful models.
    """)
    return


@app.cell
def _(FullyConnectedNet, Solver, data):
    best_model = None
    hidden_dims = [100, 100, 100, 100, 100]
    ################################################################################
    # TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
    # find batch/layer normalization and dropout useful. Store your best model in  #
    # the best_model variable.                                                     #
    learning_rate_2 = 0.001
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    weight_scale_2 = 0.05
    # Definir a arquitetura da rede (5 camadas ocultas com 100 neurônios cada)
    model_5 = FullyConnectedNet(hidden_dims, weight_scale=weight_scale_2, normalization='batchnorm')
    # Definir a taxa de aprendizado adequada para o otimizador Adam
    solver_4 = Solver(model_5, data, num_epochs=10, batch_size=200, update_rule='adam', optim_config={'learning_rate': learning_rate_2}, verbose=True, print_every=100)
    # Definir a escala para inicialização dos pesos
    solver_4.train()
    # Inicializar o modelo com a arquitetura definida e usando Batch Normalization
    # Configurar o Solver para realizar o treinamento da rede
    # Iniciar o laço de treinamento do modelo
    # Armazenar o modelo treinado na variável best_model
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #                              END OF YOUR CODE                                #
    best_model = model_5  # Número de épocas de treinamento  # Tamanho de cada batch (lote de imagens)  # Utilizar o algoritmo de otimização Adam  # Passar a taxa de aprendizado  # Mostrar o progresso do treino no console
    return (best_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test Your Model!
    Run your best model on the validation and test sets. You should achieve at least 50% accuracy on the validation set.
    """)
    return


@app.cell
def _(best_model, data, np):
    y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
    y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
    print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
    print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())
    return


if __name__ == "__main__":
    app.run()
