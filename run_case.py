import pandas as pd
from nn.network import Network


def run_case(n, hyperparameter_file='hyperparameters.csv'):
    """
    Runs the dataset and network defined on line n in the given hyperparameter file.
    :param n: The line of the scenario.
    :param hyperparameter_file: A CSV file containing references to data sets and hyperparameters.
    :return:
    """
    params = pd.read_csv(hyperparameter_file).loc[n - 1]

    dataset = params['Dataset']
    try:
        dataset = eval(dataset)
    except NameError:
        pass

    output_functions = params['Output_functions']
    try:
        output_functions = eval(output_functions)
    except NameError:
        pass

    network = Network(
        eval(params['Network']),
        dataset,
        minibatch_size=int(params['Minibatch_size']),
        steps=int(params['Steps']),
        loss_function=params['Loss_function'],
        output_functions=output_functions,
        case_fraction=float(params['case_fraction']),
        validation_fraction=float(params['validation_fraction']),
        validation_interval=int(params['validation_interval']),
        test_fraction=float(params['test_fraction']),
        learning_rate=float(params['learning_rate']),
        optimizer=params['optimizer'],
        one_hot_encode_target=bool(params['one_hot_encode_target'])
    )

    network.build()
    network.train()
    network.test()

    # network.mapping_test(10, [0])
    network.mapping_test(dataset, [0], [0])

    network.visualize_weights(weight_layers=[0, 1], bias_layers=[0, 1])


run_case(11)