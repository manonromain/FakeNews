import numpy as np
import matplotlib.pyplot as plt
from dataset import DatasetBuilder
from functools import partial
from scipy.optimize import least_squares
import pandas as pd
import time
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score


def preprocess_graph(graph_raw):
    """
    input: data point from the "raw" dataset builder
    output: list of [graph_label, graph_root_id, edge_info = [node_index_in, node_index_out, time_out, uid_in, uid_out]]
    """
    edges = [(elt[2], elt[3]) for elt in graph_raw]
    times = [elt[4] for elt in graph_raw]
    label = graph_raw[0][0]
    graph_id = graph_raw[0][1]
    return label, graph_id, edges, times


def engagement(data_point, time_lim):
    labl, graph_id, edges, times = preprocess_graph(data_point)
    number = 0
    n = len(times)
    ids_set = {edges[0][0]}
    numbers = np.zeros(n, dtype=float)
    times = np.array(times)
    for i in range(n):
        node_out = edges[i][1]
        if node_out not in ids_set:
            ids_set.add(node_out)
            number += 1
        numbers[i] = number
    indexer = times < time_lim
    return times[indexer], numbers[indexer]


# SEIZ fit parameters = optimization starting point and bounds for variables
# Parameters
# β S-I contact rate
# b S-Z contact rate
# ρ E-I contact rate
# epsilon Incubation rate
# 1/epsilon Average Incubation Time
# bl Effective rate of S -> Z
# βρ Effective rate of S -> I
# b(1-l) Effective rate of S -> E via contact with Z
# β(1 − p) Effective rate of S -> E via contact with I
# l S->Z Probability given contact with skeptics
# 1-l S->E Probability given contact with skeptics
# p S->I Probability given contact with adopters
# 1-p S->E Probability given contact with adopters
# S_0, E_0, I_0, Z_0, N_0 initial compartments sizes
x_0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 100, 0, 1, 100])
my_bounds = ([0., 0., 0., 0., 0., 0., 0, 0, 1, 0], [1., 1., 1., 1., 1., 1., 1e8, 1e8, 1e8, 1e8])


def solve_seiz_ode(x0, jac_seiz, num_steps, max_time):
    """
    Simple Forward-Euler method to solve the SEIZ ODE
    :param x0: ode params (6 elts) + starting populations (4 elts)
    :param jac_seiz: jacobian of the ode
    :param num_steps: nb of steps in discretization
    :param max_time: time limit for ode resolution
    :return: values of s, e, i, z at every point in the discretization, times in the discretization
    """
    timestep = max_time / (num_steps - 1)  # note that the number of points returned is num_steps + 1
    values = np.zeros((num_steps, 4), dtype=float)
    seiz = np.copy(x0[-4:])
    params = np.copy(x0[:6])
    values[0, :] = seiz
    for step in range(num_steps - 1):
        seiz += timestep * jac_seiz(seiz, params)
        values[step + 1, :] = seiz
    return values, np.linspace(0., max_time, num_steps)


def seiz_func(seiz, params):
    """
    params of the ODE:
    beta, b, rho, epsilon, l, p
    """
    s, e, i, z = seiz
    beta, b, rho, epsilon, l, p = params
    n = seiz.sum()
    ds = - beta * s * i / n - b * s * z / n
    de = (1 - p) * beta * s * i / n + (1 - l) * b * s * z / n - rho * e * i / n - epsilon * e
    di = p * beta * s * i / n + rho * e * i / n + epsilon * e
    dz = l * b * s * z / n
    return np.array([ds, de, di, dz])


def cost_function_tweets(parameters, target_values, target_times, num_euler_steps, max_time):
    """
    :param parameters: parameters of the ODE = params + initial s, e, i , z pops
    :param target_values: actual number of retweets over time for the data point considered
    :param target_times: time points looked at to compute the LS cost
    :param num_euler_steps: number of discretization steps for Euler resolution
    :param max_time: time limit for ODE resolution and prediction time
    :return: cost not squared = diff between fit and target_values at the target_times
    """
    fitted_values, time_points = solve_seiz_ode(parameters, seiz_func, num_euler_steps, max_time)
    fitted_values = np.interp(target_times, time_points, fitted_values[:, 2])
    return fitted_values - target_values


def seiz_fit(datapoint, nevals, time_lim, euler_steps, fitting_points, verbose):
    """
    :param datapoint: datapoint from raw dataset builder for which we wish to compute optimal SEIZ params and make classification
    :param nevals = nb of optimization steps
    :param time_lim: time at prediction, we observe retweets over time in [0, time_lim]
    :param euler_steps: nb steps for Euler disc
    :param fitting_points: nb of points in [0., time_lim] looked at in LS cost
    :param verbose: control printing
    :return: optimal params
    """
    xc, yc = engagement(datapoint, time_lim)
    times = np.linspace(0., time_lim, num=fitting_points)
    retweets_over_time = np.interp(times, xc, yc)
    cost_func = partial(cost_function_tweets, target_values=retweets_over_time, target_times=times,
                        num_euler_steps=euler_steps, max_time=time_lim)
    optimal_result = least_squares(cost_func, x0=x_0, bounds=my_bounds, method='trf', loss='linear', verbose=0,
                                   max_nfev=nevals)
    if verbose >= 2:
        opt_params = optimal_result.x
        final_fitted_values = cost_func(opt_params) + retweets_over_time
        plt.plot(times, final_fitted_values, color='red')
        plt.plot(times, retweets_over_time, color='black')
        plt.grid()

    if verbose >= 1:
        print(
            f"Optimal fit for datapoint got {np.sqrt(optimal_result.cost / fitting_points):.2f} RMSE on number of retweets over time")
        print(
            f"Estimates: N={int(optimal_result.x[-4:].sum()):d}, S0={int(optimal_result.x[-4])}, E0={int(optimal_result.x[-3])}, I0={int(optimal_result.x[-2])}, Z0={int(optimal_result.x[-1])}")

    return optimal_result.x, optimal_result.cost


def dump_seiz_dataset(raw_dataset, name, time_lim=120., euler_steps=200, fitting_points=100, n_evals=100, verbose=0):
    df_columns = ["label", "ID", "beta", "b", "rho", "epsilon", "l", "p", "S0", "E0", "I0", "Z0", "RMSE_fit"]
    df_data = []
    total = len(raw_dataset)
    start = time.time()
    for i, dp in enumerate(raw_dataset):
        if len(dp):
            try:
                opt_params, cost = seiz_fit(dp, nevals=n_evals, time_lim=time_lim, euler_steps=euler_steps,
                                            fitting_points=fitting_points, verbose=verbose)
                df_data.append(dp[0][:2] + list(opt_params) + [np.sqrt(cost / fitting_points)])
            except ValueError:
                pass
        if (i + 1) % 5 == 0:
            time_spent = time.time() - start
            progress = 100. * (i + 1) / total
            print(
                f"{progress:.2f} % DONE, in {time_spent:.2f} seconds. Total would be {time_spent * 100 / (progress * 60):.2f} mins")

    df = pd.DataFrame(data=df_data, columns=df_columns)
    df.label = df.label.astype('category')
    df.to_csv(f"seiz_dataset_{name}.csv", index=False)


if __name__ == "__main__":
    dataset_selected = 'twitter16'
    # Building a SEIZ dataset
    dataset_builder = DatasetBuilder(dataset_selected, only_binary=False, time_cutoff=10000)
    full_dataset = dataset_builder.create_dataset(dataset_type="raw", standardize_features=False)
    train_set = full_dataset['train']
    # dump_seiz_dataset(train_set, name=dataset_selected)
    dump_seiz_dataset(full_dataset['val'], name=dataset_selected + '_val')
    dump_seiz_dataset(full_dataset['test'], name=dataset_selected + '_test')

    dataset_selected = 'twitter15'
    dataset_builder = DatasetBuilder(dataset_selected, only_binary=False, time_cutoff=10000)
    full_dataset = dataset_builder.create_dataset(dataset_type="raw", standardize_features=False)
    train_set = full_dataset['train']
    dump_seiz_dataset(train_set, name=dataset_selected)
    dump_seiz_dataset(full_dataset['val'], name=dataset_selected + '_val')
    dump_seiz_dataset(full_dataset['test'], name=dataset_selected + '_test')
