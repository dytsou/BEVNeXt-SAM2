# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[args.interval - 1]]:
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}')

            if args.mode == 'eval':
                if min(epochs) == args.interval:
                    x0 = args.interval
                else:
                    # if current training is resumed from previous checkpoint
                    # we lost information in early epochs
                    # `xs` should start according to `min(epochs)`
                    if min(epochs) % args.interval == 0:
                        x0 = min(epochs)
                    else:
                        # find the first epoch that do eval
                        x0 = min(epochs) + args.interval - \
                            min(epochs) % args.interval
                xs = np.arange(x0, max(epochs) + 1, args.interval)
                ys = []
                resume_points = []  # Track resume points for marking
                
                for epoch in epochs[args.interval - 1::args.interval]:
                    ys += log_dict[epoch][metric]
                    # Check if this epoch was resumed
                    if '_resumed' in log_dict[epoch]:
                        resume_points.append(epoch)

                # if training is aborted before eval of the last epoch
                # `xs` and `ys` will have different length and cause an error
                # check if `ys[-1]` is empty here
                if not log_dict[epoch][metric]:
                    xs = xs[:-1]

                ax = plt.gca()
                ax.set_xticks(xs)
                plt.xlabel('epoch')
                
                # Plot main curve
                line = plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')[0]
                
                # Mark resume points with different markers
                if resume_points:
                    resume_xs = [x for x in xs if x in resume_points]
                    resume_ys = [ys[list(xs).index(x)] for x in resume_xs if x in xs]
                    plt.scatter(resume_xs, resume_ys, marker='s', s=100, 
                              color=line.get_color(), alpha=0.7, 
                              label=f'{legend[i * num_metrics + j]} (resumed)', 
                              edgecolors='black', linewidth=2)
                    
            else:
                xs = []
                ys = []
                resume_points = []
                
                # Use global_step if available for better x-axis
                use_global_step = 'global_step' in log_dict[epochs[0]]
                
                if use_global_step:
                    # Plot using global steps for accurate resume tracking
                    for epoch in epochs[args.interval - 1::args.interval]:
                        global_steps = log_dict[epoch].get('global_step', [])
                        metric_values = log_dict[epoch][metric]
                        
                        if len(global_steps) == len(metric_values):
                            xs.extend(global_steps)
                            ys.extend(metric_values)
                        
                        # Mark resume points
                        if '_resumed' in log_dict[epoch]:
                            resume_points.extend(global_steps[:1])  # First step of resumed epoch
                    
                    plt.xlabel('global step')
                    
                else:
                    # Fallback to original iteration-based plotting
                    num_iters_per_epoch = \
                        log_dict[epochs[args.interval-1]]['iter'][-1]
                    for epoch in epochs[args.interval - 1::args.interval]:
                        iters = log_dict[epoch]['iter']
                        if log_dict[epoch]['mode'][-1] == 'val':
                            iters = iters[:-1]
                        epoch_xs = np.array(iters) + (epoch - 1) * num_iters_per_epoch
                        xs.append(epoch_xs)
                        ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                        
                        # Mark resume points
                        if '_resumed' in log_dict[epoch]:
                            resume_points.extend([epoch_xs[0]])  # First iteration of resumed epoch
                    
                    xs = np.concatenate(xs)
                    ys = np.concatenate(ys)
                    plt.xlabel('iter')
                
                # Plot main curve
                line = plt.plot(xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)[0]
                
                # Mark resume points
                if resume_points:
                    # Find corresponding y values for resume points
                    resume_ys = []
                    for resume_x in resume_points:
                        # Find closest x value
                        closest_idx = np.argmin(np.abs(np.array(xs) - resume_x))
                        resume_ys.append(ys[closest_idx])
                    
                    plt.scatter(resume_points, resume_ys, marker='v', s=80, 
                              color=line.get_color(), alpha=0.8, 
                              label=f'{legend[i * num_metrics + j]} (resumed)', 
                              edgecolors='black', linewidth=1)
                    
            plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['mAP_0.25'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)
    parser_plt.add_argument('--mode', type=str, default='train')
    parser_plt.add_argument('--interval', type=int, default=1)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    
    for json_log, log_dict in zip(json_logs, log_dicts):
        resume_detected = False
        last_global_step = 0
        
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                
                epoch = log.pop('epoch')
                
                # Detect resume by checking for step discontinuity
                current_global_step = log.get('global_step', log.get('step', 0))
                if current_global_step < last_global_step:
                    resume_detected = True
                    print(f"Resume detected in {json_log} at epoch {epoch} (step {current_global_step} < {last_global_step})")
                
                last_global_step = current_global_step
                
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                    # Mark if this epoch started from a resume
                    if resume_detected:
                        log_dict[epoch]['_resumed'] = [True]
                        resume_detected = False  # Reset flag
                
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    
    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
