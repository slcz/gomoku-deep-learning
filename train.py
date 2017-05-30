#!/usr/bin/env python3
from optparse import OptionParser
import numpy as np
import os
import random
from subprocess import call

def train():
    parser = OptionParser()
    parser.add_option("-n", "--test_models", dest="nmodels", type=int,
            default=1, help="number of test model each iterations")
    parser.add_option("-d", "--directory", dest="directory",
            default="saved_models", help="model directory")
    (options, _) = parser.parse_args()
    dirs = os.listdir(options.directory)
    paths = [os.path.join(options.directory, d) for d in dirs]
    models_ = filter(os.path.isdir, paths)
    models = [d for d in models_]
    models.sort(key = lambda x: os.path.getmtime(x))
    models = [os.path.relpath(m, options.directory) for m in models]

    n = len(models)
    new_model = 'G' + str(n + 1)

    if n == 0:
        print()
        print("***** NEW  GENERATION {} *****".format(new_model))
        print()
        command = ["./gomoku.py", "--agent1", "dqntrain", "--agent2", "random", "--agent1_model", new_model, "--concurrency", "1024", "check_stop", "4096"]
        call(command)
        return

    command = ["./gomoku.py", "--clone", "--copy_from", models[-1], "--copy_to", new_model]
    call(command)

    weights = [1 / (2 ** x) for x in range(1, n + 1)]
    w = sum(weights)
    weights = list(reversed(list(map(lambda x: x / w, weights))))
    if (n < options.nmodels):
        test_models = models
    else:
        test_models = np.random.choice(models, replace=False, size = options.nmodels, p = weights)
    testgen = ','.join(test_models)
    print()
    print("***** NEW  GENERATION {} *****".format(new_model))
    print("*** TEST GENERATION {} ***".format(testgen))
    print()
    command = ["./gomoku.py", "--agent1", "dqntrain", "--agent2", "dqntest", "--agent1_model", new_model, "--agent2_model", testgen, "--concurrency", "512", "--boardsize", "15"]
    call(command)

if __name__ == "__main__":
    while True:
        train()
