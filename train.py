#!/usr/bin/env python3
from optparse import OptionParser
import numpy as np
import os
import random
from subprocess import call
import re

def train():
    parser = OptionParser()
    parser.add_option("-n", "--test_models", dest="nmodels", type=int,
            default=1, help="number of test model each iterations")
    parser.add_option("-d", "--directory", dest="directory",
            default="saved_models", help="model directory")
    (options, _) = parser.parse_args()
    dirs = os.listdir(options.directory)
    p = re.compile('G[\d]+')
    dirs = list(filter(lambda x: p.match(x), dirs))
    n = list(map(lambda x: int(x[1:]), dirs))
    if not n:
        latest = 0
    else:
        latest = max(n)
    paths = [os.path.join(options.directory, d) for d in dirs]
    models = list(filter(os.path.isdir, paths))
    models.sort(key = lambda x: os.path.getmtime(x))
    models = [os.path.relpath(m, options.directory) for m in models]

    new_model = 'G' + str(latest + 1)

    if not n:
        print()
        print("***** NEW  GENERATION {} *****".format(new_model))
        print()
        command = ["./gomoku.py", "--agent1", "dqntrain", "--agent2", "random", "--agent1_model", new_model, "--concurrency", "1024", "check_stop", "4096"]
        call(command)
        return

    command = ["./gomoku.py", "--clone", "--copy_from", models[-1], "--copy_to", new_model]
    call(command)

    weights = [1 / x for x in range(1, latest + 1)]
    w = sum(weights)
    weights = list(reversed(list(map(lambda x: x / w, weights))))
    if (latest < options.nmodels):
        test_models = models
    else:
        test_models = np.random.choice(models, replace=False, size = options.nmodels, p = weights)
    testgen = ','.join(test_models)
    print()
    print("***** NEW  GENERATION {} *****".format(new_model))
    print("*** TEST GENERATION {} ***".format(testgen))
    print()
    command = ["./gomoku.py", "--agent1", "dqntrain",
            "--agent2", "dqntest", "--agent1_model",
            new_model, "--agent2_model", testgen,
            "--concurrency", "512", "--boardsize", "9"]
    call(command)

if __name__ == "__main__":
    while True:
        train()
