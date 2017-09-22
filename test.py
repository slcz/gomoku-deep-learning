#!/usr/bin/env python3
from optparse import OptionParser
import numpy as np
import os
import random
import subprocess
from itertools import combinations
import re

def main():
    parser = OptionParser()
    parser.add_option("-d", "--directory", dest="directory",
            default="saved_models", help="model directory")
    parser.add_option("-t", "--target", dest="target",
            default="", help="target model")
    (options, _) = parser.parse_args()
    dirs = os.listdir(options.directory)
    p = re.compile('G[\d]+')
    dirs = list(filter(lambda x: p.match(x), dirs))
    result = dict()
    #for x, y in combinations(dirs, 2):
    for x in dirs:
        if x == options.target:
            continue
        y = options.target
        command = ["python", "gomoku.py",
                             "--agent1", "dqntest",
                             "--agent2", "dqntest",
                             "--agent1_model", x,
                             "--agent2_model", y,
                             "--test_epsilon", "0.0",
                             "--concurrency", "1",
                             "--boardsize", "11",
                             "--min_games", "1",
                             "--max_games", "2"]
        print(command)
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        out, err = process.communicate()
        out = out.decode('utf-8').splitlines()[-1]
        print("OUTPUT IS {}".format(out))
        match = re.match(r'\*\s*\d*:\s*A(G\d+)\s*([\d\.]+),\s*A(G\d+)\s*([\d\.]+).*', out)
        if match:
            k1 = match.group(1)
            k2 = match.group(3)
            v1 = float(match.group(2))
            v2 = float(match.group(4))
            if k1 not in result:
                result[k1] = v1
            else:
                result[k1] += v1
            if k2 not in result:
                result[k2] = v2
            else:
                result[k2] += v2
    for k,v in result.items():
        print("{}: {}".format(k, v))

if __name__ == "__main__":
    main()
