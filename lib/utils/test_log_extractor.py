import re
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

def main_grid(flog):

    assert os.path.isfile(flog), 'The file at the path does not exist.'
    file = open(flog, 'r')
    top1 = []
    top5 = []
    for line in file:
        match = re.search('done, Batch', line)
        if match:
            value = re.findall('[0-9]+\.[0-9]+', line)
            if value is not None:
                top1.append(float(value[5]))
                top5.append(float(value[3]))
    file.close()
    print('#points = ', len(top1))

    x = np.linspace(0, len(top1), len(top1))
    plt.plot(x, top1, linestyle='-', label='top1')
    plt.plot(x, top5, linestyle='--', label='top5')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Log extractor.')
    parser.add_argument('--path', type=str, help='The path of the file which need to be extracted.')

    args = parser.parse_args()

    path = args.path
    main_grid(path)