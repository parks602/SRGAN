import warnings
import argparse
from utils import readConfig
from running import run

##################################################################################

def main():
    warnings.filterwarnings(action='ignore')
    parser = argparse.ArgumentParser(description='Pytorch SRGAN')
    parser.add_argument('--config',type=str, help='Configure File', required=True)
    args        = parser.parse_args()
    fconfig     = args.config
    type_conf, common_conf, train_conf, test_conf, input_conf = readConfig(fconfig)
    ftimes      = range(2,3)
    run(type_conf, common_conf, train_conf, test_conf, input_conf, ftimes)

if __name__ == "__main__":
    main()


