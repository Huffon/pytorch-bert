import argparse

def main(config):
    params = Params('config/params.json')

    if config.mode == 'train':
        pass

    else:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT')
    parser.add_argument('--mode', type=str, default='traine', choices=['train', 'test'])
    args = parser.parse_args()
    main(args)
