import argparse
from Acquisition.acquisition import load_train_dataset
from Acquisition.acquisition import load_val_dataset
from Acquisition.acquisition import get_test_dataset
from Wrangling.wrangling import autotune_dataset
from Wrangling.wrangling import dataset_augmentation


def argument_parser():
    """
    parse arguments to script
    """
    parser = argparse.ArgumentParser(description='getting faces images')
    args = parser.parse_args()
    return args


def main(arguments):
    print('hi')
    train_dataset = load_train_dataset()
    print('Getting train dataset')
    train_dataset = dataset_augmentation(train_dataset)
    print('Data augmentation')
    validation_dataset = load_val_dataset()
    print('Getting val dataset')
    test_dataset = get_test_dataset()
    print('Getting test dataset')
    test_dataset = autotune_dataset(test_dataset)
    validation_dataset = autotune_dataset(validation_dataset)


if __name__ == '__main__':
    print('Starting final project...')
    my_arguments = argument_parser()
    main(my_arguments)
