#External modules
import os



user = os.environ['USER']
examples = ['/Users/{0}/Documents/CS230_Final_Project/data/storage_directories/150868984252'.format(user)]



def test_get_data(stordir):
    from CS230_Final_Project import CNN_input
    return CNN_input.CNNInputDataset([stordir])


def main():
    return test_get_data(examples[0])


if __name__ == '__main__':
    CNN_Input = main()
