from main import initialize
from itersolv_data.test_dataset import TestDataset


def main():
    helper, task = initialize()

    valid_ood_kwargs = {
        "batch_size": task.helper.args.batch_size,
        "nesting": 2,
        "num_operands": 2,
        "split": 'test',
        "s2e_baseline": True,
        "exact": True,
    }

    accuracy_table = [[None, None, None],
                      [None, None, None],
                      [None, None, None]]

    for nesting in range(2, 5):
        for n_operands in range(2, 5):
            valid_ood_kwargs["nesting"] = nesting
            valid_ood_kwargs["num_operands"] = n_operands
            task.valid_sets.ood = TestDataset(task.train_set.generator, valid_ood_kwargs)
            task.create_loaders()
            test, loss = task.validate_on_name('ood')
            print(f'ood_{nesting}_{n_operands}', test.accuracy)
            accuracy_table[nesting-2][n_operands-2] = test.accuracy

    print(accuracy_table)
    return


if __name__ == '__main__':
    main()