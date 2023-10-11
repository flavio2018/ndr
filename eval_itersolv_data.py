import pandas as pd
from main import initialize
from itersolv_data.test_dataset import TestDataset
from itersolv_data.wrapper import GeneratorWrapper


def main():
    helper, task = initialize()

    if task.helper.args.itersolv_testall:
        valid_ood_kwargs = {
            "batch_size": task.helper.args.batch_size,
            "nesting": 'all',
            "num_operands": 'all',
            "split": 'valid',
            "s2e_baseline": True,
            "exact": False,
        }
        task.valid_sets.ood = TestDataset(task.train_set.generator, valid_ood_kwargs)
        task.create_loaders()
        test, loss = task.validate_on_name('ood')
        print(f'ood_{nesting}_{n_operands}', test.accuracy)
        return

    valid_ood_kwargs = {
        "batch_size": task.helper.args.batch_size,
        "nesting": 2,
        "num_operands": 2,
        "split": 'valid',
        "s2e_baseline": True,
        "exact": False,
    }

    accuracy_table = pd.DataFrame(index=[2, 3, 4], columns=[2, 3, 4])

    for nesting in range(2, 5):
        for n_operands in range(2, 5):
            valid_ood_kwargs["nesting"] = nesting
            valid_ood_kwargs["num_operands"] = n_operands
            # task.valid_sets.ood = TestDataset(task.train_set.generator, valid_ood_kwargs)
            task.valid_sets.ood = GeneratorWrapper(task.train_set.generator, valid_ood_kwargs)
            task.create_loaders()
            test, loss = task.validate_on_name('ood')
            print(f'ood_{nesting}_{n_operands}', test.accuracy)
            accuracy_table[nesting][n_operands] = test.accuracy

    task_name = task.valid_sets.ood.task_name
    print(accuracy_table)
    accuracy_table.to_csv(f'../test_ndr_outputs/accuracy_tables/{task_name}.csv')


if __name__ == '__main__':
    main()