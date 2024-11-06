import pandas as pd
from main import initialize
from itersolv.ndr_dataset import ItersolvDataset
# from itersolv.test_dataset import TestDataset
# from itersolv.wrapper import GeneratorWrapper


def main():
    helper, task = initialize()
    task_name = task.train_set.dataset_name

    if task_name == 'listops':
        difficulty_splits = [[1, 2], [1, 3], [1, 4],
                             [2, 2], [2, 3], [2, 4],
                             [3, 2], [3, 3], [3, 4],
                             [4, 2], [4, 3], [4, 4],
                             [5, 2], [5, 3], [5, 4],
                             [6, 2], [6, 3], [6, 4]]
        accuracy_table = pd.DataFrame(index=[1, 2, 3, 4, 5, 6], columns=[2, 3, 4])

    elif task_name in ['arithmetic', 'algebra']:
        difficulty_splits = [[1, 2], [2, 2], [3, 2],
                             [4, 2], [5, 2], [6, 2]]
        accuracy_table = pd.DataFrame(index=[1, 2, 3, 4, 5, 6], columns=[2])

    elif task_name == 'logic':
        difficulty_splits = [[1, 2], [2, 2], [3, 2],
                             [4, 2], [5, 2], [6, 2],
                             [7, 2], [8, 2], [9, 2],
                             [10, 2], [11, 2], [12, 2]]
        accuracy_table = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], columns=[2])

    task.valid_sets.ood = ItersolvDataset(
        task_name,
        'test', 
        helper.args.batch_size,
        helper.args.test_batch_size,
        'cuda',
        sos=False,
        eos=False)
    task.create_loaders()
    test, loss = task.validate_on_name('ood')
    print(f'testall_acc', test.accuracy)

    pytorch_total_params = sum(p.numel() for p in task.model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in task.model.parameters() if p.requires_grad)

    print("pytorch_total_params", pytorch_total_params)
    print("pytorch_trainable_params", pytorch_trainable_params)

    for difficulty_split in difficulty_splits:
        nesting, n_operands = difficulty_split
        task.valid_sets.ood = ItersolvDataset(
            task_name,
            'test', 
            helper.args.batch_size,
            helper.args.test_batch_size,
            'cuda',
            sos=False,
            eos=False,
            difficulty_split=difficulty_split)
        task.create_loaders()
        print(len(task.valid_sets.ood))
        test, loss = task.validate_on_name('ood')
        print(f'ood_{nesting}_{n_operands}', test.accuracy)
        accuracy_table.loc[nesting, n_operands] = test.accuracy

    print(accuracy_table)
    accuracy_table.to_csv(f'../test_ndr_outputs/accuracy_tables/{task_name}.csv')


if __name__ == '__main__':
    main()