from main import initialize
from itersolv_data.wrapper import GeneratorWrapper
from itersolv_data.listops import ListOpsGenerator


def main():
    helper, task = initialize(f'wandb/run-20230916_055501-2msreb3f/files/checkpoint/model-100000.pth')

    valid_ood_kwargs = {
        "batch_size": task.helper.args.batch_size,
        "max_depth": 2,
        "max_args": 2,
        "split": 'test',
        "s2e_baseline": True,
        "exact": True,
    }

    for nesting in range(2, 5):
        for n_operands in range(2, 5):
            valid_ood_kwargs["max_depth"] = nesting
            valid_ood_kwargs["max_args"] = n_operands
            task.valid_sets.ood = GeneratorWrapper(task.train_set.generator, valid_ood_kwargs)
            task.create_loaders()
            test, loss = task.validate_on_name('ood')
            print(f'ood_{nesting}_{n_operands}', test.accuracy)
    return
    # test, loss = task.validate_on_name('ood_2_3')
    # test, loss = task.validate_on_name('ood_2_4')
    # test, loss = task.validate_on_name('ood_3_2')
    # test, loss = task.validate_on_name('ood_3_3')
    # test, loss = task.validate_on_name('ood_3_4')
    # test, loss = task.validate_on_name('ood_4_2')
    # test, loss = task.validate_on_name('ood_4_3')
    # test, loss = task.validate_on_name('ood_4_4')


if __name__ == '__main__':
    main()