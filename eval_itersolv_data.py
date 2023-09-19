from main import initialize


def main():
    helper, task = initialize(f'wandb/run-20230916_055501-2msreb3f/files/checkpoint/model-100000.pth')
    test, loss = task.validate_on_name('ood_2_2')
    print('ood_2_2', test.accuracy)
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