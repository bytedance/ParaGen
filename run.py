from paragen.entries.run import main

main()


from paragen.utils.tensor import get_avg_ckpt, save_ckpt

ckpt_list = [
    'checkpoint/last.updates-284116.epochs-148.pt',
    'checkpoint/last.updates-286062.epochs-149.pt',
    'checkpoint/last.updates-288008.epochs-150.pt',
    'checkpoint/last.updates-289954.epochs-151.pt',
    'checkpoint/last.updates-291900.epochs-152.pt',
    'checkpoint/last.updates-293846.epochs-153.pt',
    'checkpoint/last.updates-295792.epochs-154.pt',
    'checkpoint/last.updates-297738.epochs-155.pt',
    'checkpoint/last.updates-299684.epochs-156.pt',
    'checkpoint/last.updates-300001.epochs-157.pt'
]

avg_ckpt = get_avg_ckpt(ckpt_list)
save_ckpt(avg_ckpt, 'checkpoint/ckpt_avg.pt')
