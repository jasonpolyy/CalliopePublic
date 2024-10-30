from invoke import task

@task
def black(c, commit=False):
    """
    Format all code according to Black.
    """
    c.run("black calliope scripts --line-length 88")

    if commit:
        print('Committing as BLACK REFORMAT.')
        c.run("git add .")
        c.run("git commit -m 'BLACK REFORMAT'")

@task
def kernel(c):
    """
    Create an IPuython kernel
    """
    c.run('python -m ipykernel install --user --name calliope_kernel --display-name "Python (calliope)"')


@task
def tb(c, logdir='logs', port=6006):
    """
    Open tensorboard with logdirs open
    """
    c.run(f'tensorboard --logdir={logdir} --port={port}')