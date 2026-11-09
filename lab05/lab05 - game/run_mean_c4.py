import nbformat
from nbclient import NotebookClient

def run_mean_c4():
    nb_path = 'assignment_mean_connect4.ipynb'
    print("Executing assignment_mean_connect4.ipynb...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Done executing assignment_mean_connect4.ipynb!")

if __name__ == '__main__':
    run_mean_c4()

