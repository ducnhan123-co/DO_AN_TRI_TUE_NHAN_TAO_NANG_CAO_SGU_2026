import nbformat
from nbclient import NotebookClient

def run_interactive():
    nb_path = 'tictactoe_interactive.ipynb'
    print("Executing tictactoe_interactive.ipynb...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=120, kernel_name='python3')
    client.execute()

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Done executing tictactoe_interactive.ipynb!")

if __name__ == '__main__':
    run_interactive()

