import nbformat
from nbclient import NotebookClient

def run_dots():
    nb_path = 'assignment_dots_and_boxes.ipynb'
    print("Executing assignment_dots_and_boxes.ipynb...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Done executing assignment_dots_and_boxes.ipynb!")

if __name__ == '__main__':
    run_dots()

