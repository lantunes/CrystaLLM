import requests
import gzip
import queue
import csv
import json
import multiprocessing as mp
from tqdm import tqdm

BASE_URL = 'http://nomad-lab.eu/prod/v1/api/v1'


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    tot = 0
    while True:
        message = queue.get()
        tot += message
        pbar.update(message)
        if tot == n:
            break


def download_chunk(progress_queue, task_queue, out_dir):

    while not task_queue.empty():
        try:
            k, chunk = task_queue.get_nowait()
        except queue.Empty:
            break

        try:
            query_json = {
                'query': {
                    'entry_id:any': chunk
                },
                'pagination': {
                    'page_size': len(chunk)
                },
                'required': {
                    'results': {
                        'material': {
                            'material_id': '*',
                            'chemical_formula_descriptive': '*',
                            'chemical_formula_reduced': '*',
                            'symmetry': {
                                'space_group_symbol': '*'
                            }
                        },
                        'properties': {
                            'geometry_optimization': {
                                'structure_optimized': {
                                    'lattice_vectors': '*',
                                    'cartesian_site_positions': '*',
                                    'species_at_sites': '*'
                                }
                            }
                        }
                    }
                }
            }
            response = requests.post(f'{BASE_URL}/entries/archive/query', json=query_json)

            if response.status_code != 200:
                raise Exception(f"invalid status code: {response.status_code}; {response.text}")

            response_json = response.json()

            with open(f"{out_dir}/resp{k}.json", "wt") as f:
                json.dump(response_json, f)

        except Exception as e:
            print(e)
            pass

        progress_queue.put(1)


if __name__ == '__main__':
    csv_fname = "../out/nomad_entries_2023_04_30.csv.gz"
    out_dir = "../out/nomad_entries_data_2023_04_30"
    ids_per_request = 10
    workers = 2

    entry_ids = []
    with gzip.open(csv_fname, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in tqdm(reader):
            entry_ids.append(line[4])

    chunks_with_k = [(k+1, entry_ids[i:i+ids_per_request]) for k, i in enumerate(range(0, len(entry_ids), ids_per_request))]

    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()

    for chunk_with_k in chunks_with_k:
        task_queue.put(chunk_with_k)

    watcher = mp.Process(target=progress_listener, args=(progress_queue, len(chunks_with_k),))

    processes = [mp.Process(target=download_chunk, args=(progress_queue, task_queue, out_dir)) for _ in range(workers)]
    processes.append(watcher)

    for process in processes:
        process.start()

    for process in processes:
        process.join()
