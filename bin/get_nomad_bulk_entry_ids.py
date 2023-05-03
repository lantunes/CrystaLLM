import requests
import json
from tqdm import tqdm

BASE_URL = 'http://nomad-lab.eu/prod/v1/api/v1'


if __name__ == '__main__':
    out_dir = "../out/nomad_entries"
    page_size = 1000
    curr_count = 0
    query_counter = 0
    next_page = None

    pbar = None

    while True:
        query = {
            'query': {
                'structural_type': {
                    'all': [
                        'bulk'
                    ]
                }
            },
            'pagination': {
                'page_size': page_size
            },
            'required': {
                'include': [
                    'material_id',
                    'elements_exclusive',
                    'chemical_formula_descriptive',
                    'chemical_formula_reduced',
                    'entries.entry_id',
                    'entries.results.method.method_name',
                    'entries.results.properties.geometry_optimization.final_force_maximum',
                    'entries.results.properties.geometry_optimization.final_energy_difference',
                ]
            }
        }
        if next_page is not None:
            query["pagination"]["page_after_value"] = next_page

        response = requests.post(f'{BASE_URL}/materials/query', json=query)

        if response.status_code != 200:
            raise Exception(f"invalid status code: {response.status_code}")

        response_json = response.json()

        if pbar is None:
            pbar = tqdm(total=response_json["pagination"]["total"])

        results = response_json["pagination"]["page_size"]
        pbar.update(results)
        curr_count += results

        query_counter += 1
        with open(f"{out_dir}/resp{query_counter}.json", "wt") as f:
            json.dump(response_json, f)

        next_page = response_json["pagination"].get("next_page_after_value")
        if not next_page:
            break
