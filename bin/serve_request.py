import requests
import json
import time


if __name__ == '__main__':
    url = "http://ec2-34-199-200-179.compute-1.amazonaws.com/predictions/cif_model_20"

    data = {"comp": "Ba6Mn3Cr3"}

    st = time.time()
    response = requests.post(url, data=json.dumps(data))
    elapsed = time.time() - st

    print(response.text)
    print(f"elapsed: {elapsed:.4f} s")
