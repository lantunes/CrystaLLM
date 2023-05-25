import requests
import json
import time


if __name__ == '__main__':
    url = "http://localhost:8080/predictions/cif_model_19"

    data = "Na1Cl1"

    st = time.time()
    response = requests.post(url, data=json.dumps(data))
    elapsed = time.time() - st

    print(response.text)
    print(f"elapsed: {elapsed:.4f} s")
