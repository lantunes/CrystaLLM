import requests
import json
import time


if __name__ == '__main__':
    url = "http://0.0.0.0:2326"

    data = {
        "comp": "Na1Cl1",
        "sg": "Pm-3m",
    }

    st = time.time()
    response = requests.post(url, data=json.dumps(data))
    elapsed = time.time() - st

    print(response.text)
    print(f"elapsed: {elapsed:.4f} s")
