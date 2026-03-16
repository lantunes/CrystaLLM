import modal
import time


if __name__ == '__main__':
    generate = modal.Function.from_name("CrystaLLM", "CrystaLLMModel.generate")
    st = time.time()
    result = generate.remote(inputs={"comp": "Na1Cl1"})
    print(f"elapsed: {time.time() - st:.3f} s")

    print(result)
