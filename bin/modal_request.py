import modal
import time


if __name__ == '__main__':
    generate = modal.Function.lookup("CrystaLLM", "CrystaLLMModel.generate")

    st = time.time()
    result = generate.call(inputs={"comp": "Na1Cl1"})
    print(f"elapsed: {time.time() - st:.3f} s")

    print(result)
