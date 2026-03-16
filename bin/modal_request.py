import modal
import time

if __name__ == "__main__":
    CrystaLLMModel = modal.Cls.from_name("CrystaLLM", "CrystaLLMModel")

    model = CrystaLLMModel()

    st = time.time()
    result = model.generate.remote(inputs={"comp": "Na1Cl1"})
    print(f"elapsed: {time.time() - st:.3f} s")

    print(result)
