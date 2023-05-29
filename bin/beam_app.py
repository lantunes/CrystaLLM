import beam
from beam.types import GpuType, PythonVersion

app = beam.App(
    name="CrystaLLM",
    cpu=8,
    memory="8Gi",
    gpu=GpuType.A10G,
    python_version=PythonVersion.Python39,
    python_packages=[
        "pandas==1.5.3",
        "numpy==1.24.2",
        "torch==2.0.1",
        "scikit-learn==1.2.2",
        "tiktoken==0.3.2",
        "transformers==4.27.3",
        "pymatgen==2023.3.23",
    ],
)

app.Trigger.RestAPI(
    inputs={"text": beam.Types.String()},
    outputs={
        "cifs": beam.Types.Json(),
    },
    handler="bin/beam_handler.py:handle_request",
    loader="bin/beam_handler.py:load_model",
)

app.Trigger.AutoScaling.MaxRequestLatency(
  desired_latency=120,
  max_replicas=1,
)

app.Mount.PersistentVolume(name="saved_models", path="./saved_models")
