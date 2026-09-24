import onnxruntime as ort

session = ort.InferenceSession(
    "models/phi3_medium/phi3-medium-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
    providers=["CPUExecutionProvider"]
)

print("\n=== INPUTS ===")
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

print("\n=== OUTPUTS ===")
for out in session.get_outputs():
    print(out.name, out.shape, out.type)
