import torch
from audioldm_eval import EvaluationHelper
import sys
device = torch.device(f"cuda:{0}")


generation_result_path = sys.argv[1]
# generation_result_path = "example/unpaired"
target_audio_path = sys.argv[2]

evaluator = EvaluationHelper(16000, device)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
)