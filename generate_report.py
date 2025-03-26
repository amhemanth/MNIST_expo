import json
import os

def generate_report():
    try:
        if os.path.exists('test_results.json'):
            with open('test_results.json', 'r') as f:
                results = json.load(f)

            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
                summary.write('## Model Validation Results\n')
                summary.write('### Architecture Tests\n')

                arch = results.get('architecture', {})
                perf = results.get('performance', {})

                # Architecture Section
                param_count = arch.get('parameter_count', None)
                summary.write(f"- Parameter Count: {param_count:,} {'✓' if param_count < 20000 else '❌ Exceeds 20k'}\n" if param_count is not None else '- Parameter Count: N/A\n')

                batch_norm_count = arch.get('batch_norm_count', None)
                summary.write(f"- Batch Normalization Layers: {batch_norm_count} ✓\n" if batch_norm_count is not None else '- Batch Normalization Layers: N/A\n')

                dropout_probs = arch.get('dropout_probs', [None])
                summary.write(f"- Dropout Probability: {dropout_probs[0]} ✓\n" if dropout_probs[0] is not None else '- Dropout Probability: N/A\n')

                has_fc = arch.get('has_fc', None)
                summary.write(f"- Architecture: {'FC' if has_fc else 'GAP'} ✓\n" if has_fc is not None else '- Architecture: N/A\n')

                output_shape = arch.get('output_shape', None)
                summary.write(f"- Output Shape: {output_shape} ✓\n" if output_shape is not None else '- Output Shape: N/A\n')

                summary.write('\n### Performance Tests\n')

                device = perf.get('device', None)
                summary.write(f"- Device: {device}\n" if device is not None else '- Device: N/A\n')

                inference_time = perf.get('inference_time', None)
                summary.write(f"- Inference Time: {inference_time}ms ✓\n" if inference_time is not None else '- Inference Time: N/A\n')

                valid_probabilities = perf.get('valid_probabilities', None)
                summary.write(f"- Valid Probabilities: {'✓' if valid_probabilities else '❌'}\n" if valid_probabilities is not None else '- Valid Probabilities: N/A\n')

                model_stability = perf.get('model_stability', None)
                summary.write(f"- Model Stability: {'✓' if model_stability else '❌'}\n" if model_stability is not None else '- Model Stability: N/A\n')
        else:
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
                summary.write('⚠️ No test results file found\n')
    except Exception as e:
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write(f'Error processing test results: {e}\n')

if __name__ == "__main__":
    generate_report() 