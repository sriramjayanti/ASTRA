import json

with open('checkpoints/sample_models_evaluation.json') as f:
    data = json.load(f)

for ds_name, ds_res in data['dataset_evaluations'].items():
    print('=' * 80)
    print(f"DATASET: {ds_name} ({ds_res['total_samples']} samples)")
    print('=' * 80)
    print("--- 1. FULL 11-CLASS SYSTEM VIEW (ASTRA Target Classes) ---")
    print(f"{'Model':<38} | {'Top-1 (%)':<10} | {'Top-3 (%)':<10} | {'Macro-F1':<10} | {'Weighted-F1':<12}")
    print('-' * 90)
    for model_name, m_res in ds_res['results_full_11'].items():
        print(f"{model_name:<38} | {m_res['top1_accuracy']:>9.2f}% | {m_res['top3_accuracy']:>9.2f}% | {m_res['macro_f1']:>10.4f} | {m_res['weighted_f1']:>12.4f}")

    print("\n--- 2. 8-CLASS SUBSET (Only classes known to Sample Models) ---")
    print(f"{'Model':<38} | {'Top-1 (%)':<10} | {'Top-3 (%)':<10} | {'Macro-F1':<10} | {'Weighted-F1':<12}")
    print('-' * 90)
    for model_name, m_res in ds_res.get('results_sub_8', {}).items():
        print(f"{model_name:<38} | {m_res['top1_accuracy']:>9.2f}% | {m_res['top3_accuracy']:>9.2f}% | {m_res['macro_f1']:>10.4f} | {m_res['weighted_f1']:>12.4f}")

    print("\n--- 3. UNKNOWN FALSE POSITIVE RATE ---")
    for model_name, fpr in ds_res.get('noise_false_positive_rate', {}).items():
        print(f"  {model_name}: {fpr:.2f}% FPR")

    print("\n--- 4. PER-CLASS TOP-1 ---")
    sample_1d_pc = ds_res['results_full_11']['Sample ResNet-1D (8-class)']['per_class']
    sample_2d_pc = ds_res['results_full_11']['Sample Spectrogram 2D-CNN (8-class)']['per_class']
    sample_f_pc  = ds_res['results_full_11']['Sample Fused (1D+2D)']['per_class']
    v2_f_pc      = ds_res['results_full_11']['V2 Production Fusion (11-class)']['per_class']
    print(f"{'Class':<10} | {'Support':<8} | {'Sample 1D':<10} | {'Sample 2D':<10} | {'Sample Fused':<12} | {'V2 Fusion':<10}")
    print('-' * 75)
    for c in data['v2_classes']:
        s = sample_1d_pc[c]['support']
        s1 = sample_1d_pc[c]['top1']
        s2 = sample_2d_pc[c]['top1']
        sf = sample_f_pc[c]['top1']
        v2 = v2_f_pc[c]['top1']
        print(f"{c:<10} | {s:<8} | {s1:>9.1f}% | {s2:>9.1f}% | {sf:>11.1f}% | {v2:>9.1f}%")

    print("\n--- 5. UNSUPPORTED CLASS CONFUSION (DQPSK, MSK, 256QAM) ---")
    for u_cls, u_info in ds_res.get('unsupported_class_routing', {}).items():
        print(f"  {u_cls} (Total: {u_info['total']}):")
        print(f"    Sample 1D predicted: {u_info['Sample_1D_Predictions']}")
        print(f"    Sample 2D predicted: {u_info['Sample_2D_Predictions']}")

    print("\n--- 6. SNR STRATIFIED PERFORMANCE ---")
    print(f"{'SNR Bin':<12} | {'Samples':<8} | {'Sample 1D':<10} | {'Sample 2D':<10} | {'Sample Fused':<12} | {'V2 Fused':<10}")
    print('-' * 75)
    for snr_bin, snr_data in ds_res.get('snr_performance', {}).items():
        cnt = snr_data['count']
        s1 = snr_data['Sample_1D_Top1']
        s2 = snr_data['Sample_2D_Top1']
        sf = snr_data['Sample_Fused_Top1']
        vf = snr_data['V2_Fused_Top1']
        print(f"{snr_bin:<12} | {cnt:<8} | {s1:>9.1f}% | {s2:>9.1f}% | {sf:>11.1f}% | {vf:>9.1f}%")
    print('\n')
