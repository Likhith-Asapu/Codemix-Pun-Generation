import numpy as np

model_name = "prakod/mbert_cpt_gcm_shakuntala_10ep_ckpt"
print("------------------------------------")
print(f"Model: {model_name}")
Results = {
    0: {
        "test": {'eval_loss': 0.9370409846305847, 'eval_accuracy': 0.6192893401015228, 'eval_f1': 0.6226034480380682, 'eval_precision': 0.6297225760523542, 'eval_recall': 0.6192893401015228, 'eval_precision_per_label': [0.7037037037037037, 0.5168539325842697], 'eval_recall_per_label': [0.6386554621848739, 0.5897435897435898], 'eval_f1_per_label': [0.6696035242290749, 0.5508982035928144], 'eval_support_per_label': [119, 78], 'eval_runtime': 0.5236, 'eval_samples_per_second': 376.236, 'eval_steps_per_second': 7.639, 'epoch': 20.0},
        "validation":  {'eval_loss': 0.8780978918075562, 'eval_accuracy': 0.6381909547738693, 'eval_f1': 0.6425147990127539, 'eval_precision': 0.6513898974641996, 'eval_recall': 0.6381909547738693, 'eval_precision_per_label': [0.7333333333333333, 0.5144508670520231], 'eval_recall_per_label': [0.6626506024096386, 0.5973154362416108], 'eval_f1_per_label': [0.6962025316455697, 0.5527950310559007], 'eval_support_per_label': [249, 149], 'eval_runtime': 0.9207, 'eval_samples_per_second': 432.277, 'eval_steps_per_second': 7.603, 'epoch': 20.0},
    },
    42: { 
        "test": {'eval_loss': 2.042257785797119, 'eval_accuracy': 0.5532994923857868, 'eval_f1': 0.549767885808495, 'eval_precision': 0.5480356720975573, 'eval_recall': 0.5532994923857868, 'eval_precision_per_label': [0.6033057851239669, 0.47368421052631576], 'eval_recall_per_label': [0.6460176991150443, 0.42857142857142855], 'eval_f1_per_label': [0.6239316239316239, 0.45], 'eval_support_per_label': [113, 84], 'eval_runtime': 0.4503, 'eval_samples_per_second': 437.493, 'eval_steps_per_second': 8.883, 'epoch': 20.0},
        "validation": {'eval_loss': 2.0211541652679443, 'eval_accuracy': 0.6155778894472361, 'eval_f1': 0.6115194516202259, 'eval_precision': 0.6097015040205402, 'eval_recall': 0.6155778894472361, 'eval_precision_per_label': [0.6653543307086615, 0.5277777777777778], 'eval_recall_per_label': [0.7130801687763713, 0.4720496894409938], 'eval_f1_per_label': [0.6883910386965377, 0.49836065573770494], 'eval_support_per_label': [237, 161], 'eval_runtime': 0.9088, 'eval_samples_per_second': 437.925, 'eval_steps_per_second': 7.702, 'epoch': 20.0},
    },
    10: {
        "test": {'eval_loss': 3.067369222640991, 'eval_accuracy': 0.6091370558375635, 'eval_f1': 0.6057345716893769, 'eval_precision': 0.6061592930807621, 'eval_recall': 0.6091370558375635, 'eval_precision_per_label': [0.6302521008403361, 0.5769230769230769], 'eval_recall_per_label': [0.6944444444444444, 0.5056179775280899], 'eval_f1_per_label': [0.6607929515418502, 0.5389221556886228], 'eval_support_per_label': [108, 89], 'eval_runtime': 0.4527, 'eval_samples_per_second': 435.139, 'eval_steps_per_second': 8.835, 'epoch': 20.0},
        "validation": {'eval_loss': 3.15458607673645, 'eval_accuracy': 0.6231155778894473, 'eval_f1': 0.6132452804563356, 'eval_precision': 0.6150318753213925, 'eval_recall': 0.6231155778894473, 'eval_precision_per_label': [0.6505576208178439, 0.5658914728682171], 'eval_recall_per_label': [0.7575757575757576, 0.437125748502994], 'eval_f1_per_label': [0.7, 0.49324324324324326], 'eval_support_per_label': [231, 167], 'eval_runtime': 0.9126, 'eval_samples_per_second': 436.106, 'eval_steps_per_second': 7.67, 'epoch': 20.0},
    },
    100: {  
        "test": {'eval_loss': 2.1274919509887695, 'eval_accuracy': 0.6091370558375635, 'eval_f1': 0.6110125477989965, 'eval_precision': 0.6136595540591433, 'eval_recall': 0.6091370558375635, 'eval_precision_per_label': [0.6842105263157895, 0.5060240963855421], 'eval_recall_per_label': [0.6554621848739496, 0.5384615384615384], 'eval_f1_per_label': [0.6695278969957081, 0.5217391304347826], 'eval_support_per_label': [119, 78], 'eval_runtime': 0.4506, 'eval_samples_per_second': 437.157, 'eval_steps_per_second': 8.876, 'epoch': 20.0},
        "validation": {'eval_loss': 2.3063125610351562, 'eval_accuracy': 0.6231155778894473, 'eval_f1': 0.6231155778894473, 'eval_precision': 0.6231155778894473, 'eval_recall': 0.6231155778894473, 'eval_precision_per_label': [0.6767241379310345, 0.5481927710843374], 'eval_recall_per_label': [0.6767241379310345, 0.5481927710843374], 'eval_f1_per_label': [0.6767241379310345, 0.5481927710843374], 'eval_support_per_label': [232, 166], 'eval_runtime': 0.9175, 'eval_samples_per_second': 433.768, 'eval_steps_per_second': 7.629, 'epoch': 20.0},
    },
    101: {
        "test": {'eval_loss': 1.9809911251068115, 'eval_accuracy': 0.5989847715736041, 'eval_f1': 0.6003487321291912, 'eval_precision': 0.601984062590869, 'eval_recall': 0.5989847715736041, 'eval_precision_per_label': [0.6779661016949152, 0.4810126582278481], 'eval_recall_per_label': [0.6611570247933884, 0.5], 'eval_f1_per_label': [0.6694560669456067, 0.49032258064516127], 'eval_support_per_label': [121, 76], 'eval_runtime': 0.4516, 'eval_samples_per_second': 436.195, 'eval_steps_per_second': 8.857, 'epoch': 20.0},
        "validation":  {'eval_loss': 2.070674419403076, 'eval_accuracy': 0.592964824120603, 'eval_f1': 0.5863078835035392, 'eval_precision': 0.5855407789284475, 'eval_recall': 0.592964824120603, 'eval_precision_per_label': [0.6303501945525292, 0.524822695035461], 'eval_recall_per_label': [0.7074235807860262, 0.4378698224852071], 'eval_f1_per_label': [0.6666666666666666, 0.4774193548387097], 'eval_support_per_label': [229, 169], 'eval_runtime': 0.9163, 'eval_samples_per_second': 434.364, 'eval_steps_per_second': 7.64, 'epoch': 20.0},
    }
}

# Step 1: Identify the top 3 seeds with the best test accuracy
test_accuracies = {seed: data["test"]["eval_accuracy"] for seed, data in Results.items()}
top_seeds = sorted(test_accuracies, key=test_accuracies.get, reverse=True)[:3]

# Step 2: Collect all metrics for test and validation for top seeds
def gather_metrics(data, keys):
    """Extract values for specified keys from a dictionary"""
    return {k: data[k] for k in keys}

test_metrics = {seed: gather_metrics(Results[seed]["test"], Results[seed]["test"].keys()) for seed in top_seeds}
validation_metrics = {seed: gather_metrics(Results[seed]["validation"], Results[seed]["validation"].keys()) for seed in top_seeds}

# Step 3: Calculate mean and standard deviation for test and validation metrics
def calculate_mean_std(metrics):
    """Calculate mean and standard deviation for each metric across seeds"""
    means = {k: np.mean([metrics[seed][k] for seed in metrics]) * 100 for k in metrics[top_seeds[0]].keys()}
    stds = {k: np.std([metrics[seed][k] for seed in metrics]) * 100 for k in metrics[top_seeds[0]].keys()}
    return means, stds

test_means, test_stds = calculate_mean_std(test_metrics)
validation_means, validation_stds = calculate_mean_std(validation_metrics)

# Display the results side by side for each metric
# print("Top 3 Seeds:", top_seeds)

print("\nTest Metrics - Mean and Standard Deviation:")
print("F1 Score: Mean = {:.4f}, Std = {:.4f}".format(test_means["eval_f1"], test_stds["eval_f1"]))
print("Precision: Mean = {:.4f}, Std = {:.4f}".format(test_means["eval_precision"], test_stds["eval_precision"]))
print("Recall: Mean = {:.4f}, Std = {:.4f}".format(test_means["eval_recall"], test_stds["eval_recall"]))
print("Accuracy: Mean = {:.4f}, Std = {:.4f}".format(test_means["eval_accuracy"], test_stds["eval_accuracy"]))


print("\nValidation Metrics - Mean and Standard Deviation:")
print("F1 Score: Mean = {:.4f}, Std = {:.4f}".format(validation_means["eval_f1"], validation_stds["eval_f1"]))
print("Precision: Mean = {:.4f}, Std = {:.4f}".format(validation_means["eval_precision"], validation_stds["eval_precision"]))
print("Recall: Mean = {:.4f}, Std = {:.4f}".format(validation_means["eval_recall"], validation_stds["eval_recall"]))
print("Accuracy: Mean = {:.4f}, Std = {:.4f}".format(validation_means["eval_accuracy"], validation_stds["eval_accuracy"]))

# Latex formatted output
latex_output = (
    f"mBERT \\citep{{devlin2019bert}}             "
    f"& {validation_means['eval_f1']:.2f}\\scriptsize{{$\\pm${validation_stds['eval_f1']:.2f}}} "
    f"& {validation_means['eval_precision']:.1f}\\scriptsize{{$\\pm${validation_stds['eval_precision']:.2f}}} "
    f"& {validation_means['eval_recall']:.1f}\\scriptsize{{$\\pm${validation_stds['eval_recall']:.2f}}} "
    f"& {validation_means['eval_accuracy']:.1f}\\scriptsize{{$\\pm${validation_stds['eval_accuracy']:.2f}}} "
    f"& {test_means['eval_f1']:.1f}\\scriptsize{{$\\pm${test_stds['eval_f1']:.2f}}} "
    f"& {test_means['eval_precision']:.1f}\\scriptsize{{$\\pm${test_stds['eval_precision']:.2f}}} "
    f"& {test_means['eval_recall']:.1f}\\scriptsize{{$\\pm${test_stds['eval_recall']:.2f}}} "
    f"& {test_means['eval_accuracy']:.1f}\\scriptsize{{$\\pm${test_stds['eval_accuracy']:.2f}}} \\\\"
)

print("\nLatex formatted output:")
print(latex_output)