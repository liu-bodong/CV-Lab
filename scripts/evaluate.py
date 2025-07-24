# scripts/evaluate.py
def evaluate_model(model_path, test_data_path, config):
    model = load_model(model_path)
    test_dataset = BrainMRIDataset(test_data_path, split='test')
    
    metrics = compute_metrics(model, test_dataset)
    save_results(metrics, f"results_{model_name}.json")
    
    return metrics