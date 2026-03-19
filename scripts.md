## Run main script for hotpot dataset classification

### Running validation partition

python -m main --action classification --dataset hotpot --partition validation

### Running train partition

python -m main --action classification --dataset hotpot --partition train

### Running validation partition

python -m main --action classification --dataset indoqa --partition test

### Running train partition

python -m main --action classification --dataset indoqa --partition train

python -m main --action classification --dataset indoqa --partition train --from 1123 --to 2100

python -m main --action classification --dataset qasina

### Build context

python -m main --action seed_context

### Test context

python -m main --action test_context --dataset qasina

### Train classifier

python -m main --action train-classifier

python -m main --action train-classifier --undersample

## FINAL EXPERIMENT

python -m main --action experiment --experiment_type non-retrieval --dataset indoqa

python -m main --action experiment --experiment_type non-retrieval --dataset qasina

## BOOTSTRAP TESTING & STATISTICAL ANALYSIS

### Bootstrap Testing Scripts

# Run bootstrap experiments for all system types on a dataset
python -m main --action experiment --dataset qasina --experiment_type all --bootstrap --bootstrap_samples 10

# Run bootstrap for specific system type
python -m main --action experiment --dataset indoqa --experiment_type adaptive --bootstrap --bootstrap_samples 10

# Bootstrap with custom number of samples
python -m main --action experiment --dataset qasina --experiment_type non-retrieval --bootstrap --bootstrap_samples 20

### Aggregating Bootstrap Results

# Aggregate bootstrap results for all system types
python -m main --action experiment --dataset qasina --experiment_type all --aggregate

# Aggregate results for specific system type
python -m main --action experiment --dataset indoqa --experiment_type multi-retrieval --aggregate

### Generating Paper Tables

# Generate performance and comparison tables for all systems
python -m main --action experiment --dataset qasina --experiment_type all --generate_tables

# Generate tables for specific dataset
python -m main --action experiment --dataset indoqa --experiment_type all --generate_tables

### Statistical Comparisons

# Run all-pair statistical comparisons between all systems
python -m main --action experiment --dataset qasina --experiment_type all --compare

# Statistical comparison for specific dataset
python -m main --action experiment --dataset indoqa --experiment_type all --compare

### Complete Workflow Example

# Complete workflow: Run bootstrap → Aggregate → Generate tables
python -m main --action experiment --dataset qasina --experiment_type all --bootstrap --bootstrap_samples 10
python -m main --action experiment --dataset qasina --experiment_type all --aggregate
python -m main --action experiment --dataset qasina --experiment_type all --generate_tables