## Run main script for musique dataset classification

### Running validation partition

python -m main --action classification --dataset musique --partition validation

### Running train partition

python -m main --action classification --dataset musique --partition train

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