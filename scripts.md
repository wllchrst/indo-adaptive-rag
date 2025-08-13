## Run main script for musique dataset classification

### Running validation partition
python -m main --action classification --dataset musique --partition validation
### Running train partition
python -m main --action classification --dataset musique --partition train

### Running validation partition
python -m main --action classification --dataset indoqa --partition test
### Running train partition
python -m main --action classification --dataset indoqa --partition train

python -m main --action classification --dataset qasina

### Build context
python -m main --action seed_context

### Test context
python -m main --action test_context --dataset qasina
