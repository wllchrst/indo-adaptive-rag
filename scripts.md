## Run main script for musique dataset classification

### Running validation partition
python -m main --action classification --dataset musique --partition validation --testing False --context False
### Running train partition
python -m main --action classification --dataset musique --partition train --testing False --context False

### Build all context for elastic search
python -m main --action seed_context

###
python -m main --action test_context --datase qasina