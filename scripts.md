## Run main script for musique dataset classification

### Running validation partition
python -m main --action classification --dataset musique --partition validation --testing True --context True
### Running train partition
python -m main --action classification --dataset musique --partition train --testing True --context True
