#!/bin/bash

for i in {0..4}
do
  echo "Submitting FAKE Fold $i..."
  
  az ml job create -f ./jobs/test_job.yml \
    --set display_name="Fake Train - fold $i" \
    --set inputs.fold=$i
    
  echo "Fold $i submitted!"
  echo "-------------------"
done