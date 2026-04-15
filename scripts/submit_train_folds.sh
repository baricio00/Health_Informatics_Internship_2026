#!/bin/bash
#########################################
# Loop from 0 to 4
for i in {0..4}
do
  echo "Submitting Fold $i to Azure ML..."
  
  az ml job create -f train_job.yml \
    --set display_name="Myocardium SegResNet Training - fold $i" \
    --set inputs.fold=$i
    
  echo "Fold $i submitted successfully!"
  echo "--------------------------------"
done