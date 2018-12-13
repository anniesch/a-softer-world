#!/usr/bin/env sh


lstmtraining --model_output ./loveletter \
  --continue_from eng.lstm \
  --traineddata ~/tesseract/tessdata/eng.traineddata \
  --train_listfile lovelettertrain/eng.training_files.txt \
  --max_iterations 1200
