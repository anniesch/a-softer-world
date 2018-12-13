#!/usr/bin/env sh

lstmtraining --stop_training \
  --continue_from $1 \
  --traineddata ~/tesseract/tessdata/eng.traineddata \
  --model_output loveletter.traineddata
