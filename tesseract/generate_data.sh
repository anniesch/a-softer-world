#!/usr/bin/env sh

~/tesseract/src/training/tesstrain.sh --fonts_dir /home/watsonc/.local/share/fonts --lang eng --linedata_only  --noextract_font_properties --langdata_dir ~/langdata   --tessdata_dir ~/tesseract/tessdata --output_dir ./lovelettertrain --fontlist "Love LetterTW"
