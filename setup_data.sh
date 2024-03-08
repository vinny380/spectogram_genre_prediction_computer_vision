# clear all the existing data

echo Deleting data....
rm blues_train_matrices/*
rm blues_train_vectors/*
rm rock_train_matrices/*
rm rock_train_vectors/*

echo Data deleted!

echo Reducing dimensionality of images!

# replace with proper path
/opt/homebrew/bin/python3 /Users/adamclarke/Desktop/Courses/Artifical_Intelligence_CISC_352/deep-learning/spectogram_genre_prediction_computer_vision/load_data.py load_data.py

echo Flattening matrices...
python3 flatten_matrices.py


