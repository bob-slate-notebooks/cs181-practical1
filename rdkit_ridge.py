import csv
import gzip
import numpy as np
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error
from math import sqrt
from rdkit.Chem import AllChem

train_filename = 'train.csv.gz'
#test_filename = 'train.csv.gz'
#test_filename  = 'test.csv.gz'

# Load the training file.
train_data = []
with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)

    # Load the data. 
    counter = 0

    for row in train_csv: 
        smiles   = row[0]
        if counter >= 30000:
            break
        #features = np.array([float(x) for x in row[1:257]])
        gap      = float(row[257])
        mol = AllChem.MolFromSmiles(smiles)
        features = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useFeatures=True)
        counter+=1
        train_data.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })

# Compute the mean of the gaps in the training data.
gaps_train = np.array([datum['gap'] for datum in train_data])
features_train = np.array([datum['features'] for datum in train_data])

# Run this cell when uploading to Kaggle
pred_filename  = 'rdk_ridge_prediction2.csv'
test_filename = 'test.csv.gz'

# Load the test file.
test_data = []
with gzip.open(test_filename, 'r') as test_fh:

    # Parse it as a CSV file.
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(test_csv, None)

    # Load the data.
    for row in test_csv:
        id       = row[0]
        smiles   = row[1]
        features = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useFeatures=True)

        test_data.append({ 'id':       id,
                           'smiles':   smiles,
                           'features': features })
        
# Get features of the test data
features_test = np.array([datum['features'] for datum in test_data])
# Get ids of the test data
ids_test = np.array([datum['id'] for datum in test_data])

alpha = 0.9
clf = linear_model.Ridge(alpha, max_iter=100000000)
clf = clf.fit(features_train, gaps_train) # feature vectors (row is ohhhone data point), label/gap/thing
result_gaps = clf.predict(features_test) # call on feature vectors # parameter is second half of the data

# Write a prediction file.
with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    for i in xrange(0, len(ids_test)):
        pred_csv.writerow([ids_test[i], result_gaps[i]])