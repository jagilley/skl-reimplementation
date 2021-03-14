from your_code import GradientDescent, load_data

print('Starting example experiment')

train_features, test_features, train_targets, test_targets = \
    load_data('blobs', fraction=0.6)
learner = GradientDescent('squared')
learner.fit(train_features, train_targets)
predictions = learner.predict(test_features)

print('Finished example experiment')
