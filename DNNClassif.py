!wget https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/terms.txt -O /tmp/terms.txt

# Create a feature column from "terms", using a full vocabulary file.
informative_terms = None
with open("/tmp/terms.txt", 'r') as f:
  # Convert it to a set first to remove duplicates.
  informative_terms = list(set(f.read().split()))
  
terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms", 
                                                                                 vocabulary_list=informative_terms)

terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=20)
feature_columns = [ terms_embedding_column ]

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

classifier = tf.estimator.DNNClassifier(
  feature_columns=feature_columns,
  hidden_units=[10,10],
  optimizer=my_optimizer
)

# Train the model, but do so inside a loop so that we can periodically assess
# loss metrics.

print "Training model..."

for period in range (0, periods):
  # Train the model, starting from the prior state.
  print "training step"
  classifier.train(
    input_fn=lambda: _input_fn([train_path]),
    steps=steps_per_period)

  evaluation_metrics = classifier.evaluate(
    input_fn=lambda: _input_fn([train_path]),
    steps=1000)
  print "Training set metrics:"
  for m in evaluation_metrics:
    print m, evaluation_metrics[m]
  print "---"
print "step finished"

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: _input_fn([test_path]),
  steps=1000)

print "Test set metrics:"
for m in evaluation_metrics:
  print m, evaluation_metrics[m]
print "---"
