periods = 10
steps = 1000
steps_per_period = steps / periods  
feature_columns = [ terms_feature_column ]

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    optimizer=my_optimizer,
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
    steps=steps_per_period*(period+1))
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
