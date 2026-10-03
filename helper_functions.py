import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

def rmse(inputs, targets):
  return np.sqrt(mean_squared_error(inputs, targets))
  
def test_params(Class_model, train_inputs,
                val_inputs, train_targets,
                val_targets, **params):

  model = Class_model(**params).fit(train_inputs, train_targets)
  train_rmse = rmse(model.predict(train_inputs), train_targets)
  val_rmse = rmse(model.predict(val_inputs), val_targets)
  return train_rmse, val_rmse

def test_params_and_plot(Class_model,param_name, param_values,
                         train_inputs, val_inputs, train_targets,
                         val_targets, **other_params):

  train_error, val_error = [],[]
  for value in param_values:
    params = dict(**other_params)
    params[param_name] = value
    train_rmse, val_rmse = test_params(Class_model, train_inputs, val_inputs, train_targets, val_targets, **params)
    train_error.append(train_rmse)
    val_error.append(val_rmse)
  plt.title('Overfitting Curve')
  plt.plot(param_values, train_error, 'g-o', label = 'train_loss')
  plt.plot(param_values, val_error, 'r-x', label = 'val_loss')
  plt.xlabel('Curve of '+param_name)
  plt.ylabel('Model_loss')
  plt.legend();

from IPython.display import FileLink
def predict_and_submit(model, inputs, file_name):
  test_preds = model.predict(inputs)
  submission_df = pd.read_csv('data/sampleSubmission.csv')
  submission_df['WeeklySales'] = test_preds
  submission_df.to_csv(file_name, index = None)
  return FileLink(file_name)

def predict_and_plot_conf_mat(model, inputs, targets, name = ''):
  preds = model.predict(inputs)
  accuracy = accuracy_score(targets, preds)
  print(f'Accuracy : {accuracy*100:.2f}%')

  cf = confusion_matrix(targets, preds, normalize = 'true')
  sns.heatmap(cf, annot = True)
  plt.title(f'{name} Confusion Matrix')
  plt.xlabel('Predictions')
  plt.ylabel('Targets');
