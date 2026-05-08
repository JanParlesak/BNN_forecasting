import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
from sklearn.metrics import auc
import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from pyro.infer.autoguide import AutoDiagonalNormal
from collections import defaultdict

from tqdm.auto import trange


def split_sequences(sequences, n_steps):
  """
    Parameters:
      sequences:  a time series sequence
      n_steps: number of steps used for prediction
    
    Returns: n_steps of the series as input value and a single label y
  """

  X, y = list(), list()
  for i in range(len(sequences)):
      end_ix = i + n_steps
      if end_ix >= len(sequences):
          break
      seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
      X.append(seq_x)
      y.append(seq_y)
  return np.array(X), np.array(y)


def multi_time(data, n_input, n_out):

    """
    Parameters:
      data:  a time series sequence
      n_input: number of steps used for prediction
      n_out: number of steps to predict
    
    Returns: n_input of the series as input value and n_out steps as label y
  """


    X,y = list(), list()
    in_start= 0

    for i in range(len(data)):

        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            X.append(data[in_start:in_end])
            y.append(data[in_end:out_end])
            in_start +=1
    return np.array(X), np.array(y)


class myDataset(Dataset):
    """
    Parameters:
      Dataset: any dataset with item and labels
    
    Returns: items and labels
  """
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]

        return item,label

def plot_results(inputs, test_y, prediction, plus_error, minus_error, title = None, x_train = None):

  """
    Parameters:
      inputs: length of input vector used for the prediction 
      test_y: test dataset
      prediction: predictions of the network
      plus_error: vector of mean + 2 * standard deviation
      minus_error: vector of mean - 2 * standard deviation
      title: title of plot
      x_train: training data

    Returns: plot of prediction, real values and errors. Optional adds test_train split if x_train is provided
  """

  
  plt.rcParams["figure.figsize"] = (10, 7)
  x = np.arange(1, len(test_y)+1)

  plt.plot(x, test_y, label='Truth', color = 'blue')
  plt.plot(x, prediction, label='Prediction',color = 'orange')

  plt.fill_between(x, plus_error, minus_error, color='lightgray', alpha=0.5)

  if x_train is not None:
    plt.axvline(x=len(x_train), linestyle='-', color='purple', label='Train-Test Split')

  plt.xlabel('Months')
  plt.ylabel('Index')
  plt.title(title, color = 'blue')
  plt.legend()

  plt.show()


def rmse(test_data, prediction):
    

    """
    Parameters: 
      test_data: the data to test on
      prediction: the predcition made by the network
    
    Returns: 
      RMSE
  """
    mse = ((test_data - prediction)**2).sum()/len(prediction)
    return round(np.sqrt(mse), 3)

def mape(test_data, prediction):

  """
    Parameters: 
      test_data: the data to test on
      prediction: the predcition made by the network
    
    Returns: 
      MAPE
  """
  mape = abs((test_data-prediction)/test_data).sum()/len(prediction)
  return round(mape, 3)

def mae(test_data, prediction):
  
    """
    Parameters: 
      test_data: the data to test on
      prediction: the predcition made by the network
    
    Returns: 
      MAE
    """

    mae = abs(test_data - prediction).sum()/len(prediction)
    return round(mae,3)

def chi_square_cdf_probabilities(nssr, df = 1):
	"""
	Compute the probabilities of the values of nssr under a chi-square CDF with df dof.

	Parameters:
    	nssr (float or array-like): Non-centrality parameter(s).

	Returns:
    	float or ndarray: Probability value(s).
	"""
	return stats.chi2.cdf(nssr, df)



def count_elements_bigger_than_p(vector, p):
    """
    Count the number of elements in the vector that are bigger than p.

    Parameters:
        vector (list): The input vector.
        p (float): The value to compare against.

    Returns:
        int: Number of elements in the vector that are bigger than p.
    """
    return sum(1 for x in vector if x <= p)

def auc_curve(prediction, test_y, covariance_matrix, title):
  """
    Parameters:
      prediction: the prediction made by the network
      test_y: data to test on
      covariance_matrix: covariance provided as an numpy array
      title: title of the plot
    
    Returns:
      plot of the calibration curve

  """

  nssr = [(prediction[i] - test_y[i]) @  (np.linalg.inv(covariance_matrix[i])) @ (prediction[i] - test_y[i]).T for i in range(len(covariance_matrix))]
  predicted_probability = chi_square_cdf_probabilities(nssr)
  observed_probability = [count_elements_bigger_than_p(nssr,p)/len(nssr) for p in nssr]
  p_array = np.column_stack((predicted_probability.squeeze(), observed_probability))
  curve = np.row_stack(([0,0],(p_array[p_array[:, 0].argsort()])))
  plt.rcParams["figure.figsize"] = (10, 7)
  plt.plot(curve[:,0], curve[:,1])
  plt.axline([0, 0], [1, 1])
  plt.title(title, color = 'blue')

def return_auc_curve(prediction, test_y, covariance_matrix):

    """
    Parameters:
      prediction: the prediction made by the network
      test_y: data to test on
      covariance_matrix: covariance provided as an numpy array

    Returns: the calibration curve
    """
    nssr = [(prediction[i] - test_y[i]) @  (np.linalg.inv(covariance_matrix[i])) @ (prediction[i] - test_y[i]).T for i in range(len(covariance_matrix))]
    predicted_probability = chi_square_cdf_probabilities(nssr)
    observed_probability = [count_elements_bigger_than_p(nssr,p)/len(nssr) for p in nssr]
    p_array = np.column_stack((predicted_probability.squeeze(), observed_probability))
    curve = np.row_stack(([0,0],(p_array[p_array[:, 0].argsort()])))

    return curve

def distance(prediction, test_y, covariance_matrix):

    """
    Parameters:
      prediction: the prediction made by the network
      test_y: data to test on
      covariance_matrix: covariance provided as an numpy array

    Returns: the distance to the calibration curve and the area under the curve (AUC)

    """

    nssr = [(prediction[i] - test_y[i]) @  (np.linalg.inv(covariance_matrix[i])) @ (prediction[i] - test_y[i]).T for i in range(len(covariance_matrix))]
    predicted_probability = chi_square_cdf_probabilities(nssr)
    observed_probability = [count_elements_bigger_than_p(nssr,p)/len(nssr) for p in nssr]
    p_array = np.column_stack((predicted_probability, observed_probability))
    curve = np.row_stack(([0,0],(p_array[p_array[:, 0].argsort()])))
    return round(np.sqrt(auc(curve[:,0],(curve[:,1]-curve[:,0])**2)),3), round(auc(curve[:,0],curve[:,1]))

def fit_svi(model, guide, train_loader, model_name, lr=0.01, num_epochs=1000, plot=True):

  """
    Parameters:
      model: a Baysian Neural Network
      guide: the guide for the Network
      train_loader: the training data as a PyTorch dataloader
      model_name: model name for saving
      lr: learning rate for SGD
      num_epochs: number of epochs to train for
      plot: whether or not to plot the loss curve
    
    Trains the model using SVI on the training data and saves the one with lowest loss
  """

  pyro.clear_param_store()
  optimizer = pyro.optim.Adam({"lr": lr})

  svi = SVI(model, guide, optimizer, Trace_ELBO())
  progress_bar = trange(num_epochs)

  loss_best = None

  series = defaultdict(list)

  for epoch in progress_bar:
      loss=0
      for batch_id, data in enumerate(train_loader):
        loss += svi.step(data[0], data[1])

      loss = loss / len(train_loader)
      series["loss"].append(loss)
      median = guide.median()

      for name, value in median.items():
        if value.numel() == 1:
          series[name + " mean"].append(float(value))

      if loss_best is None or loss < loss_best:
        loss_best = loss
        torch.save({"model" : model.state_dict(), "guide" : guide}, f'{model_name}.pt')
        pyro.get_param_store().save(f'{model_name}_params.pt')

      progress_bar.set_postfix(loss=f"{loss:.3f}")

  if plot:
    plt.figure(figsize=(6, 6))
    for name, Y in series.items():
      if name == "loss":
        plt.plot(Y, "k--", label=name, zorder=0)
      elif name.endswith(" mean"):
        plt.plot(Y, label=name, zorder=-1)
      else:
        plt.plot(Y, label=name, alpha=0.5, lw=1, zorder=-2)
        plt.xlabel("SVI step")
        plt.title("loss, scalar parameters, and gradient norms")
        plt.yscale("log")
        plt.xscale("symlog")
        plt.xlim(0, None)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()

def predict(model, guide, test_loader, num_samples = 500):
  
  """
      Parameters:
        model: a Bayesian Neural Network
        guide: the guide for the Network
        test_loader: the test data as a PyTorch dataloader
        num_samples: number of samples to draw
      
      Returns:
        pred: prediction on the test data
        plus_error: vector of mean + 2 * standard deviation
        minus_error: vector of mean - 2 * standard deviation
        covariance: covariance matrix
        real: the test data values
    """


  predictive = Predictive(model, guide = guide, num_samples = num_samples)
  pred = []
  plus_error= []
  minus_error = []
  covariance= []
  real = []

  for batch_id, data in enumerate(test_loader):
    preds = predictive(data[0])
    y_std = preds['obs'][:,0,:].detach().numpy().std(axis=0, ddof=1)
    y_preds = preds['obs'][:,0,:].detach().numpy().sum(axis=0)/num_samples
    plus_error.append(y_preds - 2* y_std)
    minus_error.append(y_preds + 2* y_std)
    pred.append(y_preds)

    residuals = preds['obs'][:,0,:].detach().numpy() - y_preds
    Sigma_y = (residuals.T @ residuals)/(num_samples - 1)

    covariance.append(Sigma_y)

    y_real = data[1]
    real.append(y_real)

  return pred, plus_error, minus_error, covariance, real

def predict_mcmc(model, x_train, y_train, x_test, num_samples = 50):
   

   """
    Parameters:
      model: a Baysian Neural Network 
      x_train: training data items
      y_train: training data labels
      x_test : the test data
      num_samples: number of samples to draw
    
    Returns:
      preds: prediction on the test data
      plus_error_mcmc: vector of mean + 2 * standard deviation
      minus_error_mcmc: vector of mean - 2 * standard deviation
      covariance_mcmc: covariance matrix
   """


   pyro.clear_param_store()

   # define MCMC sampler
   nuts_kernel = NUTS(model, jit_compile=False)
   mcmc = MCMC(nuts_kernel, num_samples= num_samples)
   mcmc.run(x_train, y_train)


   predictive = Predictive(model = model, posterior_samples = mcmc.get_samples())
   preds_mcmc = predictive(x_test)
   residuals = preds_mcmc['obs'] - preds_mcmc['obs'].mean(dim=0)
   covariance_mcmc = (torch.transpose(residuals, 1,2) @ residuals)/(num_samples - 1)


   minus_error_mcmc= (preds_mcmc['obs'].mean(dim=0)-2*preds_mcmc['obs'].std(dim=0)).numpy().flatten()
   plus_error_mcmc= (preds_mcmc['obs'].mean(dim=0)+2*preds_mcmc['obs'].std(dim=0)).numpy().flatten()

   return preds_mcmc['obs'].mean(dim=0), plus_error_mcmc, minus_error_mcmc, covariance_mcmc



def data_process(prices, m, steps, t):

  """
    Parameters:
      prices: datalist of financial time-series
      m: embedding dimension. How many steps should be used for prediction
      steps: how many steps in the future
      t: time-lag

    Returns:
      tensors of data either for training or testing
  """

  if steps == 1:
    x, y = split_sequences(prices, m)
  else:
    x, y = multi_time(prices, m, steps)

  x = x[::t]
  y = y[::t]

  x = torch.tensor(x.reshape(x.shape[0],x.shape[1],1)).float()
  y = torch.tensor(y.reshape(y.shape[0],steps,1)).float()

  return x, y



def prediction(bnn_model, n_inputs, x_train, x_test, y_train, y_test, method = 'svi', lr = 0.01, num_epochs = 5000, plots = False):

  train_data = myDataset(x_train,y_train)
  test_data = myDataset(x_test,y_test)
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False)
  test_loader= torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

  if method == 'svi':
    guide_model = AutoDiagonalNormal(bnn_model)
    fit_svi(bnn_model, guide_model, train_loader, lr = lr, model_name = "model_one", num_epochs= num_epochs, plot = plots)
    saved = torch.load("model_one.pt")
    bnn_model.load_state_dict(saved['model'])
    guide = saved['guide']
    pyro.get_param_store().load("model_one_params.pt")

    pred, plus_error, minus_error, covariance, real = predict(bnn_model, guide, test_loader)

    if plots == True:
      plot_results(n_inputs, torch.cat(real).numpy().squeeze().flatten(), np.concatenate(pred, axis=0).flatten(), np.vstack(plus_error).squeeze().tolist(),
             np.vstack(minus_error).squeeze().tolist(), title = 'BNN Model (SVI)', x_train = x_train)

    rmse_pred = rmse(torch.cat(real).numpy().squeeze().flatten(),np.concatenate(pred, axis =0))
    mape_pred = mape(torch.cat(real).numpy().squeeze().flatten(),np.concatenate(pred, axis =0))
    mae_pred = mae(torch.cat(real).numpy().squeeze().flatten(),np.concatenate(pred, axis =0))
    dist, auc = distance(pred, torch.cat(real).numpy().squeeze().flatten(), covariance)

    return pred, plus_error, minus_error, rmse_pred, mape_pred, mae_pred, dist, auc

  if method == 'mcmc':

    preds_mcmc, plus_error_mcmc, minus_error_mcmc, covariance_mcmc = predict_mcmc(bnn_model, x_train, y_train, x_test, num_samples= 50)

    if plots == True:

      plot_results(n_inputs, y_test, preds_mcmc, plus_error_mcmc, minus_error_mcmc, title = 'BNN Model (MCMC) ', x_train = x_train)

    rmse_pred = rmse(y_test.flatten().numpy(),np.concatenate(preds_mcmc, axis =0))
    mape_pred = mape(y_test.flatten().numpy(),np.concatenate(preds_mcmc, axis =0))
    mae_pred = mae(y_test.flatten().numpy(),np.concatenate(preds_mcmc, axis =0))
    dist, auc = distance(pred, y_test.flatten().numpy(), covariance_mcmc)

    return preds_mcmc, plus_error_mcmc, minus_error, rmse_pred, mape_pred, mae_pred, dist, auc









