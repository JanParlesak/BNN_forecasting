from functions import *
from models import *
# from data_processing import data, test_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

test_x_one, test_y_one = data_process(test_data, 3, 1, 1)
train_x_one, train_y_one = data_process(data, 3, 1, 1)

test_x_multi, test_y_multi = data_process(test_data, 3, 3, 1)
train_x_multi, train_y_multi = data_process(data, 3, 3, 1)

#One-step CNN

model_cnn_one = BNN_onestep_cnn()
pred_cnn_one, plus_error_cnn_one, minus_error_cnn_one, rmse_pred_cnn_one, mape_pred_cnn_one, mae_pred_cnn_one, dist_cnn_one, auc_cnn_one = prediction(model_cnn_one, 3, train_x_one, test_x_one, train_y_one, test_y_one, method = 'svi', plots = True)


model_cnn_multi = BNN_multistep_cnn()
pred_cnn_multi, plus_error_cnn_multi, minus_error_cnn_multi, rmse_pred_cnn_multi, mape_pred_cnn_multi, mae_pred_cnn_multi, dist_cnn_multi, auc_cnn_multi = prediction(model_cnn_multi, 3, train_x_mulit, test_x_mulit, train_y_multi, test_y_multi, method = 'svi', plots = True)


model_linear_one = BNN_linear_onestep()
pred_linear_one, plus_error_linear_one, minus_error_linear_one, rmse_pred_linear_one, mape_pred_linear_one, mae_pred_linear_one, dist_linear_one, auc_linear_one = prediction(model_linear_one, 3, train_x_one, test_x_one, train_y_one, test_y_one, method = 'svi', plots = True)


model_linear_one = BNN_linear_multistep()
pred_linear_multi, plus_error_linear_multi, minus_error_linear_multi, rmse_pred_linear_multi, mape_pred_linear_multi, mae_pred_linear_multi, dist_linear_multi, auc_linear_multi = prediction(model_linear_multi, 3, train_x_multi, test_x_multi, train_y_multi, test_y_multi, method = 'svi', plots = True)

