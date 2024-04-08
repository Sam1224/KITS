import numpy as np


def mae(y_hat, y):
    return np.abs(y_hat - y).mean()


def nmae(y_hat, y):
    delta = np.max(y) - np.min(y) + 1e-8
    return mae(y_hat, y) * 100 / delta


def mape(y_hat, y):
    return 100 * np.abs((y_hat - y) / (y + 1e-8)).mean()


def mse(y_hat, y):
    return np.square(y_hat - y).mean()


def rmse(y_hat, y):
    return np.sqrt(mse(y_hat, y))


def nrmse(y_hat, y):
    delta = np.max(y) - np.min(y) + 1e-8
    return rmse(y_hat, y) * 100 / delta


def nrmse_2(y_hat, y):
    nrmse_ = np.sqrt(np.square(y_hat - y).sum() / np.square(y).sum())
    return nrmse_ * 100


def r2(y_hat, y):
    return 1. - np.square(y_hat - y).sum() / (np.square(y.mean(0) - y).sum())


def masked_mae(y_hat, y, mask):
    mask[y == 0] = 0
    err = np.abs(y_hat - y) * mask
    return err.sum() / mask.sum()


def masked_mape(y_hat, y, mask):
    mask[y == 0] = 0
    err = np.abs((y_hat - y) / (y + 1e-8)) * mask
    return err.sum() / mask.sum()


def masked_mse(y_hat, y, mask):
    mask[y == 0] = 0
    err = np.square(y_hat - y) * mask
    return err.sum() / mask.sum()


def masked_rmse(y_hat, y, mask):
    mask[y == 0] = 0
    err = np.square(y_hat - y) * mask
    return np.sqrt(err.sum() / mask.sum())


def masked_mre(y_hat, y, mask):
    mask[y == 0] = 0
    err = np.abs(y_hat - y) * mask
    return err.sum() / ((y * mask).sum() + 1e-8)


def masked_r2(y_hat, y, mask):
    mask[y == 0] = 0
    return 1. - (np.square(y_hat - y) * mask).sum() / ((np.square(y.mean(0) - y) * mask).sum())


def mse_loss(x, y, w, scale=None):
    # x => bs, num_nodes
    # y => bs, num_nodes
    # w => bs, num_nodes
    if scale is not None:
        x = x * scale
        y = y * scale
    unmasked_mse = (x - y) ** 2
    masked_mse = (unmasked_mse * w).sum(1)
    w_road_sum = w.sum(1)
    valid_positions = np.where(w_road_sum != 0)
    masked_mse = masked_mse[valid_positions]
    w_road_sum = w_road_sum[valid_positions]
    masked_mse = masked_mse / w_road_sum
    return masked_mse.mean()


def mae_loss(x, y, w, scale=None):
    # x => bs, num_nodes
    # y => bs, num_nodes
    # w => bs, num_nodes
    if scale is not None:
        x = x * scale
        y = y * scale
    unmasked_mae = np.abs(x - y)
    masked_mae = (unmasked_mae * w).sum(1)
    w_road_sum = w.sum(1)
    valid_positions = np.where(w_road_sum != 0)
    masked_mae = masked_mae[valid_positions]
    w_road_sum = w_road_sum[valid_positions]
    masked_mae = masked_mae / w_road_sum
    return masked_mae.mean()
