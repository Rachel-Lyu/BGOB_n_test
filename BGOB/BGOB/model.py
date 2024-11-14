import math
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

class GRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xg = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hg = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):

        r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h))
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        g = torch.tanh(self.lin_xg(x) + self.lin_hg(r * h))

        dh = (1 - z) * (g - h)

        return dh


class GRUODECell_Autonomous(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bias = bias

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.

        Args:
            t        time
            h        hidden state (current)

        Returns:
            Updated h
        """
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh


class GRUObservationCellLogvar(torch.nn.Module):
    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):

        super().__init__()
        self.gru_d = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        std = math.sqrt(2.0 / (4 + prep_hidden))

        self.w_prep = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size = input_size
        self.prep_hidden = prep_hidden

    # def forward(self, h, p, X_obs, M_obs, i_obs):
    #     p_obs = p[i_obs]

    #     mean, logvar = torch.chunk(p_obs, 2, dim=1)
    #     sigma = torch.exp(0.5 * logvar)
    #     epsilon = 1e-6  # Small positive value
    #     sigma_clamped = torch.clamp(sigma, min=epsilon)
    #     error = (X_obs - mean) / sigma_clamped
    #     # error = (X_obs - mean) / sigma

    #     log_lik_c = np.log(np.sqrt(2 * np.pi))
    #     losses = 0.5 * ((torch.pow(error, 2) + logvar + 2 * log_lik_c) * M_obs)

    #     gru_input1 = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
    #     gru_input = torch.matmul(gru_input1, self.w_prep).squeeze(2) + self.bias_prep
    #     gru_input.relu_()
    #     # gru_input (sample x feature x prep_hidden)
    #     gru_input = gru_input.permute(2, 0, 1)
    #     gru_input = (
    #         (gru_input * M_obs)
    #         .permute(1, 2, 0)
    #         .contiguous()
    #         .view(-1, self.prep_hidden * self.input_size)
    #     )

    #     temp = h.clone()
    #     temp[i_obs] = self.gru_d(gru_input, h[i_obs])
    #     h = temp

    #     return h, losses

    def forward(self, h, p, X_obs, M_obs, i_obs):
        p_obs = p[i_obs]

        mean, logvar = torch.chunk(p_obs, 2, dim=1)

        min_logvar_value = -10
        max_logvar_value = 10
        logvar_clamped = torch.clamp(logvar, min=min_logvar_value, max=max_logvar_value)

        sigma = torch.exp(0.5 * logvar_clamped)
        epsilon = 1e-6
        max_sigma_value = 1e6
        sigma_clamped = torch.clamp(sigma, min=epsilon, max=max_sigma_value)

        error = (X_obs - mean) / sigma_clamped
        max_error_value = 1e6
        error_clamped = torch.clamp(error, min=-max_error_value, max=max_error_value)

        log_lik_c = np.log(np.sqrt(2 * np.pi))
        losses = 0.5 * ((torch.pow(error_clamped, 2) + logvar_clamped + 2 * log_lik_c) * M_obs)

        gru_input1 = torch.stack([X_obs, mean, logvar_clamped, error_clamped], dim=2).unsqueeze(2)
        gru_input = torch.matmul(gru_input1, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        gru_input = gru_input.permute(2, 0, 1)
        gru_input = (
            (gru_input * M_obs)
            .permute(1, 2, 0)
            .contiguous()
            .view(-1, self.prep_hidden * self.input_size)
        )

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, losses



def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)


class gruodebayes_original(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        p_hidden,
        prep_hidden,
        device,
        bias=True,
        use_sigmoid=False,
        mixing=1,
        dropout_rate=0,
        solver="euler",
        impute=True,
    ):

        super().__init__()
        self.device = device
        self.impute = impute
        self.use_sigmoid = use_sigmoid
        self.p_model = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )

        if impute is False:
            self.gru_c = GRUODECell_Autonomous(hidden_size, bias=bias)
        else:
            self.gru_c = GRUODECell(2 * input_size, hidden_size, bias=bias)

        self.gru_obs = GRUObservationCellLogvar(
            input_size, hidden_size, prep_hidden, bias=bias
        )

        self.solver = solver
        self.input_size = input_size

        self.mixing = mixing
        self.hidden_size = hidden_size
        self.apply(init_weights)

    def ode_step(self, h, p, delta_t, current_time):
        if self.impute is False:
            p = torch.zeros_like(p)

        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
            p = self.p_model(h)

        if self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)

            h = h + delta_t * self.gru_c(pk, k)
            p = self.p_model(h)

        current_time += delta_t
        return h, p, current_time

    def forward(
        self,
        times,
        time_ptr,
        X,
        M,
        obs_idx,
        delta_t,
        T,
        return_path=False,
    ):
        h = torch.empty(int(max(obs_idx)) + 1, self.hidden_size, device=self.device)
        torch.nn.init.uniform_(h)
        p = self.p_model(h)
        current_time = 0.0

        loss_1 = 0
        loss_2 = 0

        if return_path:
            path_t = [0]
            path_p = [p]
            path_h = [h]

        for i, obs_time in enumerate(times):
            while current_time < (obs_time - 0.0001 * delta_t):
                h, p, current_time = self.ode_step(h, p, delta_t, current_time)
                if return_path:
                    if abs(current_time - float(path_t[-1])) > 0.0001:
                        path_t.append(current_time)
                        path_p.append(p)
                        path_h.append(h)

                    else:
                        path_p[-1] = (path_p[-1] + p) / 2
                        path_h[-1] = (path_h[-1] + h) / 2

            start = time_ptr[i]
            end = time_ptr[i + 1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)

            loss_1 = loss_1 + losses.sum()
            p = self.p_model(h)

            loss_2 = loss_2 + compute_KL_loss(
                p_obs=p[i_obs], X_obs=X_obs, M_obs=M_obs, logvar=True
            )

            if return_path:
                if abs(obs_time - float(path_t[-1])) > 0.01:
                    path_t.append(obs_time)
                    path_p.append(p)
                    path_h.append(h)

                else:
                    path_p[-1] = (path_p[-1] + p) / 2
                    path_h[-1] = (path_h[-1] + h) / 2

        while current_time < T:
            h, p, current_time = self.ode_step(h, p, delta_t, current_time)

            if return_path:
                if abs(current_time - float(path_t[-1])) > 0.01:
                    path_t.append(current_time)
                    path_p.append(p)
                    path_h.append(h)

                else:
                    path_p[-1] = (path_p[-1] + p) / 2
                    path_h[-1] = (path_h[-1] + h) / 2

        loss = loss_1 + self.mixing * loss_2

        if return_path:
            return h, loss, np.array(path_t), torch.stack(path_p), torch.stack(path_h)

        else:
            return h, loss


class bidirectional_gruodebayes(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        p_hidden,
        prep_hidden,
        device,
        bias=True,
        use_sigmoid=False,
        mixing=1,
        dropout_rate=0,
        solver="euler",
        impute=True,
    ):

        super().__init__()
        self.device = device
        self.impute = impute
        self.use_sigmoid = use_sigmoid
        self.p_model_pre = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, p_hidden, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )

        if impute is False:
            self.gru_c = GRUODECell_Autonomous(hidden_size, bias=bias)
        else:
            self.gru_c = GRUODECell(2 * input_size, hidden_size, bias=bias)

        self.gru_obs = GRUObservationCellLogvar(
            input_size, hidden_size, prep_hidden, bias=bias
        )

        self.solver = solver
        self.input_size = input_size

        self.mixing = mixing
        self.hidden_size = hidden_size
        self.apply(init_weights)

    # def ode_step(self, h, p, delta_t, current_time):
    #     if self.impute is False:
    #         p = torch.zeros_like(p)

    #     if self.solver == "euler":
    #         h = h + (delta_t) * self.gru_c(p, h)
    #         p = self.p_model(h)

    #     if self.solver == "midpoint":
    #         k = h + (delta_t) / 2 * self.gru_c(p, h)
    #         pk = self.p_model(k)

    #         h = h + (delta_t) * self.gru_c(pk, k)
    #         p = self.p_model(h)

    #     current_time += delta_t
    #     return h, p, current_time
    def ode_step(self, h, p, delta_t, current_time):
        if self.impute is False:
            p = torch.zeros_like(p)

        # Define maximum allowed change in h
        max_delta_h = 1.0  # Adjust this value based on your data

        if self.solver == "euler":
            delta_h = delta_t * self.gru_c(p, h)
            # Clamp delta_h to prevent excessive updates
            delta_h = torch.clamp(delta_h, min=-max_delta_h, max=max_delta_h)
            h = h + delta_h
            p = self.p_model(h)

        elif self.solver == "midpoint":
            # First half-step
            delta_h1 = (delta_t) / 2 * self.gru_c(p, h)
            delta_h1 = torch.clamp(delta_h1, min=-max_delta_h, max=max_delta_h)
            k = h + delta_h1
            pk = self.p_model(k)
            # Second half-step
            delta_h2 = delta_t * self.gru_c(pk, k)
            delta_h2 = torch.clamp(delta_h2, min=-max_delta_h, max=max_delta_h)
            h = h + delta_h2
            p = self.p_model(h)

        current_time += delta_t
        return h, p, current_time


    # def reverse_ode_step(self, h, p, delta_t, current_time):

    #     if self.impute is False:
    #         p = torch.zeros_like(p)

    #     if self.solver == "euler":
    #         h = h + (delta_t) * self.gru_c(p, h)
    #         p = self.p_model(h)

    #     if self.solver == "midpoint":
    #         k = h + (delta_t) / 2 * self.gru_c(p, h)
    #         pk = self.p_model(k)

    #         h = h + (delta_t) * self.gru_c(pk, k)
    #         p = self.p_model(h)

    #     current_time -= delta_t
    #     return h, p, current_time

    def reverse_ode_step(self, h, p, delta_t, current_time):
        if self.impute is False:
            p = torch.zeros_like(p)

        # Define maximum allowed change in h
        max_delta_h = 1.0  # Adjust this value based on your data

        if self.solver == "euler":
            delta_h = delta_t * self.gru_c(p, h)
            # Clamp delta_h to prevent excessive updates
            delta_h = torch.clamp(delta_h, min=-max_delta_h, max=max_delta_h)
            h = h + delta_h
            p = self.p_model(h)

        elif self.solver == "midpoint":
            # First half-step
            delta_h1 = (delta_t) / 2 * self.gru_c(p, h)
            delta_h1 = torch.clamp(delta_h1, min=-max_delta_h, max=max_delta_h)
            k = h + delta_h1
            pk = self.p_model(k)
            # Second half-step
            delta_h2 = delta_t * self.gru_c(pk, k)
            delta_h2 = torch.clamp(delta_h2, min=-max_delta_h, max=max_delta_h)
            h = h + delta_h2
            p = self.p_model(h)

        current_time -= delta_t
        return h, p, current_time


    def p_model(self, h):
        p = self.p_model_pre(h)
        if self.use_sigmoid:
            out = torch.sigmoid(p)
        else:
            out = torch.exp(p)
        return out

    def forward(
        self,
        times,
        time_ptr,
        X,
        M,
        obs_idx,
        delta_t,
        T,
        return_path=False,
    ):
        h = torch.empty(int(max(obs_idx)) + 1, self.hidden_size, device=self.device)
        torch.nn.init.uniform_(h)
        p = self.p_model(h)
        current_time = 0.0

        loss_1 = []
        loss_2 = 0

        if return_path:
            path_t = [0]
            path_p = [p]
            path_h = [h]

        for i, obs_time in enumerate(times):
            while current_time < (obs_time - 0.0001 * delta_t):
                h_temp = h
                p_temp = p
                h, p, current_time = self.ode_step(h, p, delta_t, current_time)
                if return_path:
                    if abs(current_time - float(path_t[-1])) > 0.0001:
                        path_t.append(current_time)
                        path_p.append(p)
                        path_h.append(h)

                    else:
                        path_p[-1] = (path_p[-1] + p) / 2
                        path_h[-1] = (path_h[-1] + h) / 2

            start = time_ptr[i]
            end = time_ptr[i + 1]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)

            loss_1.append(losses.sum())
            p = self.p_model(h)

            loss_2 = loss_2 + compute_KL_loss(
                p_obs=p[i_obs], X_obs=X_obs, M_obs=M_obs, logvar=True
            )

            if return_path:
                if abs(obs_time - float(path_t[-1])) > 0.01:
                    path_t.append(obs_time)
                    path_p.append(p)
                    path_h.append(h)

                else:
                    path_p[-1] = (path_p[-1] + p) / 2
                    path_h[-1] = (path_h[-1] + h) / 2

        while current_time < T:
            h, p, current_time = self.ode_step(h, p, delta_t, current_time)

            if return_path:
                if abs(current_time - float(path_t[-1])) > 0.01:
                    path_t.append(current_time)
                    path_p.append(p)
                    path_h.append(h)

                else:
                    path_p[-1] = (path_p[-1] + p) / 2
                    path_h[-1] = (path_h[-1] + h) / 2

        h_bk = torch.empty(int(max(obs_idx)) + 1, self.hidden_size, device=self.device)
        torch.nn.init.uniform_(h_bk)
        p_bk = self.p_model(h_bk)
        current_time = T

        loss_1_bk = []
        loss_2_bk = 0

        if return_path:
            path_t_bk = [T]
            path_p_bk = [p_bk]
            path_h_bk = [h_bk]

        reversed_times = times[::-1]
        for i, obs_time in enumerate(reversed_times):
            while current_time - delta_t > (obs_time - 0.0001 * delta_t):
                h_bk, p_bk, current_time = self.reverse_ode_step(
                    h_bk, p_bk, delta_t, current_time
                )
                if return_path:
                    if abs(current_time - float(path_t_bk[-1])) > 0.0001:
                        path_t_bk.append(current_time)
                        path_p_bk.append(p_bk)
                        path_h_bk.append(h_bk)

                    else:
                        path_p_bk[-1] = (path_p_bk[-1] + p_bk) / 2
                        path_h_bk[-1] = (path_h_bk[-1] + h_bk) / 2

            end = time_ptr[-1 - i]
            start = time_ptr[-2 - i]

            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]

            h_bk, losses_bk = self.gru_obs(h_bk, p_bk, X_obs, M_obs, i_obs)

            loss_1_bk.append(losses_bk.sum())
            p_bk = self.p_model(h_bk)

            loss_2_bk = loss_2_bk + compute_KL_loss(
                p_obs=p_bk[i_obs], X_obs=X_obs, M_obs=M_obs, logvar=True
            )

            if return_path:
                if abs(obs_time - float(path_t_bk[-1])) > 0.01:
                    path_t_bk.append(obs_time)
                    path_p_bk.append(p_bk)
                    path_h_bk.append(h_bk)

                else:
                    path_p_bk[-1] = (path_p_bk[-1] + p_bk) / 2
                    path_h_bk[-1] = (path_h_bk[-1] + h_bk) / 2

        while current_time - delta_t >= 0:
            h_bk, p_bk, current_time = self.reverse_ode_step(
                h_bk, p_bk, delta_t, current_time
            )

            if return_path:
                if abs(current_time - float(path_t_bk[-1])) > 0.01:
                    path_t_bk.append(current_time)
                    path_p_bk.append(p_bk)
                    path_h_bk.append(h_bk)

                else:
                    path_p_bk[-1] = (path_p_bk[-1] + p_bk) / 2
                    path_h_bk[-1] = (path_h_bk[-1] + h_bk) / 2

        loss_1_all = weight_add(loss_1, loss_1_bk)

        final_loss = sum(loss_1_all) + (loss_2 + loss_2_bk) * self.mixing
        final_p = weight_add(path_p, path_p_bk)

        if return_path:
            return h, final_loss, np.array(path_t), final_p, torch.stack(path_h)

        else:
            return h, final_loss


# def weight_add(a, b):
#     path_t = range(min(len(a), len(b)))
#     time_points = np.array(path_t).reshape(-1, 1)
#     scaler = MinMaxScaler()
#     scaler.fit(time_points)
#     weight = scaler.transform(time_points).reshape(-1)
#     final_list = []
#     for i in range(len(weight)):
#         final_list.append(a[i] * weight[i] + b[-1 - i] * (1 - weight[i]))

#     return torch.stack(final_list)


def weight_add(a, b):
    path_t = range(min(len(a), len(b)))
    time_points = np.array(path_t).reshape(-1, 1)
    if time_points.max() - time_points.min() > 0:
        scaler = MinMaxScaler()
        scaler.fit(time_points)
        weight = scaler.transform(time_points).reshape(-1)
    else:
        weight = np.ones_like(time_points).reshape(-1)
    final_list = []
    for i in range(len(weight)):
        final_list.append(a[i] * weight[i] + b[-1 - i] * (1 - weight[i]))

    return torch.stack(final_list)


# def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2, logvar=True):
#     obs_noise_std = torch.tensor(obs_noise_std)
#     if logvar:
#         mean, var = torch.chunk(p_obs, 2, dim=1)
#         max_var_value = 10  # Adjust based on acceptable range
#         var_clamped = torch.clamp(var, max=max_var_value)
#         std = torch.exp(0.5 * var_clamped)
#         # std = torch.exp(0.5 * var)
#     else:
#         mean, var = torch.chunk(p_obs, 2, dim=1)
#         std = torch.pow(torch.abs(var) + 1e-5, 0.5)

#     return (
#         gaussian_KL(mu_1=mean, mu_2=X_obs, sigma_1=std, sigma_2=obs_noise_std) * M_obs
#     ).sum()
def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2, logvar=True):
    obs_noise_std = torch.tensor(obs_noise_std)
    epsilon = 1e-6
    max_std_value = 1e6
    if logvar:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        min_var_value = -10
        max_var_value = 10
        var_clamped = torch.clamp(var, min=min_var_value, max=max_var_value)
        std = torch.exp(0.5 * var_clamped)
    else:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.pow(torch.abs(var) + 1e-5, 0.5)
    std_clamped = torch.clamp(std, min=epsilon, max=max_std_value)

    return (
        gaussian_KL(mu_1=mean, mu_2=X_obs, sigma_1=std_clamped, sigma_2=obs_noise_std) * M_obs
    ).sum()



# def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
#     return (
#         torch.log(sigma_2)
#         - torch.log(sigma_1)
#         + (torch.pow(sigma_1, 2) + torch.pow((mu_1 - mu_2), 2)) / (2 * sigma_2 ** 2)
#         - 0.5
#     )


def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    epsilon = 1e-6
    max_sigma_value = 1e6
    sigma_1_clamped = torch.clamp(sigma_1, min=epsilon, max=max_sigma_value)
    sigma_2_clamped = torch.clamp(sigma_2, min=epsilon, max=max_sigma_value)

    log_sigma_1 = torch.log(sigma_1_clamped)
    log_sigma_2 = torch.log(sigma_2_clamped)

    term1 = log_sigma_2 - log_sigma_1
    term2 = (torch.pow(sigma_1_clamped, 2) + torch.pow((mu_1 - mu_2), 2)) / (2 * sigma_2_clamped ** 2)
    return term1 + term2 - 0.5
