import sys
import os
import argparse
import tqdm
import torch
import numpy as np
import pandas as pd
from BGOB import model as ODE_model
from BGOB import data_utils

# Function to setup argument parser
def setup_arg_parser():
    parser = argparse.ArgumentParser(description="BGOB")
    parser.add_argument('--model_name', type=str, help="Model to use", default="BGOB_TEST")
    parser.add_argument('--dataset', type=str, help="Dataset CSV file", default="demo.csv")
    parser.add_argument("--with_mask", action="store_true", help="Input file has both 'Value_' columns and 'Mask_' columns", default=False)
    parser.add_argument('--seed', type=int, help="Seed for data split generation", default=100)
    parser.add_argument('--solver', type=str, choices=["euler", "midpoint","dopri5"], default="euler")

    # Hyperparameters: hidden_size, p_hidden, prep_hidden, mixing, delta_t, epoch_max, learning_rate
    parser.add_argument("--hidden_size", type=int, default=300)
    parser.add_argument("--p_hidden", type=int, default=600)
    parser.add_argument("--prep_hidden", type=int, default=500)
    parser.add_argument("--mixing", type=float, help="Mixing multiplier", default=0.0001)
    parser.add_argument("--delta_t", type=float, default=1)
    # torch.optim.Adam(model.parameters(), lr, weight_decay=0.0005)
    parser.add_argument("--epoch_max", type=int, default=600)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--max_T", type=float, default=None)
    parser.add_argument("--T_val", type=float, default=None)
    parser.add_argument("--whole_dataset", action="store_true", default=False)
    parser.add_argument("--normalization", action="store_true", default=False)
    parser.add_argument("--use_sigmoid", action="store_true", default=False)
    parser.add_argument("--zero_ratio", type=float, default=0.9)
    parser.add_argument("--out_file_dir", type=str, help="Output files path", default="./")
    return parser.parse_args()

def main():
    args = setup_arg_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_df = pd.read_csv(args.dataset, index_col=None)
    if input_df.columns[0] != 'ID' or input_df.columns[1] != 'Time':
        raise ValueError("First two columns of the input DataFrame should be 'ID' and 'Time'")
    if not args.with_mask: 
        input_df, Species_map_df = data_utils.generate_mask_file(input_df, args.out_file_dir)
        Species_map = dict(zip(Species_map_df['Value'], Species_map_df['Species']))

    raw_df= data_utils.preprocess_data(input_df, args.zero_ratio)
    zero_rate_dict = data_utils.compute_zero_rate(raw_df)
    raw_df.to_csv(args.out_file_dir + args.model_name + "_real_processed.csv", index=False)
    sel_df, scalar_list = data_utils.normalization(raw_df, log=False)
    species_idx = np.array([int(ss[6:]) for ss in sel_df.filter(like='Value_').columns])
    unique_ids = np.unique(sel_df["ID"]).astype(int)
    dl, dl_val, dl_val_whole, input_size, T = data_utils.create_dataloaders(sel_df, unique_ids, args)
    params_dict = {
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "p_hidden": args.p_hidden,
        "prep_hidden": args.prep_hidden,
        "mixing": args.mixing,
        "delta_t": args.delta_t,
        "solver": args.solver,
        "normalization": args.normalization,
        "use_sigmoid": args.use_sigmoid,
        "T": T,
        "zero_rate_dict": zero_rate_dict
    }
    
    model = ODE_model.bidirectional_gruodebayes(
        input_size=params_dict["input_size"],
        hidden_size=params_dict["hidden_size"],
        p_hidden=params_dict["p_hidden"],
        prep_hidden=params_dict["prep_hidden"],
        device=device,
        use_sigmoid=params_dict["use_sigmoid"],
        mixing=params_dict["mixing"],
        solver=params_dict["solver"],
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0005)
    epoch_max = args.epoch_max
    trained_model_path = (args.out_file_dir + args.model_name + "_final_state_dict.npy")
    mse_min = float("inf")
    best_epoch = 0
    
    # Training
    for epoch in range(1, epoch_max + 1):
        model.train()
        optimizer.zero_grad()

        for i, b in tqdm.tqdm(enumerate(dl)):
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(device)
            M = b["M"].to(device)
            obs_idx = b["obs_idx"]

            h, loss, path_t, path_p, path_h = model(
                times, time_ptr, X, M, obs_idx, delta_t=args.delta_t, return_path=True, T=T
            )

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            mse_val = 0
            loss_val = 0
            mse_val_whole = 0
            num_obs = 0
            rela_val = 0
            rela_val_whole = 0
            model.eval()
            for i, b in enumerate(dl_val):
                times = b["times"]
                time_ptr = b["time_ptr"]
                X = b["X"].to(device)
                M = b["M"].to(device)
                obs_idx = b["obs_idx"]

                X_val = b["X_val"].to(device)
                M_val = b["M_val"].to(device)
                times_val = b["times_val"]
                times_idx = b["index_val"]

                mean = (X_val * M_val).sum(0) / M_val.sum(0)

                hT, loss, t_vec, p_vec, h_vec = model(
                    times,
                    time_ptr,
                    X,
                    M,
                    obs_idx,
                    delta_t=args.delta_t,
                    T=T,
                    return_path=True,
                )

                p_values, p_variance = torch.chunk(p_vec, 2, dim=2)

                p_val = data_utils.extract_from_path(t_vec, p_vec, times_val, times_idx)
                m, v = torch.chunk(p_val, 2, dim=1)

                mse_loss = (torch.pow(X_val - m, 2) * M_val).sum()

                rela_loss = np.ma.masked_invalid(
                    (torch.abs(torch.div(X_val - m, mean)) * M_val).cpu().numpy()
                ).sum()

                mse_val += mse_loss.cpu().numpy()
                rela_val += rela_loss
                num_obs += M_val.sum().cpu().numpy()

            mse_val /= num_obs
            rela_val /= num_obs

            print(
                "Mean validation loss at epoch "
                + str(epoch)
                + ": mse="
                + str(mse_val)
                + ", relative error="
                + str(rela_val)
                + "  (num_obs="
                + str(num_obs)
                + ")"
            )

            for i, b in enumerate(dl_val_whole):
                times = b["times"]
                time_ptr = b["time_ptr"]
                X = b["X"].to(device)
                M = b["M"].to(device)
                obs_idx = b["obs_idx"]

                X_val_whole = b["X_val"].to(device)
                M_val = b["M_val"].to(device)
                times_val_whole = b["times_val"]
                times_idx_whole = b["index_val"]

                mean = (X_val_whole * M_val).sum(0) / M_val.sum(0)

                hT, loss, t_vec_whole, p_vec, h_vec = model(
                    times,
                    time_ptr,
                    X,
                    M,
                    obs_idx,
                    delta_t=args.delta_t,
                    T=T,
                    return_path=True,
                )

                p_values_whole, p_variance = torch.chunk(p_vec, 2, dim=2)

                p_val = data_utils.extract_from_path(
                    t_vec_whole, p_vec, times_val_whole, times_idx_whole
                )
                m, v = torch.chunk(p_val, 2, dim=1)

                mse_loss = (torch.pow(X_val_whole - m, 2) * M_val).sum()

                rela_loss = np.ma.masked_invalid(
                    (torch.abs(torch.div(X_val_whole - m, mean)) * M_val)
                    .cpu()
                    .detach()
                    .numpy()
                ).sum()

                mse_val_whole += mse_loss.cpu().detach().numpy()
                rela_val_whole += rela_loss
                num_obs += M_val.sum().cpu().detach().numpy()

            mse_val_whole /= num_obs
            rela_val_whole /= num_obs

            if mse_val < mse_min:
                mse_min = mse_val
                rela_min = rela_val
                mse_min_whole = mse_val_whole
                rela_min_whole = rela_val_whole
                torch.save(model.state_dict(), trained_model_path)
                results_whole = data_utils.inverse_normalization(
                    p_values_whole.cpu().numpy(), scalar_list, log=False
                )
                best_epoch = epoch

            print(
                "Mean validation loss at epoch "
                + str(epoch)
                + ": mse="
                + str(mse_val_whole)
                + ", relative error="
                + str(rela_val_whole)
                + "  (num_obs="
                + str(num_obs)
                + ")\tNOTE: This is the results of using the whole time series as input data."
            )

        if np.isnan(mse_val) or np.isnan(mse_val_whole):
            break

        if (epoch % 100 == 0) or (epoch == epoch_max):
            results_whole_df = data_utils.getDataFrame(results_whole.transpose(1,0,2), raw_df, species_idx, unique_ids)
            if not args.with_mask: 
                results_whole_df['species_name'] = results_whole_df['species'].map(Species_map)
            result_whole_file_csv = args.out_file_dir + args.model_name  + "_results_epoch" + str(epoch) + "_best_epoch" + str(best_epoch) + ".csv"
            results_whole_df.to_csv(result_whole_file_csv, index=False)

    zero_rate_list = []
    for key in zero_rate_dict:
        zero_rate_list.append(zero_rate_dict[key])
    truncated_results = data_utils.truncate(results_whole, zero_rate_list, 0.3)
    # truncated_result_file = (args.out_file_dir + args.model_name + ".npy")
    # np.save(truncated_result_file, truncated_results)
    truncated_results_df = data_utils.getDataFrame(truncated_results.transpose(1,0,2), raw_df, species_idx, unique_ids)
    if not args.with_mask: 
        truncated_results_df['species_name'] = truncated_results_df['species'].map(Species_map)
    truncated_result_file_csv = args.out_file_dir + args.model_name + "_best_epoch" + str(best_epoch) + ".csv"
    truncated_results_df.to_csv(truncated_result_file_csv, index=False)
    
    df_file_name = args.out_file_dir + args.model_name + "_model_parameter.csv"
    df_res = pd.DataFrame(
        {
            "Name": [args.model_name],
            "epoch": [epoch_max],
            "delta_t": [args.delta_t],
            "hidden_size": [params_dict["hidden_size"]],
            "p_hidden": [params_dict["p_hidden"]],
            "prep_hidden": [params_dict["prep_hidden"]],
            "mixing": [params_dict["mixing"]],
            "solver": [params_dict["solver"]],
            "MSE": [mse_min],
            "rela_loss": [rela_min],
            "MSE_whole": [mse_min_whole],
            "rela_loss_whole": [rela_min_whole],
            "whole_dataset": [args.whole_dataset],
        }
    )
    if os.path.isfile(df_file_name):
        df = pd.read_csv(df_file_name)
        df = pd.concat([df, df_res]).reset_index(drop=True)
        df.to_csv(df_file_name, index=False)
    else:
        df_res.to_csv(df_file_name, index=False)

if __name__ == '__main__':
    main()
