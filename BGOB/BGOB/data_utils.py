import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class ODE_Dataset(Dataset):
    def __init__(
        self, df, val_options={}, idx=None, validation=False, whole_seq_validation=False
    ):
        self.validation = validation
        self.whole_seq_validation = whole_seq_validation
        self.df = df.drop_duplicates(["ID", "Time"])
        if val_options.get("max_T"):
            self.df = self.df.loc[self.df["Time"] <= val_options["max_T"], :]
        self.num_unique = self.df["ID"].nunique()

        if self.validation:
            if self.whole_seq_validation == False:
                df_beforeIdx = self.df.loc[
                    self.df["Time"] <= val_options["T_val"], "ID"
                ].unique()
                df_afterIdx = self.df.loc[
                    self.df["Time"] > val_options["T_val"], "ID"
                ].unique()

                valid_idx = np.intersect1d(df_beforeIdx, df_afterIdx)
                self.df = self.df.loc[self.df["ID"].isin(valid_idx)]
        if idx is not None:
            self.df = self.df.loc[self.df["ID"].isin(idx)].copy()
            map_dict = dict(
                zip(self.df["ID"].unique(), np.arange(self.df["ID"].nunique()))
            )
            self.df["ID"] = self.df["ID"].map(map_dict)

        self.variable_num = sum([c.startswith("Value") for c in self.df.columns])

        self.df = self.df.astype(np.float32)

        if self.validation:
            if self.whole_seq_validation == False:
                self.df_before = self.df.loc[
                    self.df["Time"] <= val_options["T_val"]
                ].copy()
                self.df_after = (
                    self.df.loc[self.df["Time"] > val_options["T_val"]]
                    .sort_values("Time")
                    .copy()
                )

                self.df = self.df_before.copy()

                self.df_after.ID = self.df_after.ID.astype(int)
                self.df_after.sort_values("Time", inplace=True)
            else:
                self.df_after = self.df.copy()
                self.df_after.ID = self.df_after.ID.astype(int)
                self.df_after.sort_values("Time", inplace=True)
        else:
            self.df_after = None

        self.length = self.df["ID"].nunique()
        self.df.ID = self.df.ID.astype(int)
        self.df.set_index("ID", inplace=True)

        self.df.sort_values("Time", inplace=True)

    def max_time(self):
        return max(self.df["Time"])

    def variable_num(self):
        return self.variable_num

    def num_unique(self):
        return self.num_unique

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        subset = self.df.loc[index]

        if len(subset.shape) == 1:
            subset = self.df.loc[[index]]

        if self.validation:
            val_samples = self.df_after.loc[self.df_after["ID"] == index]

        else:
            val_samples = None

        return {"idx": index, "path": subset, "val_samples": val_samples}


class ODE_collate_fn(Dataset):
    def __init__(self, normalization=True):
        self.normalization = normalization

    def __call__(self, batch):
        idx2batch = pd.Series(np.arange(len(batch)), index=[b["idx"] for b in batch])

        pat_idx = [b["idx"] for b in batch]

        df = pd.concat([b["path"] for b in batch], axis=0)

        df.sort_values(by=["Time"], inplace=True)

        batch_ids = idx2batch[df.index.values].values

        times, counts = np.unique(df.Time.values, return_counts=True)

        time_ptr = np.concatenate([[0], np.cumsum(counts)])

        value_cols = [c.startswith("Value") for c in df.columns]
        mask_cols = [c.startswith("Mask") for c in df.columns]

        if batch[0]["val_samples"] is not None:
            df_after = pd.concat(b["val_samples"] for b in batch)
            df_after.sort_values(by=["ID", "Time"], inplace=True)

            value_cols_val = [c.startswith("Value") for c in df_after.columns]
            mask_cols_val = [c.startswith("Mask") for c in df_after.columns]

            X_val = torch.tensor(df_after.iloc[:, value_cols_val].values)
            M_val = torch.tensor(df_after.iloc[:, mask_cols_val].values)

            times_val = df_after["Time"].values
            index_val = idx2batch[df_after["ID"].values].values
        else:
            X_val = None
            M_val = None
            times_val = None
            index_val = None

        out = {}
        # real_idx (reindexed in the ODE_Dataset __init__)
        out["pat_idx"] = pat_idx
        out["times"] = times
        out["time_ptr"] = time_ptr

        out["X"] = torch.tensor(df.iloc[:, value_cols].values)
        out["M"] = torch.tensor(df.iloc[:, mask_cols].values)
        out["obs_idx"] = torch.tensor(batch_ids)

        out["X_val"] = X_val
        out["M_val"] = M_val
        out["times_val"] = times_val
        out["index_val"] = index_val

        return out


class ODE_collate_fn_classification(Dataset):
    def __init__(self, normalization=True):
        self.normalization = normalization

    def __call__(self, batch):
        idx2batch = pd.Series(np.arange(len(batch)), index=[b["idx"] for b in batch])

        pat_idx = [b["idx"] for b in batch]

        df = pd.concat([b["path"] for b in batch], axis=0)

        df.sort_values(by=["Time"], inplace=True)

        batch_ids = idx2batch[df.index.values].values

        times, counts = np.unique(df.Time.values, return_counts=True)

        time_ptr = np.concatenate([[0], np.cumsum(counts)])

        value_cols = [c.startswith("Value") for c in df.columns]
        mask_cols = [c.startswith("Mask") for c in df.columns]

        y = torch.tensor([b["y"] for b in batch])

        out = {}
        # real_idx (reindexed in the ODE_Dataset __init__)
        out["pat_idx"] = pat_idx
        out["times"] = times
        out["time_ptr"] = time_ptr

        out["X"] = torch.tensor(df.iloc[:, value_cols].values)
        out["M"] = torch.tensor(df.iloc[:, mask_cols].values)
        out["obs_idx"] = torch.tensor(batch_ids)

        out["y"] = y

        return out


def extract_from_path(t_vec, p_vec, eval_times, path_idx_eval):

    t_vec, unique_index = np.unique(t_vec, return_index=True)
    p_vec = p_vec[unique_index, :, :]

    present_mask = np.isin(eval_times, t_vec)
    eval_times[~present_mask] = map_to_closest(eval_times[~present_mask], t_vec)

    mapping = dict(zip(t_vec, np.arange(t_vec.shape[0])))

    time_idx = np.vectorize(mapping.get)(eval_times)

    return p_vec[time_idx, path_idx_eval, :]


def map_to_closest(input, reference):
    output = np.zeros_like(input)
    for idx, element in enumerate(input):
        closest_idx = (np.abs(reference - element)).argmin()
        output[idx] = reference[closest_idx]
    return output


def normalization(df, log=False):
    scaler_list = []
    new_df = df.copy()
    for index, col in df.items():
        scaler = MinMaxScaler()
        if index.startswith("Value"):
            data = np.array(col).reshape(-1, 1)
            if log:
                data = np.log(data + 1)
            scaler.fit(data)
            scaler_list.append(scaler)
            new_data = scaler.transform(data)
            new_df[index] = pd.Series(new_data.reshape(-1))

    return new_df, scaler_list


def standard_normalization(df):
    scaler_list = []
    new_df = df.copy()
    for index, col in df.items():
        scaler = StandardScaler()
        if index.startswith("Value"):
            data = np.array(col).reshape(-1, 1)
            scaler.fit(data)
            scaler_list.append(scaler)
            new_data = scaler.transform(data)
            new_df[index] = pd.Series(new_data.reshape(-1))

    return new_df, scaler_list


def inverse_normalization(data, scaler_list, log=False):
    if len(data.shape) == 3:
        new_data = data.reshape(-1, data.shape[2])
        df = pd.DataFrame(new_data)
        new_df = df.copy()
        for i, scaler in enumerate(scaler_list):
            col_name = i
            new_df[col_name] = pd.Series(
                scaler.inverse_transform(np.array(df[col_name]).reshape(-1, 1)).reshape(
                    -1
                )
            )

        out = np.array(new_df).reshape(data.shape[0], data.shape[1], -1)
    else:
        new_data = data.reshape(-1, data.shape[1])
        df = pd.DataFrame(new_data)
        new_df = df.copy()
        for i, scaler in enumerate(scaler_list):
            col_name = i
            new_df[col_name] = pd.Series(
                scaler.inverse_transform(np.array(df[col_name]).reshape(-1, 1)).reshape(
                    -1
                )
            )

        out = np.array(new_df).reshape(data.shape[0], -1)
    if log:
        out = np.exp(out) - 1
    return out


def omit_zero(df, min_value, ratio):
    valid_cols = [
        col for col, series in df.items() 
        if col.startswith("Value_") and 
        (series < min_value).mean() < ratio
    ]

    value_cols = ["Value_" + col.replace("Value_", "") for col in valid_cols]
    mask_cols = ["Mask_" + col.replace("Value_", "") for col in valid_cols]
    selected_cols = ['ID', 'Time'] + value_cols + mask_cols

    return df[selected_cols].copy()


def mask_extreme_value(df, n):
    value_cols = [col for col in df.columns if col.startswith("Value_")]
    mask_cols = [col for col in df.columns if col.startswith("Mask_")]

    for value_col in value_cols:
        value_data = df[value_col].values
        value_omit_zero = value_data[value_data != 0]
        mean, std = np.mean(value_omit_zero), np.std(value_omit_zero)
        max_value = mean + n * std
        
        mask = value_data < max_value
        df[value_col] = df[value_col] * mask
        corresponding_mask_col = mask_cols[value_cols.index(value_col)]
        df[corresponding_mask_col] = df[corresponding_mask_col] * mask

    return df


def truncate(data, zero_rate_list, t):
    final_data = []
    for i in range(data.shape[2]):
        temp_data = data[:, :, i]
        threshold = np.quantile(temp_data, zero_rate_list[i] * t)
        # any data smaller than threshold should be treated as 0
        temp_data_with_real_zero = temp_data * (temp_data > threshold)
        final_data.append(temp_data_with_real_zero)

    target = np.stack(final_data, axis=2)
    return target


def generate_mask_file(df, out_file_dir):
    sorted_df = df.sort_values(by=['ID', 'Time'])
    Value_df = sorted_df.iloc[:, 2:]
    Mask_df = (Value_df != 0) * 1
    Species_map_df = pd.DataFrame(data={'Value': range(1, sorted_df.shape[1] - 1), 'Species': sorted_df.columns[2:]})
    Species_map_df.to_csv(out_file_dir + "value_species_map.csv", index=None)
    all_data_df = pd.concat([sorted_df, Mask_df], axis=1)
    column_names = (
        ['ID', 'Time']
        + ['Value_{}'.format(i) for i in range(1, sorted_df.shape[1] - 1)]
        + ['Mask_{}'.format(i) for i in range(1, sorted_df.shape[1] - 1)]
    )
    all_data_df.columns = column_names
    return all_data_df, Species_map_df

def row_function(row, real_value_df):
    temp_row = real_value_df[(real_value_df["ID"]==int(row["subject"])) & (real_value_df["Time"]==int(row["time"]))]
    if temp_row.shape[0] > 0:
        temp_value = "Value_" + str(row["species"]) 
        return temp_row[temp_value].values[0]
    else:
        return np.nan

def getDataFrame(pre_values, input_values, species_idx, unique_ids):
    column_name = ["Predict_Value_" + str(i) for i in species_idx]
    predict_df = pd.DataFrame(columns=column_name)
    for i in range(pre_values.shape[0]):
        temp_df = pd.DataFrame(pre_values[i])
        temp_df.columns = column_name
        predict_df = pd.concat([predict_df, temp_df], axis=0)
    row_column_name = [
        "data",
        "time",
        "real_data",
        "subject",
        "species",
    ]
    df = pd.DataFrame(columns=row_column_name)
    for i in range(pre_values.shape[0]):
        temp = pre_values[i]
        data = []
        times = []
        species = []
        subject = [unique_ids[i] for _ in range(temp.shape[0]*temp.shape[1])]
        for n in range(temp.shape[0]):
            data = data + temp[n].tolist()
            times = times + [str(n) for _ in range(temp.shape[1])]
            species = species + species_idx.tolist()
        df_tmp = pd.DataFrame({'data': data, 'time': times,
                            'subject': subject, 'species': species})
        df = pd.concat([df, df_tmp], axis=0)
    df.index = pd.Series(list(range(df.shape[0])))
    df['real_data'] = df.apply(row_function, axis=1, args=(input_values,))
    return df


def preprocess_data(df, zero_ratio):
    df = omit_zero(df, 0.00001, zero_ratio).copy()
    df = mask_extreme_value(df, 3).copy()
    return df

def compute_zero_rate(df):
    value_cols = df.filter(like='Value_').columns
    zero_rate_dict = df[value_cols].apply(lambda col: np.mean(col == 0)).to_dict()
    return zero_rate_dict

def create_dataloaders(df, unique_ids, args):
    train_idx, val_idx = train_test_split(unique_ids, test_size=0.2, random_state=args.seed)
    collate_fn = ODE_collate_fn(normalization=args.normalization)
    data_train = ODE_Dataset(df=df, idx=(unique_ids if args.whole_dataset else train_idx))
    data_val_whole = ODE_Dataset(
        df=df, idx=unique_ids, validation=True, whole_seq_validation=True
    )
    T = max(data_train.max_time(), data_val_whole.max_time()) if args.max_T is None else args.max_T
    val_options = {"T_val": (3 * T / 4 if args.T_val is None else args.T_val), "max_T": T}
    data_val = ODE_Dataset(
        df=df, idx=val_idx, validation=True, val_options=val_options
    )

    dl = DataLoader(dataset=data_train, collate_fn=collate_fn, shuffle=False, batch_size=10, num_workers=4)
    dl_val = DataLoader(dataset=data_val, collate_fn=collate_fn, shuffle=False, batch_size=len(data_val), num_workers=1)
    dl_val_whole = DataLoader(dataset=data_val_whole, collate_fn=collate_fn, shuffle=False, batch_size=len(data_val_whole), num_workers=1)

    return dl, dl_val, dl_val_whole, data_train.variable_num, T
