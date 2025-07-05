from sklearn.impute import KNNImputer
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer



class DataPreprocessor:
    def __init__ (self):
        # num features truoc khi prep for step 2
        self.numeric_features = ['adv-year', 'adv-month', 'reg-year', 'runned-miles', 'engine-size', 'engine-power', 'annual-tax', 'wheelbase', 'length', 'width', 'height', 'average-mpg', 'top-speed', 'seat-num', 'door-num']
        self.drop_feats = ['color', 'genmodel-id', 'adv-id', 'adv-year', 'adv-month', 'reg-year', 'annual-tax', 'wheelbase', 'height', 'door-num']
        self.feats_with_most_nan = []
        self.NaN_ratio_threshold = 0.5 # >= x thi loai bo
        self.schema = pd.DataFrame([
            ['age', 'Float64', 'none', 'mean'],
            ['runned-miles', 'Float64', 'none', 'mean'],
            ['engine-size', 'Float64', 'none', 'mean'],
            ['engine-power', 'Float64', 'none', 'median'],
            ['width', 'Float64', 'none', 'median'],
            ['length', 'Float64', 'none', 'median'],
            ['average-mpg', 'Float64', 'none', 'median'],
            ['top-speed', 'Float64', 'none', 'median'],
            ['seat-num', 'Float64', 'one-hot', 'mode'],
            ['maker', 'Float64', 'target', 'mean'], # encode 
            ['genmodel', 'Float64', 'target', 'mean'], # encode 
            # ['color', 'Float64', 'target', 'mode'],
            ['bodytype', 'Float64', 'one-hot', 'mode'],
            ['gearbox', 'Float64', 'one-hot', 'mode'],
            ['fuel-type', 'Float64', 'one-hot', 'mode'],
        ])

        self.categorical_choices = {}
        self.encoded_cols = []
        self.continuous_cols = ['age', 'runned-miles', 'engine-size', 'engine-power', 'width', 'length', 'average-mpg', 'top-speed']

    def change_column_names (self, dataframe): #step 1
        dataframe = dataframe.copy()
        dataframe.columns = [col.lower().replace('_', '-') for col in dataframe.columns]
        return dataframe
    
    def perform_light_cleaning (self, dataframe): # step 2: loại bỏ đơn vị và chuyển đổi kiểu dữ liệu
        dataframe = dataframe.copy()
        for col in self.numeric_features:
            if col in dataframe.columns:
                dataframe[col] = dataframe[col].astype(str).str.extract(r'(\d+(?:\.\d+)?)')[0]
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        return dataframe
    
    def add_feature_age (self, dataframe): # step 3: thêm tuổi xe
        dataframe = dataframe.copy()
        if 'adv-year' in dataframe.columns and 'reg-year' in dataframe.columns:
            dataframe['age'] = dataframe['reg-year'] - dataframe['adv-year']
        return dataframe
    
    def drop_features (self, dataframe): # step 4: loại bỏ các cột không cần thiết
        dataframe = dataframe.copy()
        return dataframe.drop(columns=self.drop_feats, errors='ignore')
    
    def drop_rows_with_nan_price (self, X, y): # step 5: loại bỏ các hàng có giá trị NaN trong cột 'price'
        X = X.copy()
        y = y.copy()
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        return X, y
    
    def drop_rows_with_nan (self, dataframe, y): # step 6: drop mẫu nếu NaN ở cột có nhiều nan nhất
        dataframe = dataframe.copy()
        for col in self.feats_with_most_nan:
            if col in dataframe.columns:
                dataframe = dataframe[dataframe[col].notna()]
                y = y[dataframe.index]
        return dataframe, y
    
    def drop_rows_with_nan_ratio (self, X, y): # step 6.5: loai bo cac hang co NaN vuot qua nguong cho phep
        X = X.copy()
        y = y.copy()
        nan_ratio = X.isna().sum(axis=1) / X.shape[1]
        mask = nan_ratio < self.NaN_ratio_threshold
        X = X[mask]
        y = y[mask]
        return X, y

    def fit_for_schema (self, x_train, y_train): # step 7: fit cho schema
        # fit cho dac trung lien tuc
        self.continuous_stats = {}
        for feature in self.schema.values:
            if feature[2] == 'none' and feature[0] in x_train.columns:
                feature_name = feature[0]
                col = x_train[feature_name]
                self.continuous_stats[feature_name] = {
                    'mode': col.value_counts().index[:3].tolist(),
                    'mean': col.mean(),
                    'median': col.median()
                }

        # fit tren label
        self.target_stats = {
            'mean': y_train.mean(),
            'median': y_train.median(),
        }

        # fit cho dac trung phan loai, target encoding, one-hot encoding
        self.categorical_stats = {}
        self.target_encoders = {}
        self.one_hot_encoders = {}
        for feature in self.schema.values:
            if feature[2] != 'none' and feature[0] in x_train.columns:
                feature_name = feature[0]
                col = x_train[feature_name]
                self.categorical_stats[feature_name] = {
                    'mode': col.value_counts().index[:3].tolist(),
                }

                # Lưu các giá trị phân loại khác nhau + 'Other'
                
                unique_values = sorted(col.dropna().unique().tolist())
                self.categorical_choices[feature_name] = unique_values + ['Other']

            
            if feature[2] == 'target':
                encoder = TargetEncoder(cols=[feature_name], handle_unknown='value', smoothing=0.1)
                encoder.fit(x_train[feature_name], y_train)
                self.target_encoders[feature_name] = encoder

            elif feature[2] == 'one-hot':
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoder.fit(x_train[[feature_name]])
                self.one_hot_encoders[feature_name] = encoder


        # PHAN LOAI GENMODEL THEO MAKER
        self.categorical_choices['genmodel'] = {}
        for maker in self.categorical_choices['maker']:
            models = x_train[x_train['maker'] == maker]['genmodel'].unique()
            self.categorical_choices['genmodel'][maker] = models.tolist() + ['Other']

        # KNN IMPUTER
        knn_imputer = KNNImputer(n_neighbors=5)
        self.imputer = knn_imputer.fit(x_train[self.continuous_cols])

    def add_nan_flags(self, dataframe): # step 7.5: thêm cột đánh dấu NaN
        dataframe = dataframe.copy()
        for feature in self.schema.values:
            feature_name = feature[0]
            if feature_name in dataframe.columns:
                dataframe[f'{feature_name}_is_nan'] = dataframe[feature_name].isna().astype(int)

        return dataframe

    def impute (self, dataframe): # step 8: thay thế giá trị NaN
        dataframe = dataframe.copy()
        for feature in self.schema.values:
            feature_name = feature[0]
            feature_type = feature[1]
            feature_encode_method = feature[2]
            feature_impute_method = feature[3]
            
            # KNN IMPUTER
            dataframe[self.continuous_cols] = self.imputer.transform(dataframe[self.continuous_cols])

            if feature_name in dataframe.columns and feature_name in self.continuous_stats:
                dataframe[feature_name] = dataframe[feature_name].astype(feature_type)

                # if feature_impute_method == 'mean':
                #     dataframe[feature_name] = dataframe[feature_name].fillna(self.continuous_stats[feature_name]['mean'])
                # elif feature_impute_method == 'median':
                #     dataframe[feature_name] = dataframe[feature_name].fillna(self.continuous_stats[feature_name]['median'])
                # elif feature_impute_method == 'mode':
                #     modes = self.continuous_stats[feature_name]['mode']
                #     dataframe[feature_name] = dataframe[feature_name].fillna(modes[0] if pd.notna(modes[0]) else modes[1])

            elif feature_name in dataframe.columns and feature_name in self.categorical_stats:
                if feature_impute_method == 'mode':
                    modes = self.categorical_stats[feature_name]['mode']
                    dataframe[feature_name] = dataframe[feature_name].fillna(modes[0] if pd.notna(modes[0]) else modes[1])
                elif feature_encode_method == 'target' and feature_impute_method == 'mean':
                    dataframe[feature_name] = dataframe[feature_name].fillna(self.target_stats['mean'])

        return dataframe
                
    def encode (self, dataframe): # step 9: encode
        dataframe = dataframe.copy()

        for feature in self.schema.values:
            feature_name = feature[0]
            feature_type = feature[1]
            feature_encode_method = feature[2]

            if feature_name in dataframe.columns:
                if feature_encode_method == 'target':
                    if feature_name in self.target_encoders:
                        encoder = self.target_encoders[feature_name]
                        dataframe[feature_name] = encoder.transform(dataframe[feature_name])[feature_name]
                
                elif feature_encode_method == 'one-hot':
                    if feature_name in self.one_hot_encoders:
                        encoder = self.one_hot_encoders[feature_name]
                        encoded_data = encoder.transform(dataframe[[feature_name]])
                        new_col_names = encoder.get_feature_names_out([feature_name])
                        new_col_names = [col.replace(' ', '_').lower() for col in new_col_names]  # Replace spaces with underscores

                        encoded_df_part = pd.DataFrame(encoded_data, columns=new_col_names, index= dataframe.index)
                        df_tmp = dataframe.drop(columns=[feature_name])
                        dataframe = pd.concat([df_tmp, encoded_df_part], axis=1)

        return dataframe

    def fit_for_scaler (self, x_train): # step 10: fit cho scaler
        self.scaler = RobustScaler()
        self.scaler.fit(x_train)

    def scale (self, dataframe): # step 11: scale
        dataframe = dataframe.copy()
        if hasattr(self, 'scaler'):
            return pd.DataFrame(self.scaler.transform(dataframe), columns=dataframe.columns, index=dataframe.index)
        else:
            raise ValueError("co gi do sai sai")

    def fit_transform_pipeline (self, x_train, y_train):
        x_train = self.change_column_names(x_train)
        x_train = self.perform_light_cleaning(x_train)
        x_train = self.add_feature_age(x_train)
        x_train = self.drop_features(x_train)
        x_train, y_train = self.drop_rows_with_nan_price(x_train, y_train)
        x_train, y_train = self.drop_rows_with_nan(x_train, y_train)
        x_train, y_train = self.drop_rows_with_nan_ratio(x_train, y_train)

        self.fit_for_schema(x_train, y_train)

        # x_train = self.add_nan_flags(x_train)
        x_train = self.impute(x_train)
        x_train = self.encode(x_train)
        self.encoded_cols = x_train.columns

        self.fit_for_scaler(x_train)

        x_train = self.scale(x_train)

        return x_train, y_train

    def transform_pipeline (self, X, y, drop_nan = True):
        X = self.change_column_names(X)
        X = self.perform_light_cleaning(X)
        X = self.add_feature_age(X)
        X = self.drop_features(X)

        X, y = self.drop_rows_with_nan_price(X, y)
        
        if drop_nan == True: 
            # X, y = self.drop_rows_with_nan(X, y)
            X, y = self.drop_rows_with_nan_ratio(X, y)

        # X = self.add_nan_flags(X)
        X = self.impute(X)
        X = self.encode(X)

        X = self.scale(X)
        

        return X, y
    
    def get_categorical_choices (self) :
        if  hasattr(self, 'categorical_choices'):
            return self.categorical_choices
        else: 
            return {}
    
    def get_encoded_cols (self):
        if hasattr(self, 'encoded_cols'):
            return self.encoded_cols
        else : 
            return []