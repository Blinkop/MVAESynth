from .mvae_utils import transform_vk, transform_in, transform_tr

import torch
import pandas as pd
import joblib

class ProfilesGenerator(object):
    def __init__(self, use_memvae=False, filename=None, device='cpu'):
        if not filename and not use_memvae:
            filename = 'mvae3_best2.model'
        elif not filename and use_memvae:
            filename = 'memvae3_best.model'

        device = torch.device(device)

        self.use_memvae = use_memvae
        self._model = torch.load(filename)
        self._model.to(device)
        self._model.eval()

        self._vk_scaler = joblib.load('vk_scaler.jl')
        self._in_scaler = joblib.load('in_scaler.jl')
        self._tr_scaler = joblib.load('tr_scaler.jl')

        self._vk_bin_cols = ['sex', 'has_high_education', 'age_hidden',
                             'mobile_phone', 'twitter', 'facebook',
                             'instagram', 'movies', 'music', 'quotes']
        self._tr_bin_cols = ['is_gamer', 'is_parent', 'is_driver',
                             'has_pets', 'cash_usage']

        self._vk_cols = ['sex', 'age', 'has_high_education',
                         'relation', 'num_of_relatives', 'followers_count',
                         'status', 'mobile_phone', 'twitter',
                         'facebook', 'instagram', 'about',
                         'about_topic', 'activities', 'activities_topic',
                         'books', 'interests', 'interests_topic',
                         'movies', 'music', 'quotes',
                         'personal_alcohol', 'personal_life_main', 'personal_people_main',
                         'personal_political', 'age_hidden']
        self._in_cols = ['sbj_0_0', 'sbj_0_1', 'sbj_0_2',
                         'sbj_0_3', 'sbj_0_4', 'sbj_0_5',
                         'sbj_0_6', 'sbj_0_7', 'sbj_0_8',
                         'sbj_0_11', 'sbj_0_12', 'sbj_0_14',
                         'sbj_0_15', 'sbj_0_16', 'sbj_0_21',
                         'sbj_0_22', 'sbj_0_24', 'sbj_0_25',
                         'sbj_0_26', 'sbj_0_28', 'sbj_0_29',
                         'sbj_0_30']
        self._tr_cols = ['max_tr', 'min_tr', 'mean_tr',
                         'median_tr', '90_perc', 'sum_am',
                         'tr_per_month', 'cash_sum', 'cash_usage',
                         'game_sum', 'is_gamer', 'parent_sum',
                         'is_parent', 'driver_sum', 'is_driver',
                         'pets_sum', 'has_pets']

    def get_encoded_size(self):
        return 8

    def generate_profiles_social_media(self, noise=None, interests=None, transactions=None):
        return self._generate_profiles(noise=noise, interests=interests, transactions=transactions)[0]

    def generate_profiles_interests(self, noise=None, social_media=None, transactions=None):
        return self._generate_profiles(noise=noise, social_media=social_media, transactions=transactions)[1]

    def generate_profiles_transactions(self, noise=None, interests=None, social_media=None):
        return self._generate_profiles(noise=noise, interests=interests, social_media=social_media)[2]

    @torch.no_grad()
    def _generate_profiles(self, noise=None, social_media=None, interests=None, transactions=None):
        if noise is None and (interests is None and social_media is None and transactions is None):
            raise Exception('Neither noise or profiles were provided.')

        if noise is not None and (interests is not None or social_media is not None or transactions is not None):
            raise Exception('Both noise and profiles were provided')

        if noise is not None:
            noise = torch.tensor(noise).float().to(next(self._model.parameters()).device)
            if self.use_memvae:
                z0 = torch.normal(0, 1, size=noise.shape)
                decoded = self._model.decode(noise, z0)
            else:
                z0 = torch.normal(0, 1, size=noise.shape)
                decoded = self._model.decode(noise)

            for i in range(len(decoded)):
                decoded[i] = decoded[i].cpu().numpy()

            return self._transform_social_media(decoded[0]),\
                   self._transform_interests(decoded[1]),\
                   self._transform_transactions(decoded[2])

        _vk, _in, _tr = None, None, None
        if social_media is not None:
            _vk = self._standardize_social_media(social_media)
            _vk = torch.tensor(_vk).float().to(next(self._model.parameters()).device)
        if interests is not None:
            _in = self._standardize_interests(interests)
            _in = torch.tensor(_in).float().to(next(self._model.parameters()).device)
        if transactions is not None:
            _tr = self._standardize_transactions(transactions)
            _tr = torch.tensor(_tr).float().to(next(self._model.parameters()).device)

        if self.use_memvae:
            vk_recon, in_recon, tr_recon, _, _, _, _, _, _, _, _ = self._model(_vk, _in, _tr)
        else:
            vk_recon, in_recon, tr_recon, _, _ = self._model(_vk, _in, _tr)

        vk_recon = vk_recon.cpu().numpy()
        in_recon = in_recon.cpu().numpy()
        tr_recon = tr_recon.cpu().numpy()

        result = (self._transform_social_media(vk_recon) if social_media is None else social_media,
                  self._transform_interests(in_recon) if interests is None else interests,
                  self._transform_transactions(tr_recon) if transactions is None else transactions)

        return result


    def _standardize_social_media(self, data):
        return self._vk_scaler.transform(pd.DataFrame(data=data, columns=self._vk_cols))

    def _standardize_interests(self, data):
        return self._in_scaler.transform(pd.DataFrame(data=data, columns=self._in_cols))

    def _standardize_transactions(self, data):
        return self._tr_scaler.transform(pd.DataFrame(data=data, columns=self._tr_cols))

    def _transform_social_media(self, data):
        data = self._vk_scaler.inverse_transform(data)
        data = pd.DataFrame(data=data, columns=self._vk_cols)
        return transform_vk(data)

    def _transform_interests(self, data):
        data = self._in_scaler.inverse_transform(data)
        data = pd.DataFrame(data=data, columns=self._in_cols)
        return transform_in(data)

    def _transform_transactions(self, data):
        data = self._tr_scaler.inverse_transform(data)
        data = pd.DataFrame(data=data, columns=self._tr_cols)
        return transform_tr(data)
