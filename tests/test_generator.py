from mvaelib import generator
import numpy as np
import pandas as pd


def test_generate_profiles_social_media_from_noise():
    gen = generator.ProfilesGenerator()

    noise = np.random.normal(0, 1, size=(100, 8))
    _vk = gen.generate_profiles_social_media(noise=noise)
    assert isinstance(_vk, pd.DataFrame) and _vk.shape[0] == 100


def test_generate_profiles_social_media_from_interests():
    gen = generator.ProfilesGenerator()

    interests = np.random.uniform(0, 1, size=(100, 22))
    interests /= interests.sum()
    _vk = gen.generate_profiles_social_media(interests=interests)
    assert isinstance(_vk, pd.DataFrame) and _vk.shape[0] == 100


def test_generate_profiles_social_media_from_transactions():
    gen = generator.ProfilesGenerator()

    transactions = np.random.uniform(0, 1, size=(100, 17))
    _vk = gen.generate_profiles_social_media(transactions=transactions)
    assert isinstance(_vk, pd.DataFrame) and _vk.shape[0] == 100


def test_generate_profiles_interests_from_noise():
    gen = generator.ProfilesGenerator()

    noise = np.random.normal(0, 1, size=(100, 8))
    _in = gen.generate_profiles_interests(noise=noise)
    assert isinstance(_in, pd.DataFrame) and _in.shape[0] == 100


def test_generate_profiles_interests_from_social_media():
    gen = generator.ProfilesGenerator()

    social_media = np.random.uniform(0, 1, size=(100, 26))
    _in = gen.generate_profiles_interests(social_media=social_media)
    assert isinstance(_in, pd.DataFrame) and _in.shape[0] == 100


def test_generate_profiles_interests_from_transactions():
    gen = generator.ProfilesGenerator()

    transactions = np.random.uniform(0, 1, size=(100, 17))
    _in = gen.generate_profiles_interests(transactions=transactions)
    assert isinstance(_in, pd.DataFrame) and _in.shape[0] == 100


def test_generate_profiles_transactions_from_noise():
    gen = generator.ProfilesGenerator()

    noise = np.random.normal(0, 1, size=(100, 8))
    _tr = gen.generate_profiles_transactions(noise=noise)
    assert isinstance(_tr, pd.DataFrame) and _tr.shape[0] == 100


def test_generate_profiles_transactions_from_social_media():
    gen = generator.ProfilesGenerator()

    social_media = np.random.uniform(0, 1, size=(100, 26))
    _tr = gen.generate_profiles_transactions(social_media=social_media)
    assert isinstance(_tr, pd.DataFrame) and _tr.shape[0] == 100


def test_generate_profiles_transactions_from_interests():
    gen = generator.ProfilesGenerator()

    interests = np.random.uniform(0, 1, size=(100, 22))
    interests /= interests.sum()
    _tr = gen.generate_profiles_transactions(interests=interests)
    assert isinstance(_tr, pd.DataFrame) and _tr.shape[0] == 100

