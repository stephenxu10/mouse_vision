# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:04:15 2020

@author: Zhe
"""

import os, pickle
import numpy as np
import datajoint as dj

from . import utils

dj.config['database.host'] = 'at-database.ad.bcm.edu'

from neuro_data.static_images import data_schemas as data
from neuro_data.static_images import configs
dj.config['external-data']['location'] = '/external'

def fetch_scan(data_dir, scan):
    r"""Fetches scan data from at-database.

    Input images and responses are fetched for each scan. Anatomy data is also
    fetched if available.

    Args
    ----
    data_dir: str
        The path of mouse data directory. Fetched raw data will be saved in
        ``[data_dir]/fetched``.
    scan: str
        The scan name in the form of ``xxxxx-yy-zz``, in which ``xxxxx`` is
        animal ID, ``yy`` is session index and ``zz`` is scan index.

    """
    fetch_dir = os.path.join(data_dir, 'fetched')
    if not os.path.exists(fetch_dir):
        print('folder {} does not exist, will be created'.format(fetch_dir))
        os.makedirs(fetch_dir)

    animal_id, session, scan_idx = [int(x) for x in scan.split('-')]
    key = {'animal_id': animal_id, 'session': session, 'scan_idx': scan_idx, 'preproc_id': 0}
    print('\nfetching raw data via datajoint with\n{}'.format(key))

    # fetch raw response data
    filename = os.path.join(fetch_dir, 'response.data_{}.pickle'.format(scan))
    if os.path.exists(filename):
        print('response data already exists')
    else:
        print('response data does not exist, fetching from server...')
        group_id = np.random.choice((data.StaticMultiDataset.Member & key).fetch('group_id'))
        data_hash = (configs.DataConfig.MultipleAreasOneLayer()& \
                     {'stimulus_type': 'stimulus.Frame',
                      'exclude': '', 'layer': 'L2/3',
                      'brain_areas': 'all-unknown',
                      'normalize_per_image': False}).fetch1('data_hash')

        datasets, _ = configs.DataConfig().load_data({'group_id': group_id, 'data_hash': data_hash})
        for d_key in datasets.keys():
            if '{}-{}-{}-0'.format(animal_id, session, scan_idx) in d_key:
                break
        dset = datasets[d_key]

        responses, images, behaviors, pupil_centers = dset.responses[:, dset.transforms[1].idx], dset.images, dset.behavior, dset.pupil_center
        areas, layers, unit_ids, trial_idxs = dset.neurons.area, dset.neurons.layer, dset.neurons.unit_ids, dset.trial_idx
        stimulus = dj.create_virtual_module('pipeline_stimulus', 'pipeline_stimulus')
        stimulus_info = {}
        stimulus_info['imagenet_ids'], stimulus_info['descriptions'], stimulus_info['trial_idxs'] = \
            (stimulus.Frame*stimulus.StaticImage.ImageNet*data.InputResponse.Input&key).fetch('imagenet_id', 'description', 'trial_idx', order_by='row_id')

        print('responses shape {}'.format(responses.shape))
        print('images shape {}'.format(images.shape))
        print('areas shape {}'.format(areas.shape))

        with open(filename, 'wb') as f:
            pickle.dump({
                'key': key,
                'responses': responses,
                'images': images,
                'behaviors': behaviors,
                'pupil_centers': pupil_centers,
                'areas': areas,
                'layers': layers,
                'unit_ids': unit_ids,
                'trial_idxs': trial_idxs,
                'stimulus_info': stimulus_info,
                }, f)

    # fetch anatomy data if any stack is available
    filename = os.path.join(fetch_dir, 'anatomy.data_{}.pickle'.format(scan))
    if os.path.exists(filename):
        print('anatomy data already exists')
    else:
        print('anatomy data does not exist, fetching from server...')
        meso = dj.create_virtual_module('pipeline_meso', 'pipeline_meso')
        anatomy = data.InputResponse.ResponseKeys*meso.StackCoordinates.UnitInfo&key
        stack_sessions, stack_idxs = anatomy.fetch('stack_session', 'stack_idx')
        if len(anatomy)>0:
            stack_info = np.empty((len(anatomy),), dtype=np.object)
            for i in range(len(anatomy)):
                stack_info[i] = (stack_sessions[i], stack_idxs[i])
            stack_info = np.unique(stack_info)
            print('unique stack info:')
            for i in range(len(stack_info)):
                print('stack session: {}, stack idx {}'.format(stack_info[i][0], stack_info[i][1]))
            min_session_diff = min([abs(stack_session-key['session']) for stack_session, _ in stack_info])
            closest_stack = [abs(stack_session-key['session']) for stack_session, _ in stack_info].index(min_session_diff)
            print('use stack {}'.format(closest_stack))
            anatomy_info = {}
            anatomy_info['xs'], anatomy_info['ys'], anatomy_info['zs'], anatomy_info['unit_ids'] = \
                (anatomy&{'stack_session': stack_info[closest_stack][0], 'stack_idx': stack_info[closest_stack][1]}).fetch('stack_x', 'stack_y', 'stack_z', 'unit_id')

            print('coordinates shape {}'.format(np.array([anatomy_info['xs'], anatomy_info['ys'], anatomy_info['zs']]).shape))

            with open(filename, 'wb') as f:
                pickle.dump({
                    'key': key,
                    'anatomy_info': anatomy_info,
                    'stack_session': stack_info[closest_stack][0],
                    'stack_idx': stack_info[closest_stack][1],
                    }, f)
        else:
            print('no anatomy data available')


def split_scan(data_dir, scan):
    r"""Splits fetched scan data.

    Images, behaviors and responses from raw data are saved separately, and
    responses are splitted according to areas along with some simple analysis.

    Args
    ----
    data_dir: str
        The path of mouse data directory. Images and splitted responses will be
        saved in ``[data_dir]/images`` and ``[data_dir]/splitted``
        respectively.
    scan: str
        The scan name in the form of ``xxxxx-yy-zz``, in which ``xxxxx`` is
        animal ID, ``yy`` is session index and ``zz`` is scan index.

    """
    fetch_path = os.path.join(data_dir, 'fetched', 'response.data_{}.pickle'.format(scan))
    assert os.path.exists(fetch_path), 'fetched data not found'

    image_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    split_dir = os.path.join(data_dir, 'splitted')
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    filename = os.path.join(split_dir, '{}_basic.pickle'.format(scan))
    if os.path.exists(filename):
        print('raw data split already')
        with open(filename, 'rb') as f:
            saved = pickle.load(f)
        neuron_nums = saved['neuron_nums']
        loo_corrs = saved['loo_corrs']
        for area in ['V1', 'LM', 'AL', 'RL']:
            if neuron_nums[area]>0:
                print('area {}, {} neurons'.format(area, neuron_nums[area]))
                print('mean leave-one-out correlation {:.3f}'.format(loo_corrs[area].mean()))
        return

    print('splitting {}...'.format(scan))
    with open(fetch_path, 'rb') as f:
        saved = pickle.load(f)

    images, behaviors, pupil_centers = saved['images'], saved['behaviors'], saved['pupil_centers']
    stimulus_info = saved['stimulus_info']
    imagenet_ids = np.concatenate([stimulus_info['imagenet_ids'][stimulus_info['trial_idxs']==idx] for idx in saved['trial_idxs']])
    u_ids, counts = np.unique(imagenet_ids, return_counts=True)
    oracle_ids = u_ids[np.nonzero(counts>1)[0]]
    normal_ids = u_ids[np.nonzero(counts==1)[0]]
    print('{} unique images, in which {} are oracles'.format(len(u_ids), len(oracle_ids)))

    # save images as a dictionary
    dict_path = os.path.join(image_dir, 'imagenet_examples.pickle')
    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as f:
            image_dict = pickle.load(f)
    else:
        image_dict = {}
    image_dict.update(dict((i_id, img) for i_id, img in zip(imagenet_ids, images)))
    with open(dict_path, 'wb') as f:
        pickle.dump(image_dict, f)

    # save oracle and normal ids
    dict_path = os.path.join(image_dir, 'imagenet_ids.pickle')
    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as f:
            id_dict = pickle.load(f)
    else:
        id_dict = {'oracle': {}, 'normal': {}}
    id_dict['oracle'][scan] = oracle_ids
    id_dict['normal'][scan] = normal_ids
    with open(dict_path, 'wb') as f:
        pickle.dump(id_dict, f)

    # get indices according to oracle_ids and normal_ids, with order preserved
    oracle_idxs = [np.nonzero(imagenet_ids==oracle_id)[0] for oracle_id in oracle_ids]
    normal_idxs = [list(imagenet_ids).index(normal_id) for normal_id in normal_ids]

    oracle_nums = [len(idxs) for idxs in oracle_idxs]

    behaviors = {
        'oracle': np.concatenate([behaviors[idxs] for idxs in oracle_idxs]),
        'normal': behaviors[normal_idxs],
        }
    pupil_centers = {
        'oracle': np.concatenate([pupil_centers[idxs] for idxs in oracle_idxs]),
        'normal': pupil_centers[normal_idxs]
        }

    # split response data according to area
    neuron_nums = {}
    sn_ratios = {}
    scaled_means = {}
    loo_corrs = {}
    for area in ['V1', 'LM', 'AL', 'RL']:
        responses = saved['responses'][:, saved['areas']==area]
        neuron_nums[area] = responses.shape[1]
        if neuron_nums[area]>0:
            print('area {}, {} neurons'.format(area, neuron_nums[area]))
            oracle_responses = np.concatenate([responses[idxs] for idxs in oracle_idxs])
            normal_responses = responses[normal_idxs]

            idxs = np.cumsum([0]+oracle_nums)
            r_mean = np.array([oracle_responses[idxs[i]:idxs[i+1]].mean(axis=0) for i in range(len(oracle_nums))])
            r_var = np.array([oracle_responses[idxs[i]:idxs[i+1]].var(axis=0) for i in range(len(oracle_nums))])
            sn_ratios[area] = (r_mean.var(axis=0)/r_var.mean(axis=0))**0.5
            scaled_means[area] = oracle_responses.mean(axis=0)/(r_var.mean(axis=0)**0.5)

            loo_prediction = np.concatenate([utils.loo_mean(oracle_responses[idxs[i]:idxs[i+1]]) for i in range(len(oracle_nums))])
            loo_corrs[area] = utils.response_corrs(oracle_responses, loo_prediction)
            print('mean leave-one-out correlation {:.3f}'.format(loo_corrs[area].mean()))

            with open(os.path.join(split_dir, '{}_{}.pickle'.format(scan, area)), 'wb') as f:
                pickle.dump({
                    'oracle': oracle_responses,
                    'normal': normal_responses,
                    }, f)

    with open(os.path.join(split_dir, '{}_basic.pickle'.format(scan)), 'wb') as f:
        pickle.dump({
            'oracle_nums': oracle_nums,
            'oracle_ids': oracle_ids,
            'normal_ids': normal_ids,
            'behaviors': behaviors,
            'pupil_centers': pupil_centers,
            'neuron_nums': neuron_nums,
            'sn_ratios': sn_ratios,
            'scaled_means': scaled_means,
            'loo_corrs': loo_corrs,
            }, f)
