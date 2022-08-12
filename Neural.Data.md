# Description of neural data
The data folder, typically located at `store/data`, contains three subfolders, `fetched`, `images`
and `splitted`. All three folders contain `.pickle` files that can be loaded with protocol>=4.

The data consists mouse neural responses to static gray-scale images, with neurons recorded from
cortical areas 'V1', 'LM', 'AL', 'RL'. Raw data were fetched from Tolias lab database to `fetched`,
and saved to `images` and `splitted` after some preprocessing. Data are organized by scan, with IDs
formatted as `'{animal_id}-{session}-{scan_idx}'` like '20210-4-11'. Each scan contains thousands of
trials, each of which has all neural responses to one image. All trials are divided into two
categories, 'normal' trials and 'oracle' trials. The images used in 'normal' trials are different
from each trial, while the images in 'oracle' trials are repeated for a few times. For example, we
can use 100 different 'oracle' images, each repeated 10 times, and obtain 1000 'oracle' trials. All
trials use gray-scale downsampled ImageNet images and were randomly shuffled. The motivation of
including 'oracle' trials to experimentally measure the reproducibility of neuron responses, i.e.
the intrinsic noise of neural representation.

Training of neural predictive neural models will use data in `images` and `splitted` only.

## Folder `fetched`
For each scan, there are two files saved in `fetched`, `'anatomy.data_{scan_id}.pickle'` and
`'response.data_{scan_id}.pickle'`.

`'anatomy.data_{scan_id}.pickle'` is a dictionary with following items:
* 'key': a dictionary with scan information.
* 'anatomy_info': a dictionary with following items:
    * 'xs', 'ys', 'zs': arrays with shape `(num_neurons,)` for neuron coordinates.
    * 'unit_ids': an array with shape `(num_neurons,)` for neuron IDs.
* 'stack_session', 'stack_idx': scan information.

`'response.data_{scan_id}.pickle'` is a dictionary with following items:
* 'key': a dictionary with scan information.
* 'responses': an array with shape `(num_trials, num_neurons)` for each neuron response in each
trial.
* 'images': an array with shape `(num_trials, 1, 36, 64)` for images in each trial.
* 'behaviors': an array with shape `(num_trials, 3)` for behavior data, including running speed,
pupil size and pupil size change information.
* 'pupil_centers': an array with shape `(num_trials, 2)` for pupil center coordinates.
* 'areas', 'layers', 'unit_ids': arrays with shape `(num_neurons,)` for neuron information.
* 'trial_idxs': an array with shape `(num_neurons,)` for the present order of each trial.
* 'stimulus_info': a dictionary with more information about stimulus used in the scan.

## Folder `images`
Images used all scans are mostly shared, therefore we extract all the used ones and save them in the
`images` folder. There are two files, `'imagenet_examples.pickle'` and `'imagenet_ids.pickle'`.

`'imagenet_examples.pickle'` is a dictionary with ImageNet image IDs as keys, and each value is a
`(1, 36, 64)` array whose values are in `[0, 255]`.

`'imagenet_ids.pickle'` is a dictionary containing image IDs used in all scans, with two items:
* 'oracle': a dictionary with scan IDs as keys, each value is an array of image IDs used as oracle
images in that scan. Number of repeats of each oracle image will be specified by files in `splitted`
folder.
* 'normal': a dictionary with scan IDs as keys, each value is an array of image IDs used as normal
images in that scan.

## Folder `splitted`
For each scan, there is a file `'{scan_id}_basic.pickle'` with basic information. It is a dictionary
with following items:
* 'oracle_nums': a list of length `num_oracle_imgs`, containing the numbers of repeats for each oracle
image. Because some trials are discarded due to the poor quality, the number of trials may differ
for different oracle images.
* 'oracle_ids': an array of shape `(num_oracle_imgs,)` for oracle image IDs.
* 'normal_ids': an array of shape `(num_normal_imgs,)` for normal image IDs.
* 'behaviors', 'pupil_centers': dictionaries containing behavior and pupil position data, with the
following items:
    * 'oracle': an array of shape `(num_oracle_trials, 3)` or `(num_oracle_trials, 2)`.
    `num_oracle_trials` is the sum of `oracle_nums` and the data are ordered along the first axis
    as `num_oracle_imgs` chunks.
    * 'normal': an array of shape `(num_normal_trials, 3)` or `(num_normal_trials, 2)`, with
    `num_normal_trials` the same as `num_normal_imgs` since each normal image was only used once.
* `neuron_nums`: a dictionary of numbers of neurons in each cortical area, with 'V1', 'LM', 'AL' and
'RL' as keys.

For each recorded cortical area, the response data is saved in `'{scan_id}_{area}.pickle'`, each is
a dictionary with the following items:
* 'oracle': an array of `(num_oracle_trials, num_neurons)` for neural responses in oracle trials,
sorted in the same order as 'behaviors' and 'pupil_centers'.
* 'normal': an array of `(num_normal_trials, num_neurons)` for neural responses in normal trials,
sorted in the same order as 'behaviors' and 'pupil_centers'.

## File preparation
All data files were originally prepared by `'prepare.mouse.data.ipynb'` file, which requires
[`attorch==0.0.0`](https://github.com/atlab/attorch) and
[`neuro_data==0.0.0`](https://github.com/cajal/neuro_data). Both packages are for internal use in
Tolias lab, and are depcrecated for the use here.
