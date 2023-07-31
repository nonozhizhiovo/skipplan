import os
import json
import numpy as np
import math
import torch
import pandas as pd

def get_vids(path):
    task_vids = {}
    with open(path, 'r') as f:
        for line in f:
            task, vid, url = line.strip().split(',')
            if task not in task_vids:
                task_vids[task] = []
            task_vids[task].append(vid)
    return task_vids

def read_task_info(path):
    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(path, 'r') as f:
        idx = f.readline()
        while idx is not '':
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(',')
            next(f)
            idx = f.readline()
    return {'title': titles, 'url': urls, 'n_steps': n_steps, 'steps': steps}


class dataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 max_traj_len,
                 is_val,
                 frameduration,
                 dataset_mode):
        self.frameduration = frameduration
        self.is_val = is_val
        self.root = root
        self.max_traj_len = max_traj_len
        self.dataset_mode = dataset_mode

        self.features_path = os.path.join(
            root, 'crosstask_features'
        )
        self.constraints_path = os.path.join(
            root, 'crosstask_release', 'annotations'
        )

        primary_info = read_task_info(os.path.join(
            root, 'crosstask_release', 'tasks_primary.txt'))
        self.highlevel = primary_info['title']
        self.lowlevel = primary_info['steps']
        self.n_steps = primary_info['n_steps']

        if self.dataset_mode == 'single':
            cross_task_data_name = 'cross_task_data_single_{}.json'.format(is_val)

            if os.path.exists(cross_task_data_name):
                with open(cross_task_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(cross_task_data_name))
            else:
                val_csv_path = os.path.join(
                    root, 'crosstask_release', 'videos_val.csv')
                video_csv_path = os.path.join(
                    root, 'crosstask_release', 'videos.csv')
                all_task_vids = get_vids(video_csv_path)
                val_vids = get_vids(val_csv_path)
                if is_val:
                    task_vids1 = val_vids
                else:
                    task_vids1 = {task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]] for
                                 task, vids in
                                 all_task_vids.items()}


                all_tasks = set(self.n_steps.keys())
                task_vids = {task: vids for task,
                            vids in task_vids1.items() if task in all_tasks}

                all_vids = []
                for task, vids in task_vids.items():
                    all_vids.extend([(task, vid) for vid in vids])
                json_data = []
                for idx in range(len(all_vids)):
                    # for idx2 in range():
                    task, vid = all_vids[idx]
                    video_path = os.path.join(
                        self.features_path, str(vid) + '.npy')
                    json_data.append({'id': {'vid': vid, 'task': task, 'feature': video_path},
                                      'instruction_len': self.n_steps[task]})
                print('All primary task videos: {}'.format(len(json_data)))
                self.json_data = json_data
                with open('cross_task_data_single_{}.json'.format(is_val), 'w') as f:
                    json.dump(json_data, f)
                print('Save to {}'.format(cross_task_data_name))

            vid_names = []
            frame_cnts = []
            for listdata in self.json_data:
                vid_names.append(listdata['id'])
                frame_cnts.append(listdata['instruction_len'])
            self.vid_names = vid_names
            self.frame_cnts = frame_cnts

        elif self.dataset_mode == 'multiple':
            cross_task_data_name = 'cross_task_data_multiple_{}_{}.json'.format(is_val, self.max_traj_len)

            if os.path.exists(cross_task_data_name):
                with open(cross_task_data_name, 'r') as f:
                    self.json_data = json.load(f)
                print('Loaded {}'.format(cross_task_data_name))
            else:
                val_csv_path = os.path.join(
                    root, 'crosstask_release', 'videos_val.csv')
                video_csv_path = os.path.join(
                    root, 'crosstask_release', 'videos.csv')
                all_task_vids = get_vids(video_csv_path)
                val_vids = get_vids(val_csv_path)
                if is_val:
                    task_vids1 = val_vids
                else:
                    task_vids1 = {task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]] for
                                 task, vids in
                                 all_task_vids.items()}

                all_tasks = set(self.n_steps.keys())

                task_vids0 = {task: vids for task,
                                            vids in task_vids1.items() if task in all_tasks}


                task_vids = {task: [vid for vid in vids if (len(pd.read_csv(os.path.join(self.constraints_path, task + '_' + vid + '.csv'))) + 1) >= self.max_traj_len] for task, vids in task_vids0.items()}

                all_vids = []
                for task, vids in task_vids.items():
                    all_vids.extend([(task, vid) for vid in vids])
                json_data = []
                for idx in range(len(all_vids)):
                    task, vid = all_vids[idx]
                    for idx2 in range(((len(pd.read_csv(os.path.join(self.constraints_path, task + '_' + vid + '.csv'))) + 1) - self.max_traj_len)):
                        video_path = os.path.join(
                            self.features_path, str(vid) + '.npy')
                        json_data.append({'id': {'vid': vid, 'task': task, 'feature': video_path},
                                          'instruction_len': self.n_steps[task], 'start_ind': idx2})
                print('All primary task videos: {}'.format(len(json_data)))
                self.json_data = json_data
                with open('cross_task_data_multiple_{}_{}.json'.format(is_val, self.max_traj_len), 'w') as f:
                    json.dump(json_data, f)
                print('Save to {}'.format(cross_task_data_name))

            vid_names = []
            frame_cnts = []
            start_ind = []
            for listdata in self.json_data:
                vid_names.append(listdata['id'])
                frame_cnts.append(listdata['instruction_len'])
                start_ind.append(listdata['start_ind'])
            self.vid_names = vid_names
            self.frame_cnts = frame_cnts
            self.start_ind = start_ind

    def read_assignment(self, T, task, all_n_steps, path):
        base = 0
        for k, v in all_n_steps.items():
            if k == task:
                break
            base += v
        Y = np.zeros([T, sum(all_n_steps.values())], dtype=np.uint8)
        legal_range = []
        lowlevel_lang = []

        with open(path, 'r') as f:
            for line in f:
                step, start, end = line.strip().split(',')
                start = int(math.floor(float(start)))
                end = int(math.floor(float(end)))
                lowlevel_lang.append(self.lowlevel[task][int(step)-1])
                step = int(step) - 1 + base
                Y[start:end, step] = 1
                legal_range.append((start, end))

        return Y, legal_range, lowlevel_lang

    def curate_dataset(self, images, labels_matrix, legal_range):
        images_list = []
        labels_onehot_list = []
        idx_list = []
        for start_idx, end_idx in legal_range:
            idx = (end_idx + start_idx) // 2
            idx_list.append(idx)
            label_one_hot = labels_matrix[idx]
            image_start_idx = max(0, (idx - self.frameduration // 2))
            image_start = images[image_start_idx: image_start_idx + self.frameduration]
            images_list.append(image_start)
            labels_onehot_list.append(label_one_hot)
        images_list.append(images[end_idx - self.frameduration:end_idx])
        return images_list, labels_onehot_list, idx_list

    def sample_single(self, index):
        vid_id = self.vid_names[index]
        frames = []
        lowlevel_labels = []
        highlevel_labels = []
        lowlevel_langs = []
        highlevel_langs = self.highlevel[vid_id['task']]
        highlevel_labels_str = vid_id['task']

        if highlevel_labels_str == '23521':
            highlevel_labels_num = 0
        elif highlevel_labels_str == '59684':
            highlevel_labels_num = 1
        elif highlevel_labels_str == '71781':
            highlevel_labels_num = 2
        elif highlevel_labels_str == '113766':
            highlevel_labels_num = 3
        elif highlevel_labels_str == '105222':
            highlevel_labels_num = 4
        elif highlevel_labels_str == '94276':
            highlevel_labels_num = 5
        elif highlevel_labels_str == '53193':
            highlevel_labels_num = 6
        elif highlevel_labels_str == '105253':
            highlevel_labels_num = 7
        elif highlevel_labels_str == '44047':
            highlevel_labels_num = 8
        elif highlevel_labels_str == '76400':
            highlevel_labels_num = 9
        elif highlevel_labels_str == '16815':
            highlevel_labels_num = 10
        elif highlevel_labels_str == '95603':
            highlevel_labels_num = 11
        elif highlevel_labels_str == '109972':
            highlevel_labels_num = 12
        elif highlevel_labels_str == '44789':
            highlevel_labels_num = 13
        elif highlevel_labels_str == '40567':
            highlevel_labels_num = 14
        elif highlevel_labels_str == '77721':
            highlevel_labels_num = 15
        elif highlevel_labels_str == '87706':
            highlevel_labels_num = 16
        elif highlevel_labels_str == '91515':
            highlevel_labels_num = 17

        highlevel_labels.append(highlevel_labels_num)
        highlevel_labels = torch.reshape(torch.tensor(highlevel_labels, dtype=torch.float32), (1, 1))

        imageso = np.load(os.path.join(self.features_path,
                                      vid_id['vid']+'.npy'))[:, :1024]
        cnst_path = os.path.join(
            self.constraints_path, vid_id['task'] + '_' + vid_id['vid'] + '.csv'
        )

        labels_matrix, legal_range, lowlevel_lang = self.read_assignment(
            imageso.shape[0], vid_id['task'], self.n_steps, cnst_path)

        legal_range = [(start_idx, end_idx) for (
            start_idx, end_idx) in legal_range if end_idx < imageso.shape[0] + 1]


        images, labels_matrix, idx_list = self.curate_dataset(
            imageso, labels_matrix, legal_range
        )

        if self.dataset_mode == 'single':

            if len(labels_matrix) > self.max_traj_len:
                idx = np.random.randint(
                    0, len(labels_matrix) - self.max_traj_len
                )
            else:
                idx = 0

            for i in range(self.max_traj_len):
                frames.extend(
                    images[min(idx + i, len(images) - 1)]
                )

            frames = torch.tensor(frames)

            if idx - 1 < 0:
                lowlevel_labels.append([0])
                lowlevel_langs.append('empty')
            else:
                lowlevel_label = labels_matrix[idx - 1]
                ind = np.unravel_index(
                    np.argmax(lowlevel_label, axis=-1), lowlevel_label.shape
                )
                lowlevel_labels.append(ind)
                lowlevel_langs.append(lowlevel_lang[idx])

            for i in range(self.max_traj_len):
                if idx + i < len(labels_matrix):
                    lowlevel_label = labels_matrix[idx+i]
                    ind = np.unravel_index(
                        np.argmax(lowlevel_label, axis=-1), lowlevel_label.shape
                    )
                    lowlevel_labels.append(ind)
                    lowlevel_langs.append(lowlevel_lang[idx+i])
                else:
                    lowlevel_labels.append([0])
                    lowlevel_langs.append('empty')
            lowlevel_labels = torch.tensor(lowlevel_labels, dtype=torch.float32)

        elif self.dataset_mode == 'multiple':
            assert len(labels_matrix) >= self.max_traj_len
            start_idx = self.start_ind[index]
            for i in range(self.max_traj_len):
                frames.extend(images[start_idx + i])
                lowlevel_label = labels_matrix[start_idx + i]
                ind = np.unravel_index(
                        np.argmax(lowlevel_label, axis=-1), lowlevel_label.shape
                    )
                lowlevel_labels.append(ind)
                lowlevel_langs.append(lowlevel_lang[start_idx + i])
            goal_ind = legal_range[start_idx + self.max_traj_len - 1][1]
            frames.extend(imageso[goal_ind-self.frameduration:goal_ind, :])
            frames = torch.tensor(frames)
            lowlevel_labels = torch.tensor(lowlevel_labels, dtype=torch.float32)
            # if frames.shape[0] < 4:
            #     frames = frames.repeat(2, 1)

        return frames, lowlevel_labels, highlevel_labels, lowlevel_langs, highlevel_langs, vid_id


    def __getitem__(self, index):

        frames, lowlevel_labels, highlevel_labels, lowlevel_langs, highlevel_langs, vid_id = self.sample_single(index)
        return frames, lowlevel_labels, highlevel_labels, lowlevel_langs, highlevel_langs, vid_id

    def __len__(self):
        return min(len(self.json_data), len(self.frame_cnts))