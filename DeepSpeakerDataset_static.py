from __future__ import print_function


import numpy as np

import torch.utils.data as data


def find_classes(voxceleb):
    classes = list(set([datum['speaker_id'] for datum in voxceleb]))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def generate_triplets(features, num_triplets,n_classes):
    def create_indices(_features):
        inds = dict()
        for idx, (feature_path,label) in enumerate(_features):
            if label not in inds:
                inds[label] = []
            inds[label].append(feature_path)
        return inds

    triplets = []
    # Indices = array of labels and each label is an array of indices
    indices = create_indices(features)

    #for x in tqdm(range(num_triplets)):
    for x in range(num_triplets):
        c1 = np.random.randint(0, n_classes)
        c2 = np.random.randint(0, n_classes)
        while len(indices[c1]) < 2:
            c1 = np.random.randint(0, n_classes)

        while c1 == c2:
            c2 = np.random.randint(0, n_classes)
        if len(indices[c1]) == 2:  # hack to speed up process
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, len(indices[c1]) - 1)
            n2 = np.random.randint(0, len(indices[c1]) - 1)
            while n1 == n2:
                n2 = np.random.randint(0, len(indices[c1]) - 1)
        if len(indices[c2]) ==1:
            n3 = 0
        else:
            n3 = np.random.randint(0, len(indices[c2]) - 1)

        triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3],c1,c2])
    return triplets



class DeepSpeakerDataset(data.Dataset):

    def __init__(self, voxceleb, dir, n_triplets,loader, transform=None, *arg, **kw):

        print('Looking for audio [wav] files in {}.'.format(dir))
        #voxceleb = read_voxceleb_structure(dir)

        #voxceleb = voxceleb[voxceleb['subset'] == 'dev']

        #voxceleb = voxceleb[1:5000]
        #voxceleb = voxceleb[445:448]

        if len(voxceleb) == 0:
            raise(RuntimeError(('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')))

        classes, class_to_idx = find_classes(voxceleb)
        features = []
        for vox_item in voxceleb:
            item = (vox_item['file_path'], class_to_idx[vox_item['speaker_id']])
            # item = (dir +'/voxceleb1_wav/' + vox_item['filename']+'.wav', class_to_idx[vox_item['speaker_id']])
            features.append(item)

        self.root = dir
        self.features = features
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

        self.n_triplets = n_triplets

        print('Generating {} triplets'.format(self.n_triplets))
        self.training_triplets = generate_triplets(self.features, self.n_triplets,len(self.classes))



    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single feature

        Returns:

        '''
        def transform(feature_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
            """

            feature = self.loader(feature_path)
            return self.transform(feature)

        # Get the index of each features in the triplet
        a, p, n,c1,c2 = self.training_triplets[index]

        # transform features if required
        feature_a, feature_p, feature_n = transform(a), transform(p), transform(n)
        return feature_a, feature_p, feature_n,c1,c2

    def __len__(self):
        return len(self.training_triplets)