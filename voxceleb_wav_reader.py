import collections
import glob
import os
import random
from tqdm import tqdm

random.seed(12345)

# import numpy as np

# np.set_printoptions(threshold=np.nan)

def read_voxceleb_structure(directory, test_only=False):
    voxceleb = []
    speakers = set()
    for subset in os.listdir(directory):
        if not os.path.isdir(os.path.join(directory, subset)) or (test_only and subset == 'dev'):
            continue
        # root_dir/subset/speaker_id/uri/wav
        key = os.path.join(directory, subset) + '/*/*/*.wav'
        print('==> Reading {} set'.format(subset))
        for file_path in tqdm(glob.glob(key)):
            subset, speaker, uri, file = file_path.split('/')[-4:]
            voxceleb.append({'filename': file, 'speaker_id': speaker, 'uri': uri, 'subset': subset, 'file_path': file_path})
            speakers.add(speaker)
    print('Found {} files with {} different speakers.'.format(len(voxceleb), len(speakers)))

    voxceleb_dev = [datum for datum in voxceleb if datum['subset']=='dev']
    voxceleb_test = [datum for datum in voxceleb if datum['subset']=='test']

    # return voxceleb
    return voxceleb_dev, voxceleb_test

def generate_test_dir(voxceleb):
    voxceleb_test = []
    speakers_test = collections.defaultdict(list)
    for item in voxceleb:
        if item['subset'] == 'test':
            voxceleb_test.append(item['file_path'])
            speakers_test[item['speaker_id']].append(item['file_path'])
    voxceleb_test = [item['file_path'] for item in voxceleb if item['subset'] == 'test']

    # # negative
    pairs = list(zip(voxceleb_test, sorted(voxceleb_test, key=lambda x: random.random())))
    # positive
    for values in speakers_test.values():
        pairs += list(zip(values, sorted(values, key=lambda x: random.random())))

    csv_path = os.path.join(directory, 'test_pairs.csv')
    with open(csv_path, 'w') as f:
        for item in tqdm(pairs):
            item = list(item)
            for i in range(len(item)):
                item[i] = item[i].split('/test/')[-1].strip()

            # print('>>>>>>>>>>>', item, item[0].split('/')[0], item[1].split('/')[0])

            issame = '1' if item[0].split('/')[0] == item[1].split('/')[0] else '0'
            line = ' '.join([issame, item[0], item[1]])
            f.write(line + '\n')
        # f.write('\n'.join([','.join(item) for item in pairs]))
    print('==> Generated and saved test pairs to {}'.format(csv_path))


if __name__ == '__main__':
    directory = '/data5/xin/voxceleb/raw_data/'
    voxceleb = read_voxceleb_structure(directory, test_only=True)
    generate_test_dir(voxceleb)


# voxceleb_dir = 'voxceleb'
# data_uem = 'data/voxceleb1.{subset}.uem'
# data_mdtm = 'data/voxceleb1.{subset}.mdtm'

# list_txt = '{voxceleb_dir}/list.txt'.format(voxceleb_dir=voxceleb_dir)
# glob_exp = '{voxceleb_dir}/voxceleb1_txt/*/*.txt'.format(voxceleb_dir=voxceleb_dir)




# def parse_txt(txt):
#     lines = [line.strip() for line in open(txt, 'r').readlines()]
#     speaker = lines[0].split('\t')[-1]
#     uri = lines[1].split('\t')[-1]
#     duration = float(lines[2].split('\t')[-1].split()[0])
#     subset = lines[3].split('\t')[-1]

#     file_list = []
#     for line in lines[5:]:
#         file_location, start, end = line.split()
#         file_list.append(file_location)


#     return subset, uri, speaker, file_list



# def find_files(directory, pattern='*/*/*/*.wav'):
#     """Recursively finds all files matching the pattern."""
#     return glob(os.path.join(directory, pattern), recursive=True)


# def read_voxceleb_structure(directory):
#     voxceleb = []

#     #for path_txt in tqdm(glob(glob_exp)):
#     for path_txt in glob(glob_exp):
#         subset, uri, speaker, file_list = parse_txt(path_txt)

#         for file in file_list:
#             voxceleb.append({'filename': file, 'speaker_id': speaker, 'uri': uri, 'subset': subset})

#     #voxceleb = pd.DataFrame(filelist)
#     num_speakers = len(set([datum['speaker_id'] for datum in voxceleb]))
#     print('Found {} files with {} different speakers.'.format(str(len(voxceleb)).zfill(7), str(num_speakers).zfill(5)))
#     #print(voxceleb.head(10))
#     return voxceleb


