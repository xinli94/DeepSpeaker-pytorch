import argparse
import glob
import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms

from audio_processing import toMFB, totensor, truncatedinput,read_audio
from model import DeepSpeakerModel

class EmbedSet(data.Dataset):
    def __init__(self, audio_path, loader, transform=None):
        self.audio_path = audio_path
        # self.audio_list = list(glob.glob(os.path.join(audio_path, '*.wav')))

        self.audio_list = []
        for root, dirs, files in os.walk(audio_path):
            for file_name in files:
                if os.path.splitext(file_name)[-1] != '.wav':
                    continue
                file_path = os.path.join(root, file_name)
                self.audio_list.append(file_path)
        print('>>>>>>>>>', self.audio_list)

        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        def transform(audio_path):
            audio = self.loader(audio_path)
            return self.transform(audio)
        return transform(self.audio_list[index])

    def __len__(self):
        return len(self.audio_list)

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
    parser.add_argument('--audio-path',
                        type=str,
                        default='/data5/xin/voxceleb/raw_data/test/id00017/01dfn2spqyE/',
                        help='path to dataset')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        metavar='PATH',
                        required=True,
                        help='path to latest checkpoint (default: none)')
    # parser.add_argument('--batch-size', type=int, default=10, metavar='BS',
    #                     help='input batch size for training (default: 512)')
    # parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
    #                     help='Dimensionality of the embedding')
    # parser.add_argument('--num-classes', type=int, default=5994, metavar='ES',
    #                     help='Number of classes')

    args = parser.parse_args()
    args.cuda = not args.no_cuda

    """ TODO(xin)
    - Right now embedding_size is hardcoded in log_dir name
    # LOG_DIR = args.log_dir + '/run-optim_{}-n{}-lr{}-wd{}-m{}-embeddings{}-msceleb-alpha10'\
    #     .format(args.optimizer, args.n_triplets, args.lr, args.wd,
    #             args.margin,args.embedding_size)
    - Should move to model state_dict
    """
    args.embedding_size = int(os.path.dirname(args.checkpoint).split('-')[-3].split('embeddings')[-1].strip())

    # TODO(xin): Support batching
    args.batch_size = 1

    # TODO(xin): Add num_classes to state_dict
    args.num_classes = 5994

    return args

# def create_optimizer(model, new_lr):
#     if args.optimizer == 'sgd':
#         optimizer = optim.SGD(model.parameters(), lr=new_lr,
#                               momentum=0.9, dampening=0.9,
#                               weight_decay=args.wd)
#     elif args.optimizer == 'adam':
#         optimizer = optim.Adam(model.parameters(), lr=new_lr,
#                                weight_decay=args.wd)
#     elif args.optimizer == 'adagrad':
#         optimizer = optim.Adagrad(model.parameters(),
#                                   lr=new_lr,
#                                   lr_decay=args.lr_decay,
#                                   weight_decay=args.wd)
#     return optimizer

def main():
    args = parse_arguments()
    print('==> args: {}'.format(args))

    transform_embed = transforms.Compose([
        truncatedinput(),
        toMFB(),
        totensor(),
    ])

    file_loader = read_audio

    inference_set = EmbedSet(audio_path=args.audio_path,
                             loader=file_loader,
                             transform=transform_embed)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    inference_loader = torch.utils.data.DataLoader(inference_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    # instantiate model and initialize weights
    package = torch.load(args.checkpoint)
    # print('==> package: {}'.format(package))
    model = DeepSpeakerModel(embedding_size=args.embedding_size,
                             num_classes=args.num_classes)
    if args.cuda:
        model.cuda()

    # optimizer = create_optimizer(model, args.lr)

    model.load_state_dict(package['state_dict'])
    # optimizer.load_state_dict(package['optimizer'])

    model.eval()

    pbar = tqdm(enumerate(inference_loader))
    for batch_idx, data in pbar:
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        out = model(data)

        features = out.detach().cpu().numpy()
        print('>>>>>>>>>>>>>>> features', features)
        assert features.shape == (1, args.embedding_size)

if __name__ == '__main__':
    main()

