from sklearn.utils import shuffle
import torch
import torch.utils.data


class MelLoader(torch.utils.data.Dataset):
    def __init__(self, inputs_path, targets_path, train_identity):
        inputs = torch.load(inputs_path)
        targets = torch.load(targets_path)

        if train_identity:
            inputs = inputs + targets
            targets = targets + targets
        
        self.inputs_shuffled, self.targets_shuffled = shuffle(inputs, targets, random_state=1234)

    def __getitem__(self, index):
        input = self.inputs_shuffled[index]
        target = self.targets_shuffled[index]

        return input, target

    def __len__(self):
        return len(self.inputs_shuffled)


class MelCollate():
    def __call__(self, batch):
        """ batch = [mel_inferred_resized, mel_original_resized] """

        mel_in = torch.cuda.FloatTensor(len(batch), 1, batch[0][0].size(0), batch[0][0].size(1))
        mel_out = torch.cuda.FloatTensor(len(batch), 1, batch[0][0].size(0), batch[0][0].size(1))
        for i in range(len(batch)):
            mel_in[i] = batch[i][0]
            mel_out[i] = batch[i][1]

        return mel_in, mel_out
