from glob import glob
from torch.utils.data import DataLoader, Dataset
import sys
import torchaudio
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import normalize
from torch.utils.data.sampler import SubsetRandomSampler


AUDIO_PATH = 'dataset/kz/Audios_flac'
TRANS_PATH = 'dataset/kz/Transcriptions'

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.9
TEST_SPLIT = 1
shuffle_dataset = True


# TRAIN_IDX = int(DATASET_LEN*0.8)
# VAL_IDX = int((TRAIN_IDX + DATASET_LEN)/2)
# TEST_IDX = DATASET_LEN
# print(TRAIN_IDX)
# print(VAL_IDX)







class TextProcess:
	def __init__(self):
		char_map_str = """ 
		<SPACE> 0
		а 1
		ә 2
		б 3 
		в 4
		г 5
		ғ 6
		д 7
		е 8
		ё 9
		ж 10
		з 11
		и 12
		й 13
		к 14
		қ 15
		л 16
		м 17
		н 18
		ң 19
		о 20
		ө 21
		п 22
		р 23
		с 24
		т 25
		у 26
		ұ 27
		ү 28
		ф 29
		х 30
		һ 31
		ц 32
		ч 33
		ш 34
		щ 35
		ъ 36
		ы 37
		і 38
		ь 39
		э 40
		ю 41
		я 42
		"""
		self.char_map = {}
		self.idx_map = {}
		for line in char_map_str.strip().split("\n"):
			ch, idx = line.split()
			self.char_map[ch] = int(idx)
			self.idx_map[int(idx)] = ch

		self.idx_map[1] = ''

	
	def text_to_int_sequence(self, text):
		int_sequence = []
		for c in text:
			if c == ' ':
				ch = self.char_map['<SPACE>']
			else:
				ch = self.char_map[c]
			int_sequence.append(ch)
		
		# orig_length = len(int_sequence)
		# # post zero padding 
		# if len(int_sequence) < 500:
		# 	zeros = [0] * (500 - len(int_sequence))
		# 	int_sequence = int_sequence + zeros

		return torch.tensor(int_sequence)


	def int_to_text_sequence(self, labels):
		"Use a character map and convert integer labels into a text sequence"
		string = []
		for i in labels:
			string.append(self.idx_map[i])

		return ''.join(string).replace('<SPACE>', ' ') 




class LogMelSpec(nn.Module):

	def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
		super(LogMelSpec, self).__init__()
		self.transform = torchaudio.transforms.MelSpectrogram(
							sample_rate=sample_rate, n_mels=n_mels,
							win_length=win_length, hop_length=hop_length
						)



	def forward(self, t):
		t = self.transform(t)  # mel spectrogram
		t = np.log(t + 1e-14)  # logorithmic, add small value to avoid inf
		t = normalize(t[0])
		# orig_length = t[0].shape
		t = t[None, :] # add a new axis
		# # # post zero padding mfccs
		# # padded_t = torch.zeros(t.shape[0],t.shape[1],5000)
		# # padded_t[:,:,:t.shape[2]] = t
		return t



class SpecAugment(nn.Module):

	def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
		super(SpecAugment, self).__init__()

		self.rate = rate

		self.specaug = nn.Sequential(
			torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
			torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
		)

		self.specaug2 = nn.Sequential(
			torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
			torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
			torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
			torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
		)

		policies = { 1: self.policy1, 2: self.policy2, 3: self.policy3 }
		self._forward = policies[policy]

	def forward(self, t):
		return self._forward(t)

	def policy1(self, t):
		probability = torch.rand(1, 1).item()
		if self.rate > probability:
			return  self.specaug(t)
		return t

	def policy2(self, t):
		probability = torch.rand(1, 1).item()
		if self.rate > probability:
			return  self.specaug2(t)
		return t

	def policy3(self, t):
		probability = torch.rand(1, 1).item()
		if probability > 0.5:
			return self.policy1(t)
		return self.policy2(t)



class Data(Dataset):
	parameters = {
        "sample_rate": 8000, "n_feats": 81,
        "specaug_rate": 0.5, "specaug_policy": 3,
        "time_mask": 70, "freq_mask": 15 
    }

	def __init__(self, audio_path, transcripts_path, sample_rate,
				n_feats, specaug_rate, specaug_policy, time_mask,
				freq_mask, valid=False, log_ex=True):

		self.t = glob(audio_path + "/*.flac")
		self.y = glob(transcripts_path + "/*.txt")
		
		if len(self.t) != len(self.y):
			raise(f'Number of t and Y files do not match, ({len(self.t)} t and {len(self.y)} Y)')
			sys.exit()
		
		self.dataset_len = len(self.t)
		self.text_process = TextProcess()
		self.log_ex = log_ex

		if valid:
			self.audio_transforms = torch.nn.Sequential(
				LogMelSpec(sample_rate=sample_rate, n_mels=n_feats,  win_length=160, hop_length=80)
			)
		else:
			self.audio_transforms = torch.nn.Sequential(
				LogMelSpec(sample_rate=sample_rate, n_mels=n_feats,  win_length=160, hop_length=80),
				SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask),
			)



	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.item()
	
		try:			
			file_path = self.t[idx]
			waveform = torchaudio.load(file_path)[0]
			text = open(self.y[idx], 'r').read()
			label = self.text_process.text_to_int_sequence(text)
			spectrogram = self.audio_transforms(waveform)

			spec_len = spectrogram.shape[-1]
			label_len = len(label)
			# print(spectrogram.shape)
			# print(label.shape)
			# sys.exit()
			
			if spec_len < label_len:
				raise Exception('spectrogram len is bigger then label len')
			if spectrogram.shape[0] > 1:
				raise Exception('dual channel, skipping audio file %s'%file_path)
			if spectrogram.shape[2] > 5500:
				raise Exception('spectrogram too big. size %s'%spectrogram.shape[2])
			if label_len == 0:
				raise Exception('label len is zero... skipping %s'%file_path)
			if label_len > 400:
				raise Exception('label len is too big... skipping %s'%file_path)

		except Exception as e:
			if self.log_ex:
				print(str(e), file_path)
			return self.__getitem__(idx - 1 if idx != 0 else idx + 1)  

		return spectrogram, label, spec_len, label_len

	def __len__(self):
		return self.dataset_len



def collate_fn_padd(data):
	'''
	Padds batch of variable length
	note: it converts things ToTensor manually here since the ToTensor transform
	assume it takes in images rather than arbitrary tensors.
	'''
	# print(data)
	spectrograms = []
	labels = []
	input_lengths = []
	label_lengths = []
	
	for (spectrogram, label, input_length, label_length) in data:
		if spectrogram is None:
			continue
		# print(spectrogram.shape)
		spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
		labels.append(label)
		input_lengths.append(input_length)
		label_lengths.append(label_length)
		
	spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
	labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
	input_lengths = input_lengths
	# print(spectrograms.shape)
	label_lengths = label_lengths

	# ## compute mask
	# mask = (batch != 0).cuda(gpu)
	# return batch, lengths, mask
	return spectrograms, labels, input_lengths, label_lengths

batch_size = 16
d_params = Data.parameters
dataset = Data(audio_path=AUDIO_PATH, transcripts_path=TRANS_PATH, **d_params)


indices = list(range(dataset.__len__()))

if shuffle_dataset:
	np.random.shuffle(indices)

train_idx = indices[:int(np.floor(TRAIN_SPLIT*len(indices)))]
val_idx = indices[int(np.floor(TRAIN_SPLIT*len(indices))):int(np.floor(VAL_SPLIT*len(indices)))]
test_idx = indices[int(np.floor(VAL_SPLIT*len(indices))):int(np.floor(TEST_SPLIT*len(indices)))]

train_loader = SubsetRandomSampler(train_idx)
val_loader = SubsetRandomSampler(val_idx)
# test_loader = SubsetRandomSampler(test_idx)


train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
							 num_workers=3, sampler=train_loader,
                            pin_memory=True, collate_fn=collate_fn_padd)
val_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, 
							num_workers=3, sampler=val_loader, 
							pin_memory=True, collate_fn=collate_fn_padd)
# test_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=test_loader)

