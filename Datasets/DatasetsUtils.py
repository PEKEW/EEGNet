import torch
FPS = 30-1


# todo 这里要考虑不同长度的序列 而且要考虑不同的数据类型
class SequenceCollator:
    def __init__(self,
                sequence_length=None,
                padding_mode='zero',
                include=['eeg','label','optical', 'original']):
        self.sequence_length = sequence_length
        self.padding_mode = padding_mode
        self.include = include

    
    def _pad_sequence(self, 
                    sequence: torch.Tensor, 
                    max_length: int, 
                    pad_dims: tuple) -> torch.Tensor:
        curr_length = sequence.size(0)
        if curr_length > max_length:
            return sequence[:max_length]
        pad_length = max_length - curr_length
        if self.padding_mode == 'zero':
            pad = torch.zeros((pad_length, *pad_dims), dtype=sequence.dtype)
        else:  # repeat mode
            repeat_dims = (pad_length,) + tuple(1 for _ in pad_dims)
            pad = sequence[-1].unsqueeze(0).repeat(*repeat_dims)
        return torch.cat([sequence, pad], dim=0)
        
    def __call__(self, batch):
        """
        可序列化的 collate 函数
        """
        batch = [b for b in batch if b is not None]
        if not batch:
            return []
        
        max_length = {}
        for key in self.include:
            max_length[key] = max(s[key].size(0) for s in batch)
        
        if self.sequence_length is not None:
            max_lengths = {k: self.sequence_length for k in max_lengths}
        
        processed_batch = {
            'sub_id': [],
            'slice_id': [],
            'label': [],
            'lengths': []
        }

        processed_batch.update({k: [] for k in self.include})

        for sample in batch:
            processed_batch['sub_id'].append(sample['sub_id'])
            processed_batch['slice_id'].append(sample['slice_id'])
            processed_batch['label'].append(sample['label'])
            for key in self.include:
                processed_batch[key].append(sample[key])

        result = {
            'sub_id': processed_batch['sub_id'],
            'slice_id': processed_batch['slice_id'],
        }
        result['lengths'] = torch.tensor(processed_batch['lengths'])
        result['label'] = torch.stack(processed_batch['label'])

        for key in self.include:
            result[key] = torch.stack(processed_batch[key])
        
        return result

