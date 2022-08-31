<div align="center">
<a href="https://github.com/RetoWang/lightweight-human-pose-estimation.pytorch">
<img src="logo.png" width="20%">
</a>
</div>


# human keypoints detection
**This task is a part of the "smart operation room" project. It can identify the left and right side of the people.**


## Introduction
The smart operation room is the project that can digitise the whole processes of surgeries.
Human body keypoints detection is used to extract key points of human body and visualize right side of the patient.

## Installation
Requirements:
- Ubuntu 16.04 (or higher)
- Python 3.6 (or higher)
- PyTorch 0.4.1 ((or higher))

```bash
pip install -r requirements.txt
```

## Training

Training consists of 3 steps (given AP values for full validation dataset):

- Training from MobileNet weights. Expected AP after this step is ~38%.
- Training from weights, obtained from previous step. Expected AP after this step is ~39%. 
- Training from weights, obtained from previous step and increased number of refinement stages to 3 in network. Expected AP after this step is ~40% (for the network with 1 refinement stage, two next are discarded).

1. Download pre-trained MobileNet v1 weights mobilenet_sgd_68.848.pth.tar from: https://github.com/marvis/pytorch-mobilenet (sgd option). If this doesn't work, download from GoogleDrive.

2. Convert train annotations in internal format. Run python scripts/prepare_train_labels.py --labels <COCO_HOME>/annotations/person_keypoints_train2017.json. It will produce prepared_train_annotation.pkl with converted in internal format annotations.

    [OPTIONAL] For fast validation it is recommended to make subset of validation dataset. Run python scripts/make_val_subset.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json. It will produce val_subset.json with annotations just for 250 random images (out of 5000).

3. To train from MobileNet weights, run python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/mobilenet_sgd_68.848.pth.tar --from-mobilenet

4. Next, to train from checkpoint from previous step, run python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/checkpoint_iter_420000.pth --weights-only

5. Finally, to train from checkpoint from previous step and 3 refinement stages in network, run python train.py --train-images-folder <COCO_HOME>/train2017/ --prepared-train-labels prepared_train_annotation.pkl --val-labels val_subset.json --val-images-folder <COCO_HOME>/val2017/ --checkpoint-path <path_to>/checkpoint_iter_280000.pth --weights-only --num-refinement-stages 3. We took checkpoint after 370000 iterations as the final one.

We did not perform the best checkpoint selection at any step, so similar result may be achieved after less number of iterations.

**Known issue**

We observe this error with maximum number of open files (ulimit -n) equals to 1024:

      File "train.py", line 164, in <module>
        args.log_after, args.val_labels, args.val_images_folder, args.val_output_name, args.checkpoint_after, args.val_after)
      File "train.py", line 77, in train
        for _, batch_data in enumerate(train_loader):
      File "/<path>/python3.6/site-packages/torch/utils/data/dataloader.py", line 330, in __next__
        idx, batch = self._get_batch()
      File "/<path>/python3.6/site-packages/torch/utils/data/dataloader.py", line 309, in _get_batch
        return self.data_queue.get()
      File "/<path>/python3.6/multiprocessing/queues.py", line 337, in get
        return _ForkingPickler.loads(res)
      File "/<path>/python3.6/site-packages/torch/multiprocessing/reductions.py", line 151, in rebuild_storage_fd
        fd = df.detach()
      File "/<path>/python3.6/multiprocessing/resource_sharer.py", line 58, in detach
        return reduction.recv_handle(conn)
      File "/<path>/python3.6/multiprocessing/reduction.py", line 182, in recv_handle
        return recvfds(s, 1)[0]
      File "/<path>/python3.6/multiprocessing/reduction.py", line 161, in recvfds
        len(ancdata))
    RuntimeError: received 0 items of ancdata

To get rid of it, increase the limit to bigger number, e.g. 65536, run in the terminal: ulimit -n 65536

## Validation
    Run python val.py --labels <COCO_HOME>/annotations/person_keypoints_val2017.json --images-folder <COCO_HOME>/val2017 --checkpoint-path <CHECKPOINT>

## Tutorial
directly use the demo.py by changing the root of the video source in videos folder
