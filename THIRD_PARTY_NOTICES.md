# Third-party code and assets

The original license files are included beside each vendored component. Upstream code is unchanged; file checksums are verified before evaluation or baseline execution.

| Component | Pinned revision | Included use | License file |
| --- | --- | --- | --- |
| [TrackEval](https://github.com/JonathonLuiten/TrackEval) | `12c8791b303e0a0b50f753af204249e622d0281a` | Identity, CLEAR and HOTA evaluation | `third_party/TrackEval-12c8791b303e0a0b50f753af204249e622d0281a/LICENSE` (MIT) |
| [RoundaboutHD evaluation](https://github.com/siri-rouser/multi_camera_tracking_labelling_tool) | `8a8d416b65486a43a2d2641876f585095912f5ac` | Original evaluation audit | `third_party/roundabouthd_evaluation_8a8d416/LICENSE` (MIT) |
| [Fast Online MTMCT](https://github.com/kamkyu94/Fast_Online_MTMCT) | `f920c396aaa820e231bd3fb7ec12d563ce75c396` | Published association adaptation | `third_party/fast_online_mtmct_f920c396/LICENSE` (MIT) |

The baseline adapter extracts the pinned upstream association logic and uses the shared local tracks and available vehicle descriptors. It is not the complete original published pipeline.

Python dependencies, including Ultralytics/ByteTrack, OpenVINO, NumPy, SciPy, OpenCV and PyTorch, are installed separately and retain their upstream terms. They are not vendored here. Pretrained model weights and dataset assets are also obtained separately. The model URL and checksum are recorded in `configs/vehicle_reid.json`.

BoxMOT and its unrelated example assets were excluded from this publication package because the reported confirmation-delay comparison and published association baseline do not use them.
