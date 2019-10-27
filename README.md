# ros_patch_match

Implementation of PatchMatch algorithm (CPU and GPU version) as a ROS service. The GPU version is much faster than CPU version

Usage:

1. Run ROS master: `roscore`.
2. Start ROS PatchMatch service: `rosrun ros_patch_match ros_patch_match_node`.
3. Call service from client. The parameters of service client are explained in the `srv` folder.

The output is reconstructed image nad ann image from the PatchMatch algorithm.

## Related publications
BibTeX:
```
@inproceedings{Barnes2009PatchMatchAR,
  title={PatchMatch: a randomized correspondence algorithm for structural image editing},
  author={Connelly Barnes and Eli Shechtman and Adam Finkelstein and Dan B. Goldman},
  booktitle={SIGGRAPH 2009},
  year={2009}
}
    
@article{Simakov2008SummarizingVD,
  title={Summarizing visual data using bidirectional similarity},
  author={Denis Simakov and Yaron Caspi and Eli Shechtman and Michal Irani},
  journal={2008 IEEE Conference on Computer Vision and Pattern Recognition},
  year={2008},
  pages={1-8}
}
```
