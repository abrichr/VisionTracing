import numpy as np
import cv2
import scipy
import scipy.spatial
import scipy.linalg
from scipy.optimize import linear_sum_assignment
import random
import pickle

from cython_bbox import bbox_overlaps

from tqdm import tqdm

import os
import glob
import sys
import time
from enum import Enum


# https://github.com/facebookresearch/detectron2/issues/754#issuecomment-579463185
JOINT_NAMES = [
  "nose",
  "left_eye", "right_eye",
  "left_ear", "right_ear",
  "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow",
  "left_wrist", "right_wrist",
  "left_hip", "right_hip",
  "left_knee", "right_knee",
  "left_ankle", "right_ankle"
]

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
  1: 3.8415,
  2: 5.9915,
  3: 7.8147,
  4: 9.4877,
  5: 11.070,
  6: 12.592,
  7: 14.067,
  8: 15.507,
  9: 16.919}


# kalman filter

class KalmanFilter:
  """
  A simple Kalman filter for tracking bounding boxes in image space.

  The 8-dimensional state space

    x, y, a, h, vx, vy, va, vh

  contains the bounding box center position (x, y), aspect ratio a, height h,
  and their respective velocities.

  Object motion follows a constant velocity model. The bounding box location
  (x, y, a, h) is taken as direct observation of the state space (linear
  observation model).

  """

  def __init__(self):
    ndim, dt = 4, 1.

    # Create Kalman filter model matrices.
    self._motion_mat = np.eye(2 * ndim, 2 * ndim)
    for i in range(ndim):
      self._motion_mat[i, ndim + i] = dt
    self._update_mat = np.eye(ndim, 2 * ndim)

    # Motion and observation uncertainty are chosen relative to the current
    # state estimate. These weights control the amount of uncertainty in
    # the model. This is a bit hacky.
    self._std_weight_position = 1. / 20
    self._std_weight_velocity = 1. / 160

  def initiate(self, measurement):
    """Create track from unassociated measurement.

    Parameters
    ----------
    measurement : ndarray
      Bounding box coordinates (x, y, a, h) with center position (x, y),
      aspect ratio a, and height h.

    Returns
    -------
    (ndarray, ndarray)
      Returns the mean vector (8 dimensional) and covariance matrix (8x8
      dimensional) of the new track. Unobserved velocities are initialized
      to 0 mean.

    """
    mean_pos = measurement
    mean_vel = np.zeros_like(mean_pos)
    mean = np.r_[mean_pos, mean_vel]

    std = [
      2 * self._std_weight_position * measurement[3],
      2 * self._std_weight_position * measurement[3],
      1e-2,
      2 * self._std_weight_position * measurement[3],
      10 * self._std_weight_velocity * measurement[3],
      10 * self._std_weight_velocity * measurement[3],
      1e-5,
      10 * self._std_weight_velocity * measurement[3]]
    covariance = np.diag(np.square(std))
    return mean, covariance

  def predict(self, mean, covariance):
    """Run Kalman filter prediction step.

    Parameters
    ----------
    mean : ndarray
      The 8 dimensional mean vector of the object state at the previous
      time step.
    covariance : ndarray
      The 8x8 dimensional covariance matrix of the object state at the
      previous time step.

    Returns
    -------
    (ndarray, ndarray)
      Returns the mean vector and covariance matrix of the predicted
      state. Unobserved velocities are initialized to 0 mean.

    """
    std_pos = [
      self._std_weight_position * mean[3],
      self._std_weight_position * mean[3],
      1e-2,
      self._std_weight_position * mean[3]]
    std_vel = [
      self._std_weight_velocity * mean[3],
      self._std_weight_velocity * mean[3],
      1e-5,
      self._std_weight_velocity * mean[3]]
    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

    mean = np.dot(self._motion_mat, mean)
    covariance = np.linalg.multi_dot((
      self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

    return mean, covariance

  def project(self, mean, covariance):
    """Project state distribution to measurement space.

    Parameters
    ----------
    mean : ndarray
      The state's mean vector (8 dimensional array).
    covariance : ndarray
      The state's covariance matrix (8x8 dimensional).

    Returns
    -------
    (ndarray, ndarray)
      Returns the projected mean and covariance matrix of the given state
      estimate.

    """
    std = [
      self._std_weight_position * mean[3],
      self._std_weight_position * mean[3],
      1e-1,
      self._std_weight_position * mean[3]]
    innovation_cov = np.diag(np.square(std))

    mean = np.dot(self._update_mat, mean)
    covariance = np.linalg.multi_dot((
      self._update_mat, covariance, self._update_mat.T))
    return mean, covariance + innovation_cov

  def update(self, mean, covariance, measurement):
    """Run Kalman filter correction step.

    Parameters
    ----------
    mean : ndarray
      The predicted state's mean vector (8 dimensional).
    covariance : ndarray
      The state's covariance matrix (8x8 dimensional).
    measurement : ndarray
      The 4 dimensional measurement vector (x, y, a, h), where (x, y)
      is the center position, a the aspect ratio, and h the height of the
      bounding box.

    Returns
    -------
    (ndarray, ndarray)
      Returns the measurement-corrected state distribution.

    """
    projected_mean, projected_cov = self.project(mean, covariance)

    chol_factor, lower = scipy.linalg.cho_factor(
      projected_cov, lower=True, check_finite=False)
    kalman_gain = scipy.linalg.cho_solve(
      (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
      check_finite=False).T
    innovation = measurement - projected_mean

    new_mean = mean + np.dot(innovation, kalman_gain.T)
    new_covariance = covariance - np.linalg.multi_dot((
      kalman_gain, projected_cov, kalman_gain.T))
    return new_mean, new_covariance

  def gating_distance(self, mean, covariance, measurements,
            only_position=False):
    """Compute gating distance between state distribution and measurements.

    A suitable distance threshold can be obtained from `chi2inv95`. If
    `only_position` is False, the chi-square distribution has 4 degrees of
    freedom, otherwise 2.

    Parameters
    ----------
    mean : ndarray
      Mean vector over the state distribution (8 dimensional).
    covariance : ndarray
      Covariance of the state distribution (8x8 dimensional).
    measurements : ndarray
      An Nx4 dimensional matrix of N measurements, each in
      format (x, y, a, h) where (x, y) is the bounding box center
      position, a the aspect ratio, and h the height.
    only_position : Optional[bool]
      If True, distance computation is done with respect to the bounding
      box center position only.

    Returns
    -------
    ndarray
      Returns an array of length N, where the i-th element contains the
      squared Mahalanobis distance between (mean, covariance) and
      `measurements[i]`.

    """
    mean, covariance = self.project(mean, covariance)
    if only_position:
      mean, covariance = mean[:2], covariance[:2, :2]
      measurements = measurements[:, :2]

    cholesky_factor = np.linalg.cholesky(covariance)
    d = measurements - mean
    z = scipy.linalg.solve_triangular(
      cholesky_factor, d.T, lower=True, check_finite=False,
      overwrite_b=True)
    squared_maha = np.sum(z * z, axis=0)
    return squared_maha

# tracking helpers

### Helpers for the track class, should be moved to a utils file later

class CostTypes(Enum):
  """Enum for cost types used in distance matrix calculation."""
  IOU = 1 # intersection over union
  KEYPOINT_THRESHOLDING = 2 # keypoint algorithm in function pck_distance
  PROXIMITY_30_PERCENT = 3 # 
  MIN_OF_5_TRAILING_BOXES = 4
  MIN_OF_50_TRAILING_BOXES = 5

def xyxy_to_xyah(bbox):
  """Converts a bounding box from x1y1x2y2 to xyah format.

  xyah format is (x_center, y_center, aspect_ratio, height) where
  the aspect ratio is width / height.

  Args:
    bbox: np.ndarray; A length 4 numpy array in x1y1x2y2 format.

  Returns:
    np.ndarray; A length 4 array containing the same box in xyah format.
  """
  ret = bbox.copy()
  ret[2:] -= ret[:2]
  ret[:2] += ret[2:] / 2
  ret[2] /= ret[3]
  return ret

def xyah_to_xyxy(bbox):
  """Converts a bounding box from xyah to x1y1x2y2 format.

  xyah format is (x_center, y_center, aspect_ratio, height) where
  the aspect ratio is width / height.

  Args:
    bbox: np.ndarray; A length 4 numpy array in xyah format.

  Returns:
    np.ndarray; A length 4 array containing the same box in x1y1x2y2 format.
  """
  ret = bbox.copy()
  ret[2] *= ret[3]
  ret[:2] -= ret[2:] / 2
  ret[2:] += ret[:2]
  return ret

def get_predictions_from_active_tracks(tracks, t, max_age=3, use_kf=True):
  """Gets predictions for frame t from the tracks of age less than max_age.

  Args:
    tracks: list[Track]; A list of all tracks in the scene
    t: int; The frame of the desired predictions
    max_age: int; The maximum number of frames for which the track has not been
      updated.
    use_kf: bool; True iff the kalman filter is used in predicted box locations.
  
  Returns:
    bbox_preds: nx4 np.ndarray; The predictions from each track for frame t.
    kpt_preds: nx17x3 np.ndarray; The predicted keypoints for each track for frame t.
    idxs: list[int]; The index of the predictions in the tracks list.
      predictions[i] comes from the track tracks[idxs[i]].
  """
  bbox_preds = []
  kpt_preds = []
  idxs = []

  if not tracks:
    return [], [], []
  
  for i in range(len(tracks)):
    track = tracks[i]
    if track.is_recently_updated(t, max_age):
      bbox_preds.append(track.predict(t, use_kf=False))
      kpt_preds.append(track.predict(t, kpt=True))
      idxs.append(i)
  
  bbox_preds = np.stack(bbox_preds, axis=0)
  kpt_preds = np.stack(kpt_preds, axis=0)
  return bbox_preds, kpt_preds, idxs

### End helpers


# Design doc: 
# https://docs.google.com/document/d/1ATAqPbnDgWFUZsYsIF_K4sSXrCuYJx6n19Lr1sQW-98

# Question: Retroactively infer between short gaps with kalman filter? could be
# worthwhile
class Track:
  """A track of bounding boxes and keyframes for a subject in a video.

  A track storing the location of a subject in a video. The track can also
  predict the future location of the subject at a future time using a Kalman
  filter. Throws a ValueError if asked to give the location of the subject
  at a frame before initialization. Frames are 0-indexed.

  Attributes:
    all_boxes: list of np.ndarray; A list containing a numpy array of the
     detected bounding boxes for each frame in x1y1x2y2 format.
    all_keypoints: list of np.ndarray; A list containing a numpy array of the
      estimated keypoints for each of the 17 joints. See JOINT_NAMES for details.
    kalman_filter: KalmanFilter; A kalman filter for predicting the bounding box
      position in future frames. The filter does not store the estimates.
    _mean: np.ndarray; The 8-dimensional mean state estimate from the kalman filter.
      The estimate is of format (x, y, a, h, xv, yv, av, hv) where the *v's are the
      velocity estimates.
    _cov: np.ndarray; The 8x8 covariance matrix for the kalman filter's mean estimates.
    idx_list: list[int]; A list of the index for the subject in each frame. If 
      the subject was not detected in that frame, returns -1. Starts at the
      frame where the track was initialized.
    start_frame: int; The frame where the track was initialized.
    most_recent: int; The most recent frame where the track was updated.
  """

  def __init__(self, all_boxes, all_keypoints, frame_idx, t):
    self.all_boxes = all_boxes
    self.all_keypoints = all_keypoints
    self.kalman_filter = KalmanFilter()
    bbox = all_boxes[t][frame_idx]
    bbox_xyah = xyxy_to_xyah(bbox)
    self._mean, self._cov = self.kalman_filter.initiate(bbox_xyah)
    self.idx_list = [frame_idx]
    self.start_frame = t
    self.most_recent = t

  def is_recently_updated(self, t, max_age=3):
    """Returns True if this track has been updated in the last x frames.

    Args:
      t: int; The current frame index
      max_age: int; The amount of frames to consider the update recent.
    
    Returns:
      bool; True iff the track has been updated in the last x frames.
    """
    return abs(t - self.most_recent) < max_age

  def predict(self, t, kpt=False, keep_vel=False, use_kf=True):
    """Predict the subject's position at frame t.

    Uses the Kalman filter to predict the box location at frame t.
    Gives the most recent set of keypoints if kpt is true.

    Args:
      t: int; The desired frame index
      kpt: bool; True iff we want to get the keypoints instead of the bounding box.
      keep_vel: bool; True iff we want to get the full 8 dimensional state.
      use_kf: bool; True iff we use the kalman filter to predict
    
    Returns:
      np.ndarray; A numpy array storing the bounding box in x1y1x2y2 format or
        the keypoint, depending on kpt.
    
    Raises:
      ValueError: Location at a frame prior to the start of the track was requested.
    """
    if t <= self.most_recent:
      return get_val(t, kpt=False)
    else:
      if kpt:
        return self.all_keypoints[self.most_recent][self.idx_list[self.most_recent - self.start_frame]]
      elif not use_kf:
        return self.all_boxes[self.most_recent][self.idx_list[self.most_recent - self.start_frame]]
      # Note: This is not efficient for t far beyond self.most_recent
      t_temp = self.most_recent
      mean, cov = self._mean, self._cov
      if t > t_temp + 250:
        # catch for excessive predictions, likely from reId failing
        # It can still predict, but would only predict this far in advance from an error.
        raise ValueError(f"Track prediction {t - self.start_frame} frames ahead requested on frame {t}. Frame predictions of 250+ frames are not supported.")
      
      while t_temp < t:
        mean, cov = self.kalman_filter.predict(mean, cov)
        t_temp += 1
      
      if not keep_vel:
        bbox_xyah, _ = self.kalman_filter.project(mean, cov)
        bbox_xyxy = xyah_to_xyxy(bbox_xyah)
        return bbox_xyxy
      else:
        return mean

  
  def get_val(self, t, kpt=False):
    """Returns the value of the bounding box at frame t.

    Args:
      t: int; The desired frame index

    Returns:
      np.ndarray; A numpy array of shape (4, ) storing the bounding box in
        x1y1x2y2 format.
    
    Raises:
      ValueError: Location at a frame prior to the start of the track was requested.
    """
    if t < self.start_frame:
      raise ValueError(f"Track starting on frame {self.start_frame} queried for value at frame {t}.")
    elif t <= self.most_recent:
      idx = self.idx_list[t - self.start_frame]
      if idx == -1:
        if kpt:
          ret = np.empty((17, 3), dtype=np.float64)
          ret[:] = np.nan
          return ret
        else:
          return np.array([np.nan, np.nan, np.nan, np.nan])
      else:
        if kpt:
          return self.all_keypoints[t][idx]
        else:
          return self.all_boxes[t][idx]
    else:
      raise ValueError(f"Track does not have any value at frame {t}.")
  
  def get_values_in_last_n(self, t, n):
    """Returns the values of the track in the last n frames from frame t.

    Args:
      t: int; The frame in question
      n: int; The number of frames in the past to look back
    
    Returns:
      nx4 np.ndarray; The values of the track in the last n frames
    """
    ret = np.zeros((n, 4))
    for i in range(n):
      frame_id = t - n + i
      if frame_id < self.start_frame:
        ret[i] = np.nan
      elif frame_id > self.most_recent:
        ret[i] = np.nan
      else:
        ret[i] = self.get_val(frame_id)
    
    return ret



  def update(self, box_idx, t, filter_cutoff=5):
    """Updates the track to contain all_boxes[box_idx] at frame t.

    Updates the stored memory of the track and also updates the kalman filter
    using the value. If the new frame is more than `filter_cutoff` frames after
    the last update to the track, we will re-initialize the kalman filter based
    on the new observation, as the constant velocity process model is not accurate
    on longer time scales without continuous observations.

    Args:
      box_idx: int; The index of the detection corresponding to this track at frame t.
      t: int; The index of the frame.
      filter_cutoff: int; The number of frames ahead of a recent update before which
        the kalman filter is re-initialized on this new observation.
    
    Raises:
      ValueError: Track updated at frame where value is already set.
    """
    # Update the idx list
    if t <= self.most_recent:
      raise ValueError(f"Attempted to update track updated at frame {self.most_recent} with a previous value at frame {t}.")
    elif t == self.most_recent + 1:
      self.idx_list.append(box_idx)
    else:
      len_to_add = t - 1 - self.most_recent
      extra = [-1] * len_to_add
      self.idx_list.extend(extra)
      self.idx_list.append(box_idx)
    
    
    # Get bounding box and convert it
    bbox_xyxy = self.all_boxes[t][box_idx]
    bbox_xyah = xyxy_to_xyah(bbox_xyxy)

    # re-initialize filter if too many frames have passed
    if t > self.most_recent + filter_cutoff:
      self._mean, self._cov = self.kalman_filter.initiate(bbox_xyah)
      return
    
    # Update the kalman filter
    pred = self.predict(t, keep_vel=True)
    self._mean, self._cov = self.kalman_filter.update(pred, self._cov, bbox_xyah)

    self.most_recent = t

  def get_full_track(self):
    """Get a track of bounding boxes of length equal to the video length.

    For frames where the subject is not detected, returns np.nan for each
    bounding box coordinate.

    Returns:
      np.ndarray; nx4 numpy array containing the bounding box for the subject
        at each of the n frames in format x1y1x2y2.
    """
    n = len(self.all_boxes)
    full_track = np.zeros((n, 4))
    for i in range(n):
      if i < self.start_frame: # before first detection
        full_track[i] = [np.nan] * 4
      elif i <= self.most_recent: # in known region
        full_track[i] = self.all_boxes[i][self.idx_list[i - self.start_frame]]
      else: # after last detection 
        full_track[i] = [np.nan] * 4
    
    return full_track

# tracking code

# https://github.com/facebookresearch/DetectAndTrack/blob/d66734498a4331cd6fde87d8269499b8577a2842/lib/core/tracking_engine.py#L106
def compute_pairwise_iou(a, b):
  """Computes the pairwise intersection over union for the arrays of boxes a and b.

  Args:
    a: np.ndarray; Array of N boxes in format x1y1x2y2.
    b: np.ndarray; Array of M boxes in format x1y1x2y2.
  
  Returns:
    np.ndarray; A NxM array where the entry at (i, j) is the intersection over
      union of box i from a, and box j from b.
  """

  C = 1 - bbox_overlaps(
    np.ascontiguousarray(a, dtype=np.float64),
    np.ascontiguousarray(b, dtype=np.float64),
  )
  return C

# Based on
# https://github.com/facebookresearch/DetectAndTrack/blob/d66734498a4331cd6fde87d8269499b8577a2842/lib/utils/keypoints.py#L266
def compute_head_size(kps, kpt_names):
    """Estimates the head size of the subject based on the named keypoints.

    This function expects the keypoints to be the 17 used by detectron2, in
    particular including nose and shoulder estimates. This is a very rough estimate.
    Based on
    https://github.com/leonid-pishchulin/poseval/blob/954d8d84f459e942a185f835fc2a0fbdee5ce354/py/eval_helpers.py#L73  # noQA

    Args:
      kps: np.ndarray; The keypoints for this subject.
      kpt_names: list[string]; The ordered list of keypoint names.
    
    Returns:
      float; An estimate of the head size of the subject
    """
    nose = kps[:2, kpt_names.index('nose')]
    shoulder = kps[:2, kpt_names.index('left_shoulder')]
    # 0.6 x hypotenuse of the head, but don't have those kpts

    # The above is from detectrons previous heuristic where they had access to
    # the top and bottom of the head, which we do not have access to.
    # Thus, I chose two keypoints which have vertical displacement so that
    # the estimate will not be extremely small when we have a profile view
    # of the subject.

    return .4 * np.linalg.norm(nose - shoulder) + 1  # to avoid 0s


def pck_distance(kps_a, kps_b, kpt_names=JOINT_NAMES, dist_thresh=0.5):
    """Compute distance between the 2 keypoints, where each is represented
    as a 3x17 or 4x17 np.ndarray.

      Computes the proportion of keypoints which are a threshold away from the
    corresponding keypoint in the outcome. The threshold is based on the size
    of the individual's head.

    Args:
      kps_a: np.ndarray; The keypoints of subject A.
      kps_b: np.ndarray; The keypoints of subject B.
      kpt_names: list[string]; The ordered names of the keypoints.
      dist_thresh: float; The number of 'head_sizes' away from the previous
        corresponding keypoint to be considered accurate.
    
    Returns:
      float; The PCK (Percentage of Correct Keypoints) distance between the two subjects.
    """
    # This code expects 3x17 instead of 17x3
    kps_a = np.swapaxes(kps_a, 0, 1) 
    kps_b = np.swapaxes(kps_b, 0, 1)
    # compute head size as heuristic scale for point separation
    head_size = compute_head_size(kps_a, kpt_names)
    # distance between all points
    normed_dist = np.linalg.norm(kps_a[:2] - kps_b[:2], axis=0) / head_size
    match = normed_dist < dist_thresh
    pck = np.sum(match) / match.size
    pck_dist = 1.0 - pck
    return pck_dist


# https://github.com/facebookresearch/DetectAndTrack/blob/d66734498a4331cd6fde87d8269499b8577a2842/lib/core/tracking_engine.py#L114
def compute_pairwise_kpt_distance(a, b, kpt_names=JOINT_NAMES):
  """Computes a distance matrix between two lists of keypoints based on PCK.

  This tries to recreate the assignGT function from the evaluation code_dir
  https://github.com/leonid-pishchulin/poseval/blob/954d8d84f459e942a185f835fc2a0fbdee5ce354/py/eval_helpers.py#L423  # noQA
  Main points:
      prToGT is the prediction_to_gt output that I want to recreate
      Essentially it represents a form of PCK metric

  Args:
     a, b (poses): Two sets of poses to match
     Each "poses" is represented as a list of 3x17 or 4x17 np.ndarray
    
  Returns:
    np.ndarray; The pairwise keypoint distance between the subjects in A and B.
  """
  res = np.zeros((len(a), len(b)))
  for i in range(len(a)):
    for j in range(len(b)):
      res[i, j] = pck_distance(a[i], b[j], kpt_names)
  return res

def compute_pairwise_proximity(prev_boxes, cur_boxes, ratio=1.0):
  """Computes a rough proximity metric, if boxes are within ratio of the box.

  Uses the min of the width and height
  """
  # Not particularly efficient, could be improved if it is a bottleneck
  ret = np.ones((len(prev_boxes), len(cur_boxes)))
  for i in range(len(prev_boxes)):
    prev_cent = (prev_boxes[i][:2] + prev_boxes[i][2:]) / 2
    prev_size = abs(min(prev_boxes[i][2] - prev_boxes[i][0], prev_boxes[i][3] - prev_boxes[i][1]))
    for j in range(len(cur_boxes)):
      cur_cent = (cur_boxes[j][:2] + cur_boxes[j][2:]) / 2
      if np.linalg.norm(prev_cent - cur_cent) < prev_size * ratio:
        ret[i][j] = 0
  
  return ret

def min_of_trailing_boxes_iou(tracks, t, cur_boxes, n=5, max_age=5):
  """Computes a distance matrix based on the IOU distance of the n most recent values.
  """
  temp_tracks = []
  for track in tracks:
    if track.is_recently_updated(t, max_age):
      temp_tracks.append(track)
  tracks = temp_tracks
  n_most_recent = np.array([track.get_values_in_last_n(t, n) for track in tracks])
  n_most_recent = n_most_recent.transpose(1, 0, 2)
  Cs = []
  
  # Get cost matrix for each frame
  for i in range(n):
    Cs.append(compute_pairwise_iou(n_most_recent[i], cur_boxes))

  # Min each, or 1 if all is nan
  C = np.ones_like(Cs[0])
  for i in range(C.shape[0]):
    for j in range(C.shape[1]):
      for k in range(n):
        val = Cs[k][i][j]
        if val != np.nan:
          C[i][j] = min(val, C[i][j])
  
  return C

# based on facebook research detect and track compute_distance_matrix
def compute_distance_matrix(
    prev_boxes, prev_keypoints,
    cur_boxes, cur_kpt, 
    tracks, t, max_age=3,
    cost_types=[CostTypes.IOU], cost_weights=[1.0],
):
  """Computes a distance matrix using the tracks and current
  boxes and keypoints.

  Uses the tracks to get the list of previous boxes and keypoints from recently
  updated tracks. 
  """
  assert(len(cost_weights) == len(cost_types))
  all_Cs = []
  for cost_type, cost_weight in zip(cost_types, cost_weights):
    if cost_weight == 0:
      continue
    if cost_type == CostTypes.IOU:
      all_Cs.append(compute_pairwise_iou(prev_boxes, cur_boxes))
    elif cost_type == CostTypes.KEYPOINT_THRESHOLDING:
      all_Cs.append(compute_pairwise_kpt_distance(
        prev_keypoints, cur_kpt))
    elif cost_type == CostTypes.PROXIMITY_30_PERCENT:
      all_Cs.append(compute_pairwise_proximity(prev_boxes, cur_boxes, ratio=1.3))
    elif cost_type == CostTypes.MIN_OF_5_TRAILING_BOXES:
      all_Cs.append(min_of_trailing_boxes_iou(tracks, t, cur_boxes, n=5, max_age=max_age))
    elif cost_type == CostTypes.MIN_OF_50_TRAILING_BOXES:
      all_Cs.append(min_of_trailing_boxes_iou(tracks, t, cur_boxes, n=50, max_age=50))
    else:
      raise NotImplementedError('Unknown cost type {}'.format(cost_type))
    all_Cs[-1] *= cost_weight
  return np.sum(np.stack(all_Cs, axis=0), axis=0)

# based on
# https://github.com/facebookresearch/DetectAndTrack/blob/d66734498a4331cd6fde87d8269499b8577a2842/lib/core/tracking_engine.py#L184
def bipartite_matching_greedy(C, max_cost=1.0):
  """
  Computes the bipartite matching between the rows and columns, given the
  cost matrix, C. If the cost is greater than or equal to max_cost, the rows and columns
  will be matched with the index -1.
  """
  C = C.copy()  # to avoid affecting the original matrix
  prev_ids = []
  cur_ids = []
  row_ids = np.arange(C.shape[0])
  col_ids = np.arange(C.shape[1])
  while C.size > 0:
    # Find the lowest cost element
    i, j = np.unravel_index(C.argmin(), C.shape)
    # If all remaining costs are greater than max_cost, then
    # set the rest of the rows/cols as unmatched and return.
    if C.min() >= max_cost:
      for row_id in row_ids:
        prev_ids.append(row_id)
        cur_ids.append(-1)
      for col_id in col_ids:
        prev_ids.append(-1)
        cur_ids.append(col_id)
      return prev_ids, cur_ids
    # Add to results and remove from the cost matrix
    row_id = row_ids[i]
    col_id = col_ids[j]
    prev_ids.append(row_id)
    cur_ids.append(col_id)
    C = np.delete(C, i, 0)
    C = np.delete(C, j, 1)
    row_ids = np.delete(row_ids, i, 0)
    col_ids = np.delete(col_ids, j, 0)
  return prev_ids, cur_ids

def compute_matches(tracks, t, max_age,
                    cur_boxes, cur_keypoints,
                     cost_types, cost_weights,
                     bipart_match_algo, C=None, track_idx=None, use_kf=True):
  """
  C (cost matrix): num_prev_boxes x num_current_boxes
  Optionally input the cost matrix, in which case you can input dummy values
  for the boxes and poses
  Returns:
      matches: A 1D np.ndarray with as many elements as boxes in current
      frame (cur_boxes). For each, there is an integer to index the previous
      frame box that it matches to, or -1 if it does not match to any previous
      box.
  """
  # If there are no tracks, just set everything as new tracks
  if not tracks:
    nboxes = cur_boxes.shape[0]
    matches = -np.ones((nboxes,), dtype=np.int32)
    return matches
  
  # matches structure keeps track of which of the current boxes matches to
  # which box in the previous frame. If any idx remains -1, it will be set
  # as a new track.
  if C is None:
    nboxes = cur_boxes.shape[0]
    matches = -np.ones((nboxes,), dtype=np.int32)
    prev_boxes, prev_keypoints, track_idx = get_predictions_from_active_tracks(tracks, t, max_age, use_kf=use_kf)
    C = compute_distance_matrix(prev_boxes, prev_keypoints,
        cur_boxes, cur_keypoints, 
        tracks, t, max_age=max_age,
        cost_types=cost_types,
        cost_weights=cost_weights)
  else:
    matches = -np.ones((C.shape[1],), dtype=np.int32)
  
  if bipart_match_algo == 'hungarian':
    prev_inds, next_inds = scipy.optimize.linear_sum_assignment(C)
  elif bipart_match_algo == 'greedy':
    prev_inds, next_inds = bipartite_matching_greedy(C)
  else:
    raise NotImplementedError('Unknown matching algo: {}'.format(bipart_match_algo))
    
  assert(len(prev_inds) == len(next_inds))
  for i in range(len(prev_inds)):
    if next_inds[i] == -1:
      # If no match was found for the track, continue
      continue
    elif prev_inds[i] == -1:
      # If no match was found for the box, leave it as -1
      matches[next_inds[i]] = -1
    else:
      matches[next_inds[i]] = track_idx[prev_inds[i]]
  return matches

def update_tracks(tracks, matches, i, all_boxes, all_keypoints):
  """Updates the tracks for frame i given the matches, and creates new tracks when necessary.

  Args:
  """
  
  print('Matches {}'.format(matches))

  for t in range(len(matches)):
    idx = matches[t]
    if idx == -1:
      # There was no previous track, so we instantiate a new track
      tracks.append(Track(all_boxes, all_keypoints, t, i))
    else:
      # Update the track with the new data
      track = tracks[idx]
      track.update(t, i)

def run_tracker(all_boxes, all_keypoints, max_age, matching_algo="greedy", 
                cost_types=[CostTypes.IOU], cost_weights=[1.0], use_kf=True):
  """Runs the full tracker on the boxes and keypoints.

  The weights should sum to 1, and should be equal to 1 iff you do not
  want those two boxes to match.

  Args:
    all_boxes: list[np.ndarray]; The bounding boxes by frame.
    all_keypoints: list[np.ndarray]; The keypoints by frame.
    max_age: int; The maximum number of frames during which a track can have no
      matches before no longer being considered for new frames.
    matching_algo: ('greedy', 'hungarian'); The matching algorithm used to match
      detections between frames
    cost_types: list[CostTypes]; The cost types used in the distance calculation.
    cost_weights: list[float]; The weights attached to the cost type with the same
      index in the distance calculation. The weights should sum to 1, and should
      be equal to 1 iff you do not want those two boxes to match.
  
  Returns:
    list[np.ndarray]: The tracks by frame. Each track is of length len(all_boxes)
      and contains the bounding box coordinates by frame. The tracks use np.nan
      when detections are not present.
  """
  n = len(all_boxes)
  tracks = []
  for i in tqdm(range(n)):
    cur_boxes = all_boxes[i]
    cur_kpts = all_keypoints[i]
    matches = compute_matches(tracks, i, max_age, cur_boxes, cur_kpts,
                              cost_types, cost_weights, matching_algo, use_kf=use_kf)
    update_tracks(tracks, matches, i, all_boxes, all_keypoints)

  arr_tracks = []
  for track in tracks:
    arr_tracks.append(track.get_full_track())
  
  return arr_tracks



def get_tracks(all_predictions):
    all_keypoints = []
    all_boxes = []
    for predictions in all_predictions:
        instances = predictions['instances'].to('cpu')
        keypoints = np.asarray(instances.pred_keypoints)
        boxes = np.asarray(instances.pred_boxes.tensor)
        all_keypoints.append(keypoints)
        all_boxes.append(boxes)
    cost_types = [
		  CostTypes.IOU,
		  CostTypes.KEYPOINT_THRESHOLDING,
		  CostTypes.PROXIMITY_30_PERCENT
    ]
    cost_weights = [
		    .8,
		    .19,
		    .01
    ]
    tracks = run_tracker(all_boxes, all_keypoints, 9, cost_types=cost_types, cost_weights=cost_weights)
    return tracks
